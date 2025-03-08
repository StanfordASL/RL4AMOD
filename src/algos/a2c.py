import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from src.misc.utils import dictsum
from src.nets.actor import GNNActor
from src.nets.critic import GNNValue
from src.algos.reb_flow_solver import solveRebFlow
import os 
from tqdm import trange
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10
#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, cfg, parser, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        self.act_dim = env.nregion
        
        self.actor = self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic = GNNValue(self.input_size, self.hidden_size)
        self.parser = parser
        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)
    
    def forward(self, obs, deterministic=False):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parser.parse_obs(obs).to(self.device)
        # actor: computes concentration parameters of a Dirichlet distribution
        action, log_prob = self.actor(x.x, x.edge_index, deterministic=deterministic)
        # critic: estimates V(s_t)
        value = self.critic(x)
        return action.squeeze(), log_prob, value

    def select_action(self, obs, deterministic=False):
        action, log_prob, value = self.forward(obs, deterministic=deterministic)
        if not deterministic:
            self.saved_actions.append(SavedAction(log_prob.squeeze(), value.squeeze()))
        return action.detach().cpu().numpy()
    
    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
       
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
  
        if self.wandb is not None:
            self.wandb.log({"Policy Loss": a_loss.item(), "Critic Loss": v_loss.item()})
    
    def learn(self, cfg):

        train_episodes = cfg.model.max_episodes #set max number of training episodes
        T = cfg.model.max_steps #set episode length
        epochs = trange(train_episodes)     # epoch iterator
        best_reward = -np.inf   # set best reward
        self.train()   # set model in train mode
        
        for i_episode in epochs:
    
            # Initialize the reward
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_rebalanced_vehicles = 0
            # Reset the environment
            obs, rew = self.env.reset()  # initialize environment
           
            self.rewards.append(rew)
            for step in range(T):
                # take matching step (Step 1 in paper)
                action_rl = self.select_action(obs)

                desiredAcc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time + 1))for i in range(len(self.env.region))}
                # solve minimum rebalancing distance problem (Step 3 in paper)

                reb_action = solveRebFlow(
                    self.env,
                    self.env.cfg.directory,
                    desiredAcc,
                    self.cplexpath,
                )
                obs, rew, done, info = self.env.step(reb_action=reb_action)
               
                self.rewards.append(rew)
                # track performance over episode
                episode_reward += rew
                episode_served_demand += info["profit"]
                episode_rebalancing_cost += info["rebalancing_cost"]
                # stop episode if terminating conditions are met
                if done:
                    break
            
            # perform on-policy backprop
            self.training_step()

            # Send current statistics to screen
            epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
            
            if self.wandb is not None:
                self.wandb.log({"Reward": np.mean(episode_reward), "Served Demand": np.mean(episode_served_demand), "Rebalancing Cost": np.mean(episode_rebalancing_cost), "Episode": i_episode})
            
            self.save_checkpoint(
                path=f"ckpt/{cfg.model.checkpoint_path}.pth"
            )
            if episode_reward > best_reward: 
                best_reward = episode_reward
                self.save_checkpoint(
                    path=f"ckpt/{cfg.model.checkpoint_path}_best.pth"
                )

    def test(self, test_episodes, env, verbose = True):
        sim = env.cfg.name
        if sim == "sumo":
            # traci.close(wait=False)
            os.makedirs(f'saved_files/sumo_output/{env.cfg.city}/', exist_ok=True)
            matching_steps = int(env.cfg.matching_tstep * 60 / env.cfg.sumo_tstep)  # sumo steps between each matching
            if env.scenario.is_meso:
                matching_steps -= 1

            sumo_cmd = [
                "sumo", "--no-internal-links", "-c", env.cfg.sumocfg_file,
                "--step-length", str(env.cfg.sumo_tstep),
                "--device.taxi.dispatch-algorithm", "traci",
                "--summary-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.summary.xml",
                "--tripinfo-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.tripinfo.xml",
                "--tripinfo-output.write-unfinished", "true",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", "10",
                "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(env.cfg.sumocfg_file), "SUMO configuration file not found!"
        if verbose:
            epochs = trange(test_episodes)  # epoch iterator
        else: 
            epochs = range(test_episodes)
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rebalanced_vehicles = []
        episode_actions = []
        episode_inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_rebalancing_veh = 0
            done = False
            if sim =='sumo':
                traci.start(sumo_cmd)
            obs, rew = env.reset()  # initialize environment
            eps_reward += rew
            eps_served_demand += rew
            actions = []
            inflow = np.zeros(len(env.region))
            while not done:
                
                action_rl = self.select_action(obs, deterministic=True)
                actions.append(action_rl)
                desiredAcc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(self.env.region))
                }
                reb_action = solveRebFlow(
                    self.env,
                    self.env.cfg.directory,
                    desiredAcc,
                    self.cplexpath,
                )
                obs, rew, done, info = env.step(reb_action=reb_action)
               
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]
                
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                #eps_rebalancing_veh += info["rebalanced_vehicles"]

            if verbose:
                epochs.set_description(
                    f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}"
                )
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_actions.append(np.mean(actions, axis=0))
            episode_inflows.append(inflow)
            #episode_rebalanced_vehicles.append(eps_rebalancing_veh)

        return (
            episode_reward,
            episode_served_demand,
            episode_rebalancing_cost,
            episode_inflows,
        )

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-4)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-4)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)
