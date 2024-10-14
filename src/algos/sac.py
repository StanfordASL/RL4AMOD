import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.nets.actor import GNNActor, GNNActorLSTM
from src.nets.critic import GNNCritic, GNNCriticLSTM
import random
from tqdm import trange
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import pickle

class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class ReplayData:
    """
    Replay buffer for SAC agents
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        self.data_list.append(PairData(data1.edge_index, data1.x, torch.as_tensor(
            reward), torch.as_tensor(action), data2.edge_index, data2.x))
        self.rewards.append(reward)
    
    def create_dataset(self, edge_index, memory_path, rew_scale, size=60000):
        w = open(f"data/{memory_path}.pkl", "rb")
        data = pickle.load(w)
   
        (state_batch, action_batch, reward_batch, next_state_batch) = (
            data["state"],
            data["action"],
            rew_scale * data["reward"],
            data["next_state"],
        )
        for i in range(len(state_batch)):
            self.data_list.append(
                PairData(
                    edge_index,
                    state_batch[i],
                    reward_batch[i],
                    action_batch[i],
                    edge_index,
                    next_state_batch[i],
                )
            )

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32):
        data = random.sample(self.data_list, batch_size)
 
        return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

#########################################
############## A2C AGENT ################
#########################################
class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        cfg, 
        parser,
        device=torch.device("cpu"),
    ):
        super(SAC, self).__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item(),
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion

        # SAC parameters
        self.alpha = cfg.alpha
        self.polyak = 0.995
        self.env = env
        self.BATCH_SIZE = cfg.batch_size
        self.p_lr = cfg.p_lr
        self.q_lr = cfg.q_lr
        self.gamma = 0.99
        self.use_automatic_entropy_tuning = cfg.auto_entropy
        self.clip = cfg.clip
        self.use_LSTM = cfg.use_LSTM
        self.parser = parser

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.agent_name = cfg.agent_name
        self.step = 0
        self.nodes = env.nregion

        self.replay_buffer = ReplayData(device=device)
        # nnets
  
        if self.use_LSTM:
            self.actor = GNNActorLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic1 = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic2 = GNNCriticLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)
        else:
            self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic1 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
            self.critic2 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

        #if min_q_weigth in cfg set it 
        if hasattr(cfg, 'min_q_weight'):
            self.min_q_weight = cfg.min_q_weight
        if hasattr(cfg, 'temp'):
            self.temp = cfg.temp
        if hasattr(cfg, 'num_random'):
            self.num_random = cfg.num_random
    def select_action(self, data, deterministic=True):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def compute_loss_q(self, data, conservative=False):
        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.nodes),
        )

        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)

        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        if conservative:
            batch_size = action_batch.shape[0]
            action_dim = action_batch.shape[-1]

            (
                random_log_prob,
                current_log,
                next_log,
                q1_rand,
                q2_rand,
                q1_current,
                q2_current,
                q1_next,
                q2_next,
            ) = self._get_action_and_values(data, 10, batch_size, action_dim)

            """Importance sampled version"""
            cat_q1 = torch.cat(
                [
                    q1_rand - random_log_prob.detach(),
                    q1_next - next_log.detach(),
                    q1_current - current_log.detach(),
                ],
                1,
            )
            cat_q2 = torch.cat(
                [
                    q2_rand - random_log_prob.detach(),
                    q2_next - next_log.detach(),
                    q2_current - current_log.detach(),
                ],
                1,
            )

            min_qf1_loss = (
                torch.logsumexp(
                    cat_q1 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )
            min_qf2_loss = (
                torch.logsumexp(
                    cat_q2 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2.mean() * self.min_q_weight

            loss_q1 = loss_q1 + min_qf1_loss
            loss_q2 = loss_q2 + min_qf2_loss

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        state_batch, edge_index = (
            data.x_s,
            data.edge_index_s,
        )

        actions, logp_a = self.actor(state_batch, edge_index)
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()
        return loss_pi

    def update(self, data, conservative=False, only_q=False):
        loss_q1, loss_q2 = self.compute_loss_q(data, conservative)

        self.optimizers["c1_optimizer"].zero_grad()

        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        loss_q1.backward(retain_graph=True)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        if not only_q:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            # one gradient descent step for policy network
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.optimizers["a_optimizer"].step()

            # Unfreeze Q-networks
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

    def _get_action_and_values(self, data, num_actions, batch_size, action_dim):
      
        alpha = torch.ones(action_dim)
        d = torch.distributions.Dirichlet(alpha)
        random_actions = d.sample((batch_size * self.num_random,))
        random_log_prob = (
            d.log_prob(random_actions).view(batch_size, num_actions, 1).to(self.device)
        )
        random_actions = random_actions.to(self.device)
        data_list = data.to_data_list()
        data_list = data_list * num_actions
        batch_temp = Batch.from_data_list(data_list).to(self.device)
        current_actions, current_log = self.actor(
            batch_temp.x_s, batch_temp.edge_index_s
        )
        current_log = current_log.view(batch_size, num_actions, 1)

        next_actions, next_log = self.actor(batch_temp.x_t, batch_temp.edge_index_t)
        next_log = next_log.view(batch_size, num_actions, 1)

        q1_rand = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, num_actions, 1)

        q2_rand = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, num_actions, 1)

        q1_current = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, num_actions, 1)

        q2_current = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, num_actions, 1)

        q1_next = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, num_actions, 1)

        q2_next = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, num_actions, 1)
        
        return (
            random_log_prob,
            current_log,
            next_log,
            q1_rand,
            q2_rand,
            q1_current,
            q2_current,
            q1_next,
            q2_next,
        )
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        return optimizers
    
    def learn(self, cfg, Dataset=None):
        sim = cfg.simulator.name
        if sim == "sumo": 
            #traci.close(wait=False)
            os.makedirs('saved_files/sumo_output/scenario_lux/', exist_ok=True)
            matching_steps = int(cfg.simulator.matching_tstep * 60 / cfg.simulator.sumo_tstep)  # sumo steps between each matching
            if 'meso' in cfg.simulator.net_file:
                matching_steps -= 1 
                
            sumo_cmd = [
            "sumo", "--no-internal-links", "-c", cfg.simulator.sumocfg_file,
            "--step-length", str(cfg.simulator.sumo_tstep),
            "--device.taxi.dispatch-algorithm", "traci",
            "-b", str(cfg.simulator.time_start * 60 * 60), "--seed", "10",
            "-W", 'true', "-v", 'false',
            ]
            print(cfg.simulator.sumocfg_file)
            assert os.path.exists(cfg.simulator.sumocfg_file), "SUMO configuration file not found!"

        if Dataset is not None:
            train_episodes = cfg.model.max_episodes  # set max number of training episodes
            T = cfg.simulator.max_steps  # set episode length
            epochs = trange(train_episodes*T)  #  # epoch iterator
            best_reward = -np.inf  # set best reward
            self.train()  # set model in train mode

            for step in epochs:
                if step % 1000 == 0:
                    self.eval()
                    (
                        episode_reward,
                        episode_served_demand,
                        episode_rebalancing_cost,
                        _,
                    ) = self.test(1, self.env, verbose = False)
                    self.train()
                    epochs.set_description(
                        f"Offline Step {step} | Reward: {np.mean(episode_reward):.2f} | ServedDemand: {np.mean(episode_served_demand):.2f} | Reb. Cost: {np.mean(episode_rebalancing_cost):.2f}"
                    )
                    epochs.update(1000)

                self.save_checkpoint(
                path=f"ckpt/{cfg.model.checkpoint_path}.pth"
                )

                batch = Dataset.sample_batch(self.BATCH_SIZE)
                self.update(data=batch, conservative=True)

        else:
            train_episodes = cfg.model.max_episodes  # set max number of training episodes
            epochs = trange(train_episodes)  # epoch iterator
            best_reward = -np.inf  # set best reward
            self.train()  # set model in train mode

            for i_episode in epochs:
                if sim =='sumo':
                    traci.start(sumo_cmd)
                obs, rew = self.env.reset()  # initialize environment
                step = 0
                obs = self.parser.parse_obs(obs)
                episode_reward = 0
                episode_reward += rew
                episode_served_demand = 0
                episode_rebalancing_cost = 0
                episode_served_demand += rew
                done = False

                while not done:
                    action_rl = self.select_action(obs)
                    desiredAcc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time + 1))
                        for i in range(len(self.env.region))
                    }
            
                    reb_action = solveRebFlow(
                        self.env,
                        self.env.cfg.directory,
                        desiredAcc,
                        self.cplexpath,
                    )
                    new_obs, rew, done, info = self.env.step(reb_action=reb_action)
                    step += 1
                    episode_reward += rew
                    episode_served_demand += info["profit"]
                    episode_rebalancing_cost += info["rebalancing_cost"]
                    
                    if not done: 
                        new_obs = self.parser.parse_obs(new_obs)
                        self.replay_buffer.store(obs, action_rl, cfg.model.rew_scale * rew, new_obs)

                    obs = new_obs
                    if i_episode > 10:
                        batch = self.replay_buffer.sample_batch(cfg.model.batch_size)
                        if step < cfg.model.only_q_steps:
                            self.update(data=batch, only_q=True)
                        else:
                            self.update(data=batch)
                epochs.set_description(
                    f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
                )
        
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
            obs = self.parser.parse_obs(obs)
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
                new_obs, rew, done, info = env.step(reb_action=reb_action)
                #calculate inflow to each node in the graph
               
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]

                if not done:
                    obs = self.parser.parse_obs(new_obs)
                
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

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path)
        try:
            # Attempt to load the model state dict as is
            self.load_state_dict(checkpoint["model"])
            #print(checkpoint["model"].keys())
        except RuntimeError as e:
        
            model_state_dict = checkpoint["model"]
            new_state_dict = {}
            # Remapping the keys
            for key in model_state_dict.keys():
                if "conv1.weight" in key:
                    new_key = key.replace("conv1.weight", "conv1.lin.weight")
                #elif "lin.bias" in key:
                #    new_key = key.replace("lin.bias", "bias")
                else:
                    new_key = key
                new_state_dict[new_key] = model_state_dict[key]

            self.load_state_dict(new_state_dict)
        
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
