import numpy as np
import torch
from torch import nn
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.nets.actor import GNNActor
import os 
import traci
from tqdm import trange

class BC(nn.Module):
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
        super(BC, self).__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item(),
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion

        # BC parameters
        self.env = env
        self.BATCH_SIZE = cfg.batch_size
        self.p_lr = cfg.p_lr
        self.clip = cfg.clip
        self.parser = parser

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.agent_name = cfg.agent_name

        # nnets
        self.actor = GNNActor(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        print(self.actor)
        
        self.optimizers = self.configure_optimizers()
    
        self.qf_criterion = nn.MSELoss()

        self.to(self.device)

    def parse_obs(self, obs, device):
        state = self.obs_parser.parse_obs(obs, device)
        return state

    def select_action(self, data, deterministic=True):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def update(self, data):
        state_batch, edge_index, action_batch = (
            data.x_s,
            data.edge_index_s,
            data.action.reshape(-1, self.env.nregion),
        )

        m = self.actor(state_batch, edge_index, return_dist=True)

        policy_logpp = m.log_prob(action_batch)
        
        loss_pi = (-policy_logpp).mean()
        '''
        #alternative loss
        a = m.rsample()
        a = a.squeeze(-1)
        loss_pi = F.mse_loss(a, action_batch)
        '''

        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.optimizers["a_optimizer"].step()

        return

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
       
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)

        return optimizers

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

    def learn(self, cfg, Dataset=None):
        sim = cfg.simulator.name
        if sim == "sumo": 
            #traci.close(wait=False)
            scenario_path = '/src/envs/data/lux/'
            sumocfg_file = 'dua_meso.static.sumocfg'
            net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
            os.makedirs('saved_files/sumo_output/scenario_lux/', exist_ok=True)
            matching_steps = int(cfg.simulator.matching_tstep * 60 / cfg.simulator.sumo_tstep)  # sumo steps between each matching
            if 'meso' in net_file:
                matching_steps -= 1 
                
            sumo_cmd = [
            "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
            "--step-length", str(cfg.simulator.sumo_tstep),
            "--device.taxi.dispatch-algorithm", "traci",
            "-b", str(cfg.simulator.time_start * 60 * 60), "--seed", "10",
            "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(os.path.join(scenario_path, sumocfg_file)), "SUMO configuration file not found!"


        train_episodes = cfg.model.max_episodes  # set max number of training episodes
        T = cfg.simulator.max_steps  # set episode length
        epochs = trange(train_episodes*T)  #  # epoch iterator
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
            self.update(data=batch)

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
