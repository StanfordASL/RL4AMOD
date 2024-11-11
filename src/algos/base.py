import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from tqdm import trange
import numpy as np
class BaseAlgorithm:
    def __init__(self, **kwargs):
        """
        Base class for baseline algorithms.
        """
        pass
        
    def select_action(self, env):
        """
        This method should be overridden by derived classes to return the rebalance action.

        Possible variables to get from the environment:

        env.nregion: number of regions
        len(env.edges): number of edges
        env.acc: accumulated number of vehicles in each region
        env.time: current time step in environment 
        env.G.edges[i, j]["time"]: travel time from region i to region j
        env.scenario.demand_input[i, j][t]: demand forecast from region i to region j at time t
        """

        raise NotImplementedError("The select_action method must be implemented by subclasses.")

    def test(self, num_episodes, env):
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
                "--device.rerouting.threads", "1",
                "--summary-output", "saved_files/sumo_output/" + env.cfg.city + "/"+ "_dua_meso.static.summary.xml",
                "--tripinfo-output", "saved_files/sumo_output/" + env.cfg.city + "/"+ "_dua_meso.static.tripinfo.xml",
                "--tripinfo-output.write-unfinished", "true",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", str(env.cfg.seed),
                "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(env.cfg.sumocfg_file), "SUMO configuration file not found!"
        epochs = trange(num_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            inflow = np.zeros(env.nregion)
            done = False
            if sim =='sumo':
                traci.start(sumo_cmd)
            obs, rew = env.reset()
            eps_reward += rew
            eps_served_demand += rew
            while not done:

                reb_action = self.select_action(env)
            
                obs, rew, done, info = env.step(reb_action=reb_action)

                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]
                
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
            
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            inflows.append(inflow)
            epochs.set_description(
                f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}"
            )
        return episode_reward, episode_served_demand, episode_rebalancing_cost, inflows
        