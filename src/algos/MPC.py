from collections import defaultdict
import subprocess
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import re
from tqdm import trange
from src.misc.utils import mat2str
import numpy as np


class MPC:
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
        self.policy_name = kwargs.get('policy_name')
        self.T = kwargs.get('T')
        self.platform = None
    
    def MPC_exact(self, env, sumo=False):
        tstep = env.tstep
        t = env.time

        flows = list()
        demandAttr = list()
        
        if sumo:
            # Demand generation
            reservations = traci.person.getTaxiReservations(3)
            for trip in reservations:
                persons = trip.persons[0]
                waiting_time = traci.person.getWaitingTime(persons)
                if waiting_time > env.max_waiting_time * 60:
                    traci.person.remove(persons)
                    persons = ''
                if not persons:
                    continue
                o = int(persons[(persons.find('o') + 1):persons.find('d')])
                d = int(persons[(persons.find('d') + 1):persons.find('#')])
                if (o, d, env.demand_time[o, d][t], env.price[o, d][t]) in flows:
                    idx = flows.index((o, d, env.demand_time[o, d][t], env.price[o, d][t]))
                    flow = demandAttr[idx][3] + 1
                    demandAttr[idx] = (o, d, t, flow, env.demand_time[(o, d)][t], env.price[o, d][t])
                else:
                    flows.append((o, d, env.demand_time[o, d][t], env.price[o, d][t]))
                    demandAttr.append((o, d, t, 1, env.demand_time[o, d][t], env.price[o, d][t]))
            # Future demand
            demandAttr.extend([(i,j,tt,env.demand[i,j][tt], env.demand_time[i,j][tt], env.price[i,j][tt]) for i,j in env.demand for tt in range(t+tstep, min(t+self.T, env.duration), tstep) if env.demand[i,j][tt]>1e-3])
            accTuple = [(n,env.acc[n][t+tstep]) for n in env.acc]
            daccTuple = [(n,tt,env.dacc[n][tt]) for n in env.acc for tt in range(t,min(t+self.T, env.duration))]
            edgeAttr = [(i,j,env.rebTime[i,j][t]) for i,j in env.edges]
            modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
            MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/' + self.directory + '/'
        else: 

            t = env.time
            demandAttr = [
                (
                    i,
                    j,
                    tt,
                    env.demand[i, j][tt],
                    env.demandTime[i, j][tt],
                    env.price[i, j][tt],
                )
                for i, j in env.demand
                for tt in range(t, t + self.T)
                if env.demand[i, j][tt] > 1e-3
            ]
            accTuple = [(n, env.acc[n][t]) for n in env.acc]
            daccTuple = [
                (n, tt, env.dacc[n][tt])
                for n in env.acc
                for tt in range(t, t + self.T)
            ]
            edgeAttr = [(i, j, env.rebTime[i, j][t]) for i, j in env.edges]
            modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
            MPCPath = os.getcwd().replace("\\", "/") + "/saved_files/cplex_logs/" + self.directory + "/"
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + "data_{}.dat".format(t)
        resfile = MPCPath + "res_{}.dat".format(t)
        with open(datafile, "w") as file:
            file.write('path="' + resfile + '";\r\n')
            file.write("t0=" + str(t) + ";\r\n")
            file.write("T=" + str(self.T) + ";\r\n")
            file.write("beta=" + str(env.beta) + ";\r\n")
            file.write("demandAttr=" + mat2str(demandAttr) + ";\r\n")
            file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
            file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
            file.write("daccAttr=" + mat2str(daccTuple) + ";\r\n")

        modfile = modPath + "MPC.mod"
        my_env = os.environ.copy()
        if self.platform == None:
            my_env["LD_LIBRARY_PATH"] = self.cplexpath
        else:
            my_env["DYLD_LIBRARY_PATH"] = self.cplexpath
        out_file = MPCPath + "out_{}.dat".format(t)
        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [self.cplexpath + "oplrun", modfile, datafile],
                stdout=output_f,
                env=my_env,
            )
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f1, f2 = v.split(",")
                        f1 = float(re.sub("[^0-9e.-]", "", f1))
                        f2 = float(re.sub("[^0-9e.-]", "", f2))
                        paxFlow[int(i), int(j)] = float(f1)
                        rebFlow[int(i), int(j)] = float(f2)
        paxAction = [
            paxFlow[i, j] if (i, j) in paxFlow else 0 for i, j in env.edges
        ]
        rebAction = [
            rebFlow[i, j] if (i, j) in rebFlow else 0 for i, j in env.edges
        ]

        return paxAction, rebAction

    def test(self, num_episodes, env):
        """
        for testing MPC
        - num_episodes: An integer representing the number of episodes to run the test.
        - env: The AMoD environment object that contains various attributes and methods.
        """
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
                "--summary-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.policy_name + "_dua_meso.static.summary.xml",
                "--tripinfo-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.policy_name + "_dua_meso.static.tripinfo.xml",
                "--tripinfo-output.write-unfinished", "true",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", "10",
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
            _ = env.reset_old()
            
            while not done:
                
                if sim == 'sumo':
                    # Taxis information
                    env.time = int(((traci.simulation.getTime() - env.scenario.time_start * 60) // 60) - env.tstep)
                    taxi_ids = traci.vehicle.getTaxiFleet(0)
                    env.set_taxi_to_region(taxi_ids)
                    # Info initialization
                    for i in env.region:
                        num_taxis = len(env.regions_sumo[i]['taxis'])
                        env.acc[i][env.time + env.tstep] = num_taxis
                    # MPC optimization step
                    pax_action, reb_action = self.MPC_exact(env, True)
                    # Environment step
                    _, paxreward, done, info = env.pax_step(paxAction=pax_action, CPLEXPATH=self.cplexpath)
                    _, rebreward, done, info = env.reb_step(reb_action)
                    env.sumo_steps()
                    rew = paxreward + rebreward
                    if done:
                        traci.simulationStep()
                        traci.close()
                else:
                    pax_action, reb_action = self.MPC_exact(env)

                    _, paxreward, done, info = env.pax_step(paxAction=pax_action, CPLEXPATH=self.cplexpath)

                    _, rebreward, done, info = env.reb_step(reb_action)

                    rew = paxreward + rebreward

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
            epochs.set_description(f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}")
        return episode_reward, episode_served_demand, episode_rebalancing_cost, inflows
        