"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator.
"""
import torch
import numpy as np
import subprocess
import os
from collections import defaultdict
import networkx as nx
from src.misc.utils import mat2str
import json
from src.misc.utils import dictsum
from src.algos.reb_flow_solver import solveRebFlow
import gymnasium as gym
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib
import traci
import math
from lxml import etree as ET
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import KMeans

demand_ratio = {
    "san_francisco": 2,
    "washington_dc": 4.2,
    "nyc_brooklyn": 9,
    "shenzhen_downtown_west": 2.5,
    "lux": 0.1
}
json_hr = {
    "san_francisco": 19,
    "washington_dc": 19,
    "nyc_brooklyn": 19,
    "shenzhen_downtown_west": 8,
}
beta_dict = {
    "san_francisco": 0.2,
    "washington_dc": 0.5,
    "nyc_brooklyn": 0.5,
    "shenzhen_downtown_west": 0.5,
}
scenario_path = 'data/LuSTScenario/'
sumocfg_file = 'dua_meso.static.sumocfg'
net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
matching_tstep = 1
sumo_cmd = [
    "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
    "--device.taxi.dispatch-algorithm", "traci",
    "-b", "0", "--seed", "10",
    "-W", 'true', "-v", 'false',
]
# TODO: make this configurable
CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/"
# CPLEXPATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, grid_h=4, grid_w=4, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = self.env.scenario.thorizon
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w
   
    def get_edge_index(self):
        return torch.cat([torch.tensor([self.env.region]), torch.tensor([self.env.region])])
 
    def parse_obs(self, obs):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1]*self.s for n in self.env.region]).view(1, 1, self.env.nregions).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float(),
            torch.tensor([[sum([(self.env.demand[i,j][t])*(self.env.price[i,j][t])*self.s \
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float()),
              dim=1).squeeze(0).view(21, self.env.nregions).T

        # Edge index self-connected tensor definition (da modificare)
        edge_index = self.get_edge_index()
        return {
            "node_features": x.numpy(),
            "edge_index": edge_index.numpy()
        }

    def parse_obs(self, obs):
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] *
                            self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                (obs[0][n][self.env.time + 1] +
                                 self.env.dacc[n][t])
                                * self.s
                                for n in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                sum(
                                    [
                                        (self.env.scenario.demand_input[i, j][t])
                                        * (self.env.price[i, j][t])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + self.T, self.env.nregion)
            .T
        )
        edge_index = self.get_edge_index()
        return {
            "node_features": x.numpy(),
            "edge_index": edge_index.numpy()
        }
    


class AMoD(gym.Env):
    # initialization
    def __init__(
        self, beta=0.2, city="lux", reward_scale_factor=1
    ):  # updated to take scenario and beta (cost for rebalancing) as input
        print("Running for city ", city)
        self.json_file = f"data/scenario_{city}.json"
        
        
        scenario = Scenario(num_cluster=8, json_file="data/scenario_nyc4x4.json", sumo_net_file=net_file, acc_init=20, sd=100, demand_ratio=demand_ratio[city], time_start=0, time_horizon=10, duration=2, tstep=matching_tstep)
        self.city = city
        self.reward_scale_factor=reward_scale_factor
        self.scenario = scenario
        self.G = scenario.G  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.regions_sumo = scenario.regions_sumo
        self.taxi_routes = scenario.taxi_routes
        self.reservations = list()
        self.reservations_assigned = list()
        self.demand_time = self.scenario.demand_time
        self.reb_time = self.scenario.reb_time
        self.time = 0  # current time
        self.tstep = self.scenario.tstep
        self.duration = scenario.duration  # final time
        self.demand = defaultdict(dict)  # demand
        self.region = list(self.G)  # set of regions
        self.price = defaultdict(dict)  # price
        self.acc = defaultdict(dict)  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of vehicles arriving at each region, key: i - region, t - time
        self.demand_res = defaultdict(dict)  # reservations from sumo assigned to each (origin, destination) pair
        self.rebFlow = defaultdict(dict)  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.traci_connected = False
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.reb_time[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
            self.demand_res[i, j] = defaultdict(dict)  # reservations from sumo assigned to each (origin, destination) pair
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        self.matching_steps = int(matching_tstep * 60)  # sumo steps between each matching
        if scenario.is_meso:
            self.matching_steps -= 1     # In the meso setting one step is done within the reb_step
 
        # observation: current vehicle distribution, time, future arrivals, demand        
        self.obs = (self.acc, self.time, self.dacc, self.demand) 
        self.parser = GNNParser(self)
        edge_index = self.parser.get_edge_index()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.nregion, 1), dtype=np.float32)
        self.observation_space_dict = {
            "node_features": gym.spaces.Box(low=0, high=float('inf'), shape=(self.nregion, 21), dtype=np.float32),
            "edge_index": gym.spaces.Box(low=0, high=self.nregion, shape=edge_index.shape, dtype=int)
        }
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)
        
    def matching(self, demandAttr=[], PATH='', platime_endorm='linux'):
        """
        Matching step method: generation of a flow demand between the nodes of the aggregated net given the input demand
        and the available taxi from sumo
        """
        tstep = self.tstep
        t = self.time
        accTuple = [(n, self.acc[n][t + tstep]) for n in self.acc]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        matchingPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/matching/' + PATH + '/'
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath + 'data_{}.dat'.format(t)
        resfile = matchingPath + 'res_{}.dat'.format(t)
        with open(datafile, 'w') as file:
            file.write('path="' + resfile + '";\r\n')
            file.write('demandAttr=' + mat2str(demandAttr) + ';\r\n')
            file.write('accInitTuple=' + mat2str(accTuple) + ';\r\n')
        modfile = modPath + 'matching.mod'
        my_env = os.environ.copy()
        if platime_endorm == 'mac':
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = matchingPath + 'out_{}.dat'.format(t)
        with open(out_file, 'w') as output_f:
            subprocess.check_call([CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()
        flow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        flow[int(i), int(j)] = float(f)
        paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return paxAction

    def pax_step(self, paxAction=None, PATH='', platime_endorm='linux'):
        sumo_step = 0
        while sumo_step < self.matching_steps:
            traci.simulationStep()
            sumo_step += 1  
        tstep = self.tstep
        self.time = int(((traci.simulation.getTime() - self.scenario.time_start * 60) // 60) - tstep)
        t = self.time
        self.reward = 0
        # Step info dictionary initialization
        self.info['served_demand'] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        self.info['reward'] = 0
        # Matching step
        demandAttr = self.get_demand_attr()
        if paxAction is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            # Taxis information
            taxi_ids = traci.vehicle.getTaxiFleet(0)
            self.set_taxi_to_region(taxi_ids)
            # Info initialization
            for i in self.region:
                num_taxis = len(self.regions_sumo[i]['taxis'])
                self.acc[i][t + tstep] = num_taxis
            paxAction = self.matching(demandAttr=demandAttr, PATH=PATH, platime_endorm=platime_endorm)
        self.paxAction = paxAction
        # Serving passengers
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.demand or t not in self.demand[i, j] or self.paxAction[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert paxAction[k] < self.acc[i][t + tstep] + 1e-3
            self.paxAction[k] = min(self.acc[i][t + tstep], int(paxAction[k]))
            self.servedDemand[i, j][t] = self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i, j][t]
            self.acc[i][t + tstep] -= self.paxAction[k]
            self.info['revenue'] += self.paxAction[k] * (self.price[i, j][t])
            # SUMO taxi dispatch
            taxi_match = 0
            while taxi_match < self.paxAction[k]:
                taxi_num = 0
                reservation = self.demand_res[i, j][t][taxi_match]
                reservation_id = reservation.id
                persons = reservation.persons[0]
                edge_o = reservation.fromEdge
                edge_d = reservation.toEdge
                taxi = self.regions_sumo[i]['taxis'][taxi_num]
                taxi_id = taxi[0]
                # Direction check
                edge = traci.vehicle.getRoadID(taxi_id)
                while edge[1].isdigit != edge_o[1].isdigit:
                    if taxi_num == len(self.regions_sumo[i]['taxis']) - 1:
                        taxi = self.regions_sumo[i]['taxis'][0]
                        taxi_id = taxi[0]
                        break
                    taxi_num += 1
                    taxi = self.regions_sumo[i]['taxis'][taxi_num]
                    taxi_id = taxi[0]
                    edge = traci.vehicle.getRoadID(taxi_id)
                # Dispatch and taxi remove from the region
                traci.vehicle.dispatchTaxi(taxi_id, [reservation_id])
                route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi')
                self.regions_sumo[i]['taxis'].remove(taxi)
                self.reservations_assigned.append([reservation_id, persons, taxi_id])
                # Find taxi travel time and travel info computation
                demand_time = int(math.ceil(route.travelTime / (traci.vehicle.getSpeedFactor(taxi_id) * 60)))
                arrival_time = t + demand_time      # Use the speed factor to have a more accurate forecast of the travel time
                if arrival_time % tstep != 0:
                    arrival_time = (arrival_time // tstep + 1) * tstep
                self.paxFlow[i, j][arrival_time] += 1
                self.dacc[j][arrival_time] += 1
                self.info["operating_cost"] += demand_time * self.beta
                self.reward += (self.price[i, j][t] - demand_time * self.beta)
                taxi_match += 1

        self.obs = (self.acc, self.time, self.dacc, self.demand)  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        self.info["reward"] += max(0, self.reward)
        return self.obs, max(0, self.reward), done, self.info

    def reb_step(self, rebAction):
        tstep = self.tstep
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.info['rebalanced_vehicles'] = 0
        self.rebAction = rebAction
        # rebalancing
        reb_assign = list()
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges or self.rebAction[k] < 1e-3:
                continue
            assert rebAction[k] < self.acc[i][t + tstep] + 1e-3
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t + tstep], int(rebAction[k]))
            self.acc[i][t + tstep] -= self.rebAction[k]
            self.info['rebalanced_vehicles'] += self.rebAction[k]
            # SUMO travels assignment
            taxis_reb = 0
            while taxis_reb < self.rebAction[k]:
                taxi = self.regions_sumo[i]['taxis'][0]
                taxi_id = taxi[0]
                edge_d = self.regions_sumo[j]['in_edges'][np.random.randint(len(self.regions_sumo[j]['in_edges']))].getID()
                edge_d_length = traci.lane.getLength(edge_d + '_0')
                edge_o = traci.vehicle.getRoadID(taxi_id)
                route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi', routingMode=1)
                reb_time = int(math.ceil(route.travelTime / (traci.vehicle.getSpeedFactor(taxi_id) * 60)))
                arrival_time = t + reb_time      # Use the speed factor to have a more accurate forecast of the travel time
                if arrival_time % tstep != 0:
                    arrival_time = (arrival_time // tstep + 1) * tstep
                if self.scenario.is_meso:
                    edge_o_length = traci.lane.getLength(edge_o + '_0')
                    reb_id = taxi_id + 'o' + str(i) + 'd' + str(j) + '#' + str(taxis_reb) + '_rebalancing'
                    traci.person.add(personID=reb_id, edgeID=edge_o, pos=edge_o_length)
                    traci.person.appendDrivingStage(personID=reb_id, toEdge=edge_d, lines='taxi')
                    reb_assign.append((taxi_id, reb_id))
                else:
                    traci.vehicle.resume(taxi_id)
                    traci.vehicle.setRoute(taxi_id, route.edges)
                    traci.vehicle.setStop(taxi_id, edge_d, pos=edge_d_length, flags=1)  # Set the stop in the new location
                    traci.vehicle.setStopParameter(taxi_id, 0, 'actType', 'rebalancing')  # Set the taxi condition to rebalancing

                self.regions_sumo[i]['taxis'].remove(taxi)
                self.rebFlow[i, j][arrival_time] += 1
                self.dacc[i][arrival_time] += 1
                self.info['rebalancing_cost'] += reb_time * self.beta
                self.info["operating_cost"] += reb_time * self.beta
                self.reward -= reb_time * self.beta
                taxis_reb += 1

        if self.scenario.is_meso:
            traci.simulationStep()
            reservations = traci.person.getTaxiReservations(3)
            reservation_ids = [r.id for r in reservations]
            reservation_p = [r.persons[0] for r in reservations]
            for taxi_id, person_id in reversed(reb_assign):
                res_id = reservation_ids[reservation_p.index(person_id)]
                traci.vehicle.dispatchTaxi(taxi_id, [res_id])

        self.obs = (self.acc, self.time, self.dacc, self.demand)  # use self.time to index the next time step
        # Travel time update from sumo
        self.update_routes(time=t+tstep)
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.reb_time[i, j][self.time]

        done = (self.duration == t + tstep)  # if the episode is completed
        self.info["reward"] += self.reward
        return self.obs, self.reward, done, self.info

    def step(self, action_rl):
        """
        Params:
            action_rl is action outputted by stable baselines RL policy
            It is desired distribution of vehicles
        """

        action_rl = action_rl / np.sum(action_rl)
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        desiredAcc = {
            self.region[i]: int(
                action_rl[i] * dictsum(self.acc, self.time + 1))
            for i in range(len(self.region))
        }
        # solve minimum rebalancing distance problem (Step 3 in paper)
        rebAction = solveRebFlow(
            self,
            f"scenario_{self.city}",
            desiredAcc,
            CPLEXPATH,
            directory="saved_files",
        )
        # take action in environment (rebalancing step)
        obs, rebreward, done, info = self.reb_step(rebAction)
        info_copied = info.copy()
        parsed_obs = self.parser.parse_obs(obs)
        self.time += 1
        # Do matching step
        paxreward = 0
        if not done:
            obs, paxreward, done, _ = self.pax_step(
                PATH=f"scenario_{self.city}" )
            parsed_obs = self.parser.parse_obs(obs)
        else:
            self.traci_connected = False
            traci.close()
        return parsed_obs, (paxreward + rebreward) * self.reward_scale_factor, done, False, info_copied

    def reset(self, seed=None):
        # reset the episode
        super().reset(seed=seed)
        if self.traci_connected:
            traci.close()
        self.traci_connected = True
        traci.start(sumo_cmd)
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        self.demand_res = defaultdict(dict)  # demand with reservations from sumo
        self.reservations_assigned = list()  # reservation assigned during the episode
        trip_attr = self.scenario.get_random_demand()
        self.regionDemand = defaultdict(dict)
        for i, j, t, d, p in trip_attr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        for n in self.region:
            self.demand[n, n] = {t: 0 for t in range(self.duration + self.scenario.thorizon)}
            self.price[n, n] = {t: self.scenario.price[n, n][t] for t in range(self.duration + self.scenario.thorizon)}

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
            self.demand_res[i, j] = defaultdict(dict)
            self.servedDemand[i, j] = defaultdict(float)
            if (i, j) in self.taxi_routes:
                self.demand_time[i, j][0] = self.taxi_routes[(i, j)][2] / 60
                self.demand_time[i, j][0] = max(int(math.ceil(self.demand_time[i, j][0])), 1)
                self.reb_time[i, j][0] = self.demand_time[i, j][0]
            else:
                self.demand_time[i, j][0] = 0
                self.reb_time[i, j][0] = 0

        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
            self.regions_sumo[n]['taxis'] = list()

        # Initialize taxis in the network
        self.scenario.set_taxi_lines()
        self.scenario.set_taxi_distribution()

        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        if 'meso' in net_file:
            traci.simulationStep()
        obs, paxreward, done, info = self.pax_step(
            PATH=f"scenario_{self.city}"
        )
        self.reward = 0
        return self.parser.parse_obs(obs), {}

    def get_demand_attr(self):
        """
        Method to set the demand for each time step to be satisfied through the matching step
        """
        tstep = self.tstep
        t = self.time
        # Get the reservations demand
        self.reservations = list()
        reservations = traci.person.getTaxiReservations(3)  # Consider only the retrived reservations (the reservation in progress are not appended)
        demandAttr = []
        trips = []
        for trip in reservations:
            trip_id = trip.id
            persons = trip.persons[0]
            if not persons:
                continue
            o = int(persons[(persons.find('o') + 1):persons.find('d')])
            d = int(persons[(persons.find('d') + 1):persons.find('#')])
            price = self.price[o, d][t]
            state = trip.state
            if state >= 4:
                continue
            # Condition to increase the flow if the same trip demand is present in the reservations list
            if not (o, d) in trips:
                trips.append((o, d))
                demandAttr.append((o, d, 1, price))
            else:
                demandAttr[trips.index((o, d))] = (o, d, demandAttr[trips.index((o, d))][2] + 1, price)
            if not trip in self.reservations:
                self.reservations.append(trip)
                if not self.demand_res[o, d][t]:
                    self.demand_res[o, d][t] = [trip]
                else:
                    self.demand_res[o, d][t].append(trip)
        return demandAttr

    def update_routes(self, time):
        """
        Method to update the travel time in the network
        """
        new_routes = defaultdict(tuple)
        for o, d in self.edges:
            # Get travel time from the o-d route from the sumo net
            if (o, d) in self.taxi_routes:
                edge_o = self.taxi_routes[(o, d)][0][0]
                edge_d = self.taxi_routes[(o, d)][0][1]
                route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi', depart=traci.simulation.getTime(), routingMode=1)
                new_routes[(o, d)] = ((edge_o, edge_d), route.edges, route.travelTime)
                self.demand_time[o, d][time] = new_routes[(o, d)][2] / 60
                self.demand_time[o, d][time] = max(int(math.ceil(self.demand_time[o, d][time])), 1)
                self.reb_time[o, d][time] = self.demand_time[o, d][time]
            else:
                self.demand_time[o, d][time] = 0
                self.reb_time[o, d][time] = 0
        self.taxi_routes = new_routes

    def set_taxi_to_region(self, taxi_ids):
        """
        Method to assign the avaiable taxi to the relative region
        """
        for taxi_id in taxi_ids:
            stop = traci.vehicle.getStops(taxi_id)
            stop_info = stop[0].actType
            # Ignore in-rebalancing taxis
            if stop_info == 'rebalancing':
                parking = self.check_parking(taxi_id, stop)
                if not parking:
                    continue
            position = traci.vehicle.getPosition(taxi_id)
            region = self.scenario.cluster_alg.predict(np.array(position).reshape(1, -1))
            if not (taxi_id, stop_info) in self.regions_sumo[region[0]]['taxis']:
                self.regions_sumo[region[0]]['taxis'].append((taxi_id, stop_info))

    def check_parking(self, taxi_id, stop):
        """
        Method to check if the vehicle is in parking mode
        """
        if self.scenario.is_meso:
            stop_edge = stop[0].lane.split('_')[0]
            taxi_edge = traci.vehicle.getRoadID(taxi_id)
            taxi_spd = traci.vehicle.getSpeed(taxi_id)
            if stop_edge == taxi_edge and taxi_spd == 0:
                return True
            else:
                return False
        else:
            return traci.vehicle.isStoppedParking(taxi_id)

class Scenario:
    def __init__(self, num_cluster=4, duration=2, sd=None, demand_ratio=None, json_file=None, time_start=7,
            time_horizon=10, tstep=2, varying_time=False, json_regions=None, sumo_net_file=None, acc_init=100):
        """
        Method to initialize the scenario for the AMoD problem with the following features
            demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
            static_demand will then be sampled according to a Poisson distribution
            alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        """
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)

        # Aggregated net creation
        self.N = num_cluster
        if sumo_net_file is None:
            sumo_net_file = 'data/LuSTScenario/lust.net.xml'

        self.is_meso = 'meso' in sumo_net_file
        self.sumo_net = sumolib.net.readNet(sumo_net_file)
        self.regions_sumo, self.cluster_alg = self.sumo_net_clustering()
        self.taxi_routes = self.get_taxi_routes()
        self.G = nx.complete_graph(self.N)
        self.G = self.G.to_directed()
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        self.acc_init = acc_init  # Initial vehicle distribution per node (Settato esternamente dal main e non più dal json)
        for n in self.G.nodes:
            self.G.nodes[n]['accInit'] = acc_init

        # Demand creation (from json)
        self.varying_time = varying_time
        self.is_json = True
        with open(json_file, "r") as file:
            data = json.load(file)

        self.tstep = tstep
        self.thorizon = time_horizon
        self.demand_input = defaultdict(dict)
        self.json_regions = json_regions
        self.price = defaultdict(dict)
        self.alpha = 0
        self.demand_time = defaultdict(dict)
        self.reb_time = defaultdict(dict)
        self.time_start = time_start * 60
        self.duration = duration * 60
        # Time demand between nodes initialization
        for i, j in self.edges:
            self.demand_time[i, j] = defaultdict(int)
            self.reb_time[i, j] = defaultdict(int)
            self.price[i, j] = defaultdict(int)
            self.demand_input[i, j] = defaultdict(int)

        # Demand input and prices initialization from the json file (all the available data)
        for item in data["demand"]:
            t, o, d, v, p = item["time_stamp"], item["origin"], item["destination"], item["demand"], item["price"]
            if json_regions != None and (o not in json_regions or d not in json_regions):
                continue
            if (o, d) not in self.edges:
                continue
            if (o, d) not in self.demand_input:
                self.demand_input[o, d], self.price[o, d], self.demand_time[o, d] = defaultdict(float), defaultdict(
                    float), defaultdict(float)
            # Demand data aggregation based on the tstep
            self.demand_input[o, d][(t - self.time_start) // self.tstep] += v * demand_ratio
            self.price[o, d][(t - self.time_start) // self.tstep] += p * v * demand_ratio

        # Demand time and prices update for the effective nodes in the net and the effective episode duration
        for o, d in self.edges:
            # Get travel time from the o-d route from the sumo net
            for t in range(0, self.time_start + self.duration + self.thorizon):
                if (o, d) in self.taxi_routes:
                    self.demand_time[o, d][t] = self.taxi_routes[(o, d)][2] / 60
                    self.demand_time[o, d][t] = max(int(math.ceil(self.demand_time[o, d][t])), 1)
                    self.reb_time[o, d][t] = self.demand_time[o, d][t]
                else:
                    self.demand_time[o, d][t] = 0
                    self.reb_time[o, d][t] = 0
                if t in self.demand_input[o, d]:
                    self.price[o, d][t] /= self.demand_input[o, d][t]
                else:
                    self.demand_input[o, d][t] = 0
                    self.price[o, d][t] = 0

        # Create the taxi initialization routes xml file
        # self.set_taxi_init_xml()

    def sumo_net_clustering(self):
        """
        Function to cluster the junctions of a SUMO net, once a .sumocgf file has been started
        """
        # Nodes id and position
        nodes = self.sumo_net.getNodes()
        nodes_id = list()
        nodes_pos = list()
        for node in nodes:
            nodes_id.append(node.getID())
            nodes_pos.append(node.getCoord())

        # Clustering
        kmeans = KMeans(n_clusters=self.N, random_state=0)
        kmeans.fit(np.array(nodes_pos))
        labels = kmeans.labels_
        regions_sumo = list()
        for region in range(self.N):
            nodes_cluster = []
            nodes_id_cluster = []
            nodes_pos_cluster = []
            cluster_center = kmeans.cluster_centers_[region]
            for idx in range(len(nodes_pos)):
                if labels[idx] == region:
                    nodes_cluster.append(nodes[idx])
                    nodes_id_cluster.append(nodes_id[idx])
                    nodes_pos_cluster.append(nodes_pos[idx])

            distance = np.linalg.norm(np.array(nodes_pos_cluster) - cluster_center, axis=1)
            # Terminal edges correction and urban center selection
            is_urban = False
            while not is_urban:
                is_urban = True
                min_idx = np.argmin(distance)
                node = nodes_cluster[min_idx]
                in_edges = node.getIncoming()
                out_edges = node.getOutgoing()
                if node.getType() == 'dead_end':
                    distance[min_idx] = math.inf
                    is_urban = False
                    continue

                for in_edge in in_edges:
                    if not in_edge.getIncoming():
                        in_edges.remove(in_edge)
                    if 'motorway' in in_edge.getType():
                        distance[min_idx] = math.inf
                        is_urban = False

                for out_edge in out_edges:
                    if not out_edge.getOutgoing():
                        out_edges.remove(out_edge)
                    if 'motorway' in out_edge.getType():
                        distance[min_idx] = math.inf
                        is_urban = False

            regions_sumo.append({'id': nodes_id_cluster, 'position': np.array(nodes_pos_cluster),
                                 'id_center': node.getID(), 'position_center': nodes_pos_cluster[min_idx],
                                 'in_edges': in_edges,
                                 'out_edges': out_edges,
                                 'taxis': []}
                                )

        # Cluster centers edges
        return regions_sumo, kmeans

    def set_taxi_init_xml(self):
        """
        Method to create a .xml file with the initialization routes for the nodes in the net
        """
        routes = ET.Element('routes')
        for node in self.G.nodes:
            for edge_o in self.regions_sumo[node]['in_edges']:
                for edge_d in self.regions_sumo[node]['out_edges']:
                    edges = [edge_o.getID(), edge_d.getID()]
                    route_id = edge_o.getID() + edge_d.getID() + 'init'
                    route = ET.SubElement(routes, 'route')
                    route.set('edges', " ".join(edges))
                    route.set('id', route_id)
        tree = ET.ElementTree(routes)
        tree.write("data/LuSTScenario/input/routes/taxi_initialization.rou.xml", xml_declaration=True, pretty_print=True)

    def get_taxi_routes(self):
        """
        Method to compute the shortest route given the center of an aggregated net and save a .xml with the trip
        """
        # Cluster centers edges
        taxi_routes = defaultdict(tuple)
        routes = list(combinations(list(range(len(self.regions_sumo))), 2))
        for o, d in routes:
            taxi_routes[(o, d)] = (None, None, float('inf'))
            for edge_o in self.regions_sumo[o]['out_edges']:
                for edge_d in self.regions_sumo[d]['in_edges']:
                    route = self.sumo_net.getOptimalPath(edge_o, edge_d, fastest=True)
                    if route[1] < taxi_routes[(o, d)][2]:
                        taxi_routes[(o, d)] = (
                            (route[0][0].getID(), route[0][-1].getID()), route[0], route[1])
            # Add the way back
            edge_o = taxi_routes[(o, d)][0][1]
            edge_d = taxi_routes[(o, d)][0][0]
            # Edge correction for the backward road
            if edge_o[1].isdigit():
                edge_o = edge_o[1:]
            else:
                edge_o = edge_o[2:]
            edge_found = False
            for edge in self.regions_sumo[d]['out_edges']:
                if edge_o in edge.getID():
                    edge_o = edge.getID()
                    edge_found = True
                    break
            if not edge_found:
                edge_o = self.regions_sumo[d]['out_edges'][0].getID()

            if edge_d[1].isdigit():
                edge_d = edge_d[1:]
            else:
                edge_d = edge_d[2:]
            edge_found = False
            for edge in self.regions_sumo[o]['in_edges']:
                if edge_d in edge.getID():
                    edge_d = edge.getID()
                    edge_found = True
                    break
            if not edge_found:
                edge_d = self.regions_sumo[o]['in_edges'][0].getID()

            ids = [edge.getID() for edge in self.regions_sumo[d]['out_edges']]
            edge_o = self.regions_sumo[d]['out_edges'][ids.index(edge_o)]
            ids = [edge.getID() for edge in self.regions_sumo[o]['in_edges']]
            edge_d = self.regions_sumo[o]['in_edges'][ids.index(edge_d)]
            route = self.sumo_net.getOptimalPath(edge_o, edge_d, fastest=True)
            try:
                taxi_routes[(d, o)] = ((route[0][0].getID(), route[0][-1].getID()), route[0], route[1])
            except Exception as e:
                raise RuntimeError(f"Route from {d} to {o} not found: please reduce the number of regions")

        taxi_routes = defaultdict(tuple, sorted(taxi_routes.items(), key=lambda x: x[0]))
        return taxi_routes

    def get_random_demand(self):
        """
        generate demand and price
        reset = True means that the function is called in the reset() method of AMoD enviroment,
        assuming static demand is already generated
        reset = False means that the function is called when initializing the demand
        In the same method a .xml file is generated and it will be loaded in SUMO
        """
        demand = defaultdict(dict)
        price = defaultdict(dict)
        trip_attr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, (self.duration + self.thorizon) // self.tstep):
                t *= self.tstep
                t0 = (self.time_start + t) * 60  # [s]
                tf = (self.time_start + t + self.tstep) * 60
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(self.demand_input[i, j][t])
                        price[i, j][t] = self.price[i, j][t]
                        if i == j:
                            demand[i, j][t] = 0
                            continue
                        if demand[i, j][t] > 0:
                            edge_o = self.regions_sumo[i]['out_edges'][np.random.randint(len(self.regions_sumo[i]['out_edges']))].getID()
                            edge_d = self.regions_sumo[j]['in_edges'][np.random.randint(len(self.regions_sumo[j]['in_edges']))].getID()
                            edge_o_length = traci.lane.getLength(edge_o + '_0')
                            person_id = 'p' + str(t) + 'o' + str(i) + 'd' + str(j) + '#'
                            depart_time = np.random.randint(t0, tf)  # Time instant at which the person appears in the network
                            for person_num in range(demand[i, j][t]):
                                person_id = person_id + str(person_num)
                                traci.person.add(personID=person_id, edgeID=edge_o, depart=depart_time, pos=edge_o_length)
                                traci.person.setColor(typeID=person_id, color=(237, 177, 32))
                                traci.person.appendDrivingStage(personID=person_id, toEdge=edge_d, lines='taxi')
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0

                    trip_attr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:
            self.static_demand = dict()
            region_rand = (np.random.rand(len(self.G)) * self.alpha * 2 + 1 - self.alpha)
            if type(self.demand_input) in [float, int, list, np.array]:

                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:
                    self.region_demand = region_rand * np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    prob = np.array([np.math.exp(-self.reb_time[i, j][0] * self.trip_length_preference) for j in J])
                    prob = prob / sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i, J[idx]] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = self.demand_input[i, j] if (i, j) in self.demand_input else \
                        self.demand_input['default']

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception("demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.price
            for t in range(0, self.duration):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(self.static_demand[i, j] * self.demand_ratio[i, j][t])
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(2) + 1) * self.demand_time[i, j][t]
                    trip_attr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        print(f'Total demand: {sum([demand[i, j][t] for i, j in demand for t in range(0, self.duration, self.tstep)])}')
        return trip_attr

    def set_taxi_lines(self):
        """
        Method to upload the taxi lines generated in node_routes in sumo
        """

        for node in self.G.nodes:
            for edge_o in self.regions_sumo[node]['in_edges']:
                for edge_d in self.regions_sumo[node]['out_edges']:
                    route_id = edge_o.getID() + edge_d.getID() + 'init'
                    traci.route.add(route_id, [edge_o.getID(), edge_d.getID()])

    def set_taxi_distribution(self):
        """
        Method to uniformly assign the initial number of taxis in the net to the nodes
        accInit is the desired initial number of taxi
        """
        for node in self.G.nodes:
            taxi_num = 0
            while taxi_num < self.acc_init:
                taxi_id = 'taxi' + str(node) + '#' + str(taxi_num)
                edge_o = self.regions_sumo[node]['in_edges'][np.random.randint(len(self.regions_sumo[node]['in_edges']))]
                edge_d = self.regions_sumo[node]['out_edges'][np.random.randint(len(self.regions_sumo[node]['out_edges']))]
                route_id = edge_o.getID() + edge_d.getID() + 'init'
                traci.vehicle.add(vehID=taxi_id, typeID='taxi', routeID=route_id)
                taxi_num += 1
