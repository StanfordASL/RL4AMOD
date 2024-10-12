"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import subprocess
import os
import sys
import networkx as nx
import json
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib
import traci
import math

from lxml import etree as ET
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import KMeans
from src.misc.utils import mat2str
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data


class AMoD:
    # initialization
    def __init__(self, scenario, cfg, beta=0.2):  # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = scenario
        self.G = scenario.G  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.regions_sumo = scenario.regions_sumo
        self.taxi_routes = scenario.taxi_routes
        self.reservations = list()
        self.reservations_assigned = list()
        self.demand_time = self.scenario.demand_time
        self.rebTime = self.scenario.rebTime
        self.time = 0  # current time
        self.tstep = self.scenario.tstep
        self.duration = scenario.duration  # final time
        self.max_waiting_time = self.scenario.max_waiting_time  # Maximum waiting time of passengers
        self.waiting_time = defaultdict(dict)   # Waiting time per region
        self.demand = defaultdict(dict)  # demand
        self.region = list(self.G)  # set of regions
        self.cfg = cfg
        self.matching_steps = int(self.cfg.matching_tstep * 60 / self.cfg.sumo_tstep)  # sumo steps between each matching
        if scenario.is_meso:
            self.matching_steps -= 1     # In the meso setting one step is done within the reb_step
        self.price = defaultdict(dict)  # price
        self.acc = defaultdict(dict)  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of vehicles arriving at each region, key: i - region, t - time
        self.passengers = defaultdict(dict)     # number of passengers per region
        self.demand_res = defaultdict(dict)  # reservations from sumo assigned to each (origin, destination) pair
        self.rebFlow = defaultdict(dict)  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
            self.demand_res[i, j] = defaultdict(dict)  # reservations from sumo assigned to each (origin, destination) pair
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
            self.waiting_time[n] = defaultdict(float)
            self.passengers[n] = defaultdict(float)
        self.beta = beta * scenario.tstep
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def matching(self, demandAttr=[], CPLEXPATH=None, PATH='', platime_endorm='linux'):
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
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
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

    # pax step
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH='', platime_endorm='linux'):
        tstep = self.tstep
        self.time = int(((traci.simulation.getTime() - self.scenario.time_start * 60) // 60) - tstep)
        t = self.time
        self.reward = 0
        # Step info dictionary initialization
        self.info['served_demand'] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info['revenue'] = 0
        self.info['profit'] = 0
        self.info['rebalancing_cost'] = 0
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
            paxAction = self.matching(demandAttr=demandAttr, CPLEXPATH=CPLEXPATH, PATH=PATH, platime_endorm=platime_endorm)
        self.paxAction = paxAction
        # Serving passengers
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.demand or t not in self.demand[i, j] or self.paxAction[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert paxAction[k] < self.acc[i][t + tstep] + 1e-3
            self.paxAction[k] = min(self.acc[i][t + tstep], int(paxAction[k]))
            # SUMO taxi dispatch
            taxi_match = 0
            while taxi_match < self.paxAction[k]:
                taxi_num = 0
                reservation = self.demand_res[i, j][t][taxi_match]
                reservation_id = reservation.id
                persons = reservation.persons[0]
                taxi, taxi_valid, arrival_time, demand_time = self.dispatch_taxi(taxi_num, o=i, reservation=reservation)
                taxi_id = taxi[0]
                if taxi_valid:
                    self.servedDemand[i, j][t] += 1
                    self.info['served_demand'] += 1
                    self.acc[i][t + tstep] -= 1
                    self.info['revenue'] += (self.price[i, j][t])
                    self.regions_sumo[i]['taxis'].remove(taxi)
                    self.reservations_assigned.append([reservation_id, persons, taxi_id])
                    self.paxFlow[i, j][arrival_time] += 1
                    self.dacc[j][arrival_time] += 1
                    self.info["operating_cost"] += demand_time * self.beta
                    self.reward += (self.price[i, j][t] - demand_time * self.beta)
                taxi_match += 1
        self.info['profit'] += (self.info['revenue']-self.info["operating_cost"])

        self.obs = (self.acc, self.time, self.dacc, self.demand)  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self.obs, max(0, self.reward), done, self.info

    def dispatch_taxi(self, taxi_num, o, reservation):
        """
        Method to dispatch the taxi, given the matching action
        """
        tstep = self.tstep
        t = self.time
        reservation_id = reservation.id
        persons = reservation.persons[0]
        edge_o = reservation.fromEdge
        edge_d = reservation.toEdge
        taxi_valid = True
        if self.scenario.aggregated_demand:
            taxi = self.regions_sumo[o]['taxis'][taxi_num]
            taxi_id = taxi[0]
            # Direction check
            edge = traci.vehicle.getRoadID(taxi_id)
            while edge[1].isdigit != edge_o[1].isdigit:
                if taxi_num == len(self.regions_sumo[o]['taxis']) - 1:
                    taxi = self.regions_sumo[o]['taxis'][0]
                    taxi_id = taxi[0]
                    break
                taxi_num += 1
                taxi = self.regions_sumo[o]['taxis'][taxi_num]
                taxi_id = taxi[0]
                edge = traci.vehicle.getRoadID(taxi_id)
            traci.vehicle.dispatchTaxi(taxi_id, [reservation_id])
            route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi')
            demand_time = int(math.ceil(route.travelTime / (traci.vehicle.getSpeedFactor(taxi_id) * 60)))
        else:
            distance = self.get_taxi_by_distance(region=o, trip=reservation)
            taxi_valid = False
            while not taxi_valid:
                taxi = distance[taxi_num][0]
                taxi_id = taxi[0]
                edge = traci.vehicle.getRoadID(taxi_id)
                route_pickup = traci.simulation.findRoute(edge, edge_o, vType='taxi')
                # Dispatch and taxi remove from the region
                if route_pickup.edges:
                    traci.vehicle.dispatchTaxi(taxi_id, [reservation_id])
                    taxi_valid = True
                else:
                    taxi_num += 1
                    if taxi_num == len(distance):
                        traci.person.remove(personID=persons, reason=3)
                        break
                    else:
                        continue
            route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi')
            demand_time = int(math.ceil((route.travelTime + route_pickup.travelTime) / (traci.vehicle.getSpeedFactor(taxi_id) * 60)))

        arrival_time = t + demand_time  # Use the speed factor to have a more accurate forecast of the travel time
        if arrival_time % tstep != 0:
            arrival_time = (arrival_time // tstep + 1) * tstep
        return taxi, taxi_valid, arrival_time, demand_time

    # reb step
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
                taxi, arrival_time, rebTime = self.reb_taxi(reb_assign, taxis_reb, o=i, d=j)
                self.regions_sumo[i]['taxis'].remove(taxi)
                self.rebFlow[i, j][arrival_time] += 1
                self.dacc[i][arrival_time] += 1
                self.info['rebalancing_cost'] += rebTime * self.beta
                self.info["operating_cost"] += rebTime * self.beta
                self.reward -= rebTime * self.beta
                taxis_reb += 1

        if self.scenario.is_meso:
            traci.simulationStep()
            reservations = traci.person.getTaxiReservations(3)
            reservation_ids = [r.id for r in reservations if 'rebalancing' in r.persons[0]]
            reservation_p = [r.persons[0] for r in reservations if 'rebalancing' in r.persons[0]]
            for taxi_id, person_id in reversed(reb_assign):
                res_id = reservation_ids[reservation_p.index(person_id)]
                traci.vehicle.dispatchTaxi(taxi_id, [res_id])

        self.obs = (self.acc, self.time, self.dacc, self.demand)  # use self.time to index the next time step
        # Travel time update from sumo
        self.update_routes(time=t+tstep)
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]

        done = (self.duration == t + tstep)  # if the episode is completed
        return self.obs, self.reward, done, self.info

    def step(self, pax_action=None, reb_action=None):
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        # Take action in environment
        rew = 0
        # Rebalancing step
        obs, rebreward, done, info = self.reb_step(reb_action)
        rew += rebreward
        # Check for episode end
        if done:
            traci.close()
            return obs, rew, done, info
        # Matching step
        self.sumo_steps()
        obs, paxreward, done, info = self.pax_step(paxAction=pax_action, CPLEXPATH=self.cfg.cplexpath, PATH=self.cfg.directory)
        rew += paxreward
        return obs, rew, done, info

    def sumo_steps(self):
        """
        Method to run sumo simulation steps until the next decision process
        """
        sumo_step = 0
        while sumo_step < self.matching_steps:
            traci.simulationStep()
            sumo_step += 1

    def reb_taxi(self, reb_assign, taxis_reb, o, d):
        """
        Method to rebalance taxis
        """
        tstep = self.tstep
        t = self.time
        taxi = self.regions_sumo[o]['taxis'][0]
        taxi_id = taxi[0]
        edge_d = self.regions_sumo[d]['in_edges'][np.random.randint(len(self.regions_sumo[d]['in_edges']))].getID()
        edge_d_length = traci.lane.getLength(edge_d + '_0')
        edge_o = traci.vehicle.getRoadID(taxi_id)
        route = traci.simulation.findRoute(edge_o, edge_d, vType='taxi', routingMode=1)
        rebTime = int(math.ceil(route.travelTime / (traci.vehicle.getSpeedFactor(taxi_id) * 60)))
        arrival_time = t + rebTime  # Use the speed factor to have a more accurate forecast of the travel time
        if arrival_time % tstep != 0:
            arrival_time = (arrival_time // tstep + 1) * tstep
        if self.scenario.is_meso:
            edge_o_length = traci.lane.getLength(edge_o + '_0')
            reb_id = taxi_id + 'o' + str(o) + 'd' + str(d) + '#' + str(taxis_reb) + '_rebalancing'
            traci.person.add(personID=reb_id, edgeID=edge_o, pos=edge_o_length)
            traci.person.appendDrivingStage(personID=reb_id, toEdge=edge_d, lines='taxi')
            reb_assign.append((taxi_id, reb_id))
        else:
            traci.vehicle.resume(taxi_id)
            traci.vehicle.setRoute(taxi_id, route.edges)
            traci.vehicle.setStop(taxi_id, edge_d, pos=edge_d_length, flags=1)  # Set the stop in the new location
            traci.vehicle.setStopParameter(taxi_id, 0, 'actType', 'rebalancing')  # Set the taxi condition to rebalancing
        return taxi, arrival_time, rebTime

    def reset(self):
        """
        Method to reset the environment with matching in before first env step
        """
        # reset the episode
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
                self.rebTime[i, j][0] = self.demand_time[i, j][0]
            else:
                self.demand_time[i, j][0] = 0
                self.rebTime[i, j][0] = 0

        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
            self.regions_sumo[n]['taxis'] = list()

        # Initialize taxis in the network
        self.scenario.set_taxi_lines()
        self.scenario.set_taxi_distribution()

        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        # 1st pax step
        if self.scenario.is_meso:
            traci.simulationStep()
        self.sumo_steps()
        obs, paxreward, done, info = self.pax_step(CPLEXPATH=self.cfg.cplexpath, PATH=self.cfg.directory)
        return obs, paxreward

    def reset_old(self):
        """
        Method to reset the environment without matching (used for MPC)
        """
        # reset the episode
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
                self.rebTime[i, j][0] = self.demand_time[i, j][0]
            else:
                self.demand_time[i, j][0] = 0
                self.rebTime[i, j][0] = 0

        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
            self.regions_sumo[n]['taxis'] = list()

        # Initialize taxis in the network
        self.scenario.set_taxi_lines()
        self.scenario.set_taxi_distribution()

        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        # 1st pax step
        if self.scenario.is_meso:
            traci.simulationStep()
        self.sumo_steps()
        return self.obs

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
            waiting_time = traci.person.getWaitingTime(persons)
            if waiting_time > self.max_waiting_time * 60:
                traci.person.remove(persons)
                persons = ''
            if not persons:
                continue
            o = int(persons[(persons.find('o') + 1):persons.find('d')])
            d = int(persons[(persons.find('d') + 1):persons.find('#')])
            price = self.price[o, d][t]
            # Update the price in case the route has not the same origin-destination as the initial one (due to randomness)
            if price == 0:
                price = self.scenario.get_price(t, trip)
            state = trip.state
            # Check for already assigned trips
            if state >= 4:
                continue
            self.passengers[o][self.time] += 1
            self.waiting_time[o][self.time] += waiting_time / 60
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
        # Compute average waiting time per region
        for n in range(self.nregion):
            if self.waiting_time[n][self.time] != 0:
                self.waiting_time[n][self.time] /= self.passengers[n][self.time]
            else:
                self.passengers[n][self.time] = 0

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
                self.rebTime[o, d][time] = self.demand_time[o, d][time]
            else:
                self.demand_time[o, d][time] = 0
                self.rebTime[o, d][time] = 0
        self.taxi_routes = new_routes

    def set_taxi_to_region(self, taxi_ids):
        """
        Method to assign the avaiable taxi to the relative region
        """
        taxi_info = []
        for taxi_id in taxi_ids:
            stop = traci.vehicle.getStops(taxi_id)
            stop_info = stop[0].actType
            # Ignore in-rebalancing taxis
            if stop_info == 'rebalancing':
                parking = self.check_parking(taxi_id, stop)
                if not parking:
                    continue
            position = traci.vehicle.getPosition(taxi_id)
            taxi_info.append((taxi_id, stop_info, position))

        # Batch process the position
        positions = np.array([info[2] for info in taxi_info])
        regions = np.array([])
        if positions.size > 0:
            regions = self.scenario.cluster_alg.predict(positions)
        # Assign taxis to regions
        for (taxi_id, stop_info, _), region in zip(taxi_info, regions):
            if (taxi_id, stop_info) not in self.regions_sumo[region]['taxis']:
                self.regions_sumo[region]['taxis'].append((taxi_id, stop_info))

    def get_taxi_by_distance(self, region, trip):
        """
        Method to find the taxi in the region closest to the person to be picked-up
        """
        taxis = [taxi for taxi in self.regions_sumo[region]['taxis']]
        distance = []
        for taxi in taxis:
            taxi_pos = traci.vehicle.getPosition(taxi[0])
            person_pos = traci.person.getPosition(trip.persons[0])
            taxi_distance = math.sqrt((taxi_pos[0] - person_pos[0])**2 + (taxi_pos[1] - person_pos[1])**2)
            distance.append((taxi, taxi_distance))
        return sorted(distance, key=lambda x: x[1])

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
    def __init__(self, num_cluster=4, duration=2, sd=None, demand_ratio=None, json_file=None, aggregated_demand=True, time_start=7,
                 time_horizon=10, tstep=2, max_waiting_time=5,varying_time=False, json_regions=None, sumo_net_file=None, acc_init=100):
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
            sumo_net_file = 'data/lux/lust.net.xml'

        self.is_meso = 'meso' in sumo_net_file
        self.sumo_net = sumolib.net.readNet(sumo_net_file)
        self.adjacency_matrix = np.array
        self.regions_sumo, self.cluster_alg = self.sumo_net_clustering()
        self.taxi_routes = self.get_taxi_routes()
        self.G = nx.complete_graph(self.N)
        self.G = self.G.to_directed()
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        self.acc_init = acc_init  # Initial vehicle distribution per node (Settato esternamente dal main e non più dal json)
        for n in self.G.nodes:
            self.G.nodes[n]['accInit'] = acc_init

        self.tstep = tstep
        self.thorizon = time_horizon
        self.max_waiting_time = max_waiting_time
        self.demand_input = defaultdict(dict)
        self.json_regions = json_regions
        self.price = defaultdict(dict)
        self.alpha = 0
        self.demand_time = defaultdict(dict)
        self.rebTime = defaultdict(dict)
        self.time_start = time_start * 60
        self.duration = duration * 60

        # Time demand between nodes initialization
        for i, j in self.edges:
            self.demand_time[i, j] = defaultdict(int)
            self.rebTime[i, j] = defaultdict(int)
            self.price[i, j] = defaultdict(int)
            self.demand_input[i, j] = defaultdict(int)

        # Demand creation (from json)
        self.varying_time = varying_time
        self.aggregated_demand = aggregated_demand      # Passengers appears in random edges inside the region
        self.is_json = True
        if json_file is None:
            self.is_json = False
        with open(json_file, "r") as file:
            data = json.load(file)
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
                    self.rebTime[o, d][t] = self.demand_time[o, d][t]
                else:
                    self.demand_time[o, d][t] = 0
                    self.rebTime[o, d][t] = 0
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
        nodes_pos_fit = [node.getCoord() for node in nodes]
        edges = self.sumo_net.getEdges()
        # Nodes and edges filter
        nodes, edges = self.filter_nodes_edges(nodes, edges)
        nodes_id = [node.getID() for node in nodes]
        nodes_pos = [node.getCoord() for node in nodes]

        # Clustering
        kmeans = KMeans(n_clusters=self.N, random_state=0, n_init=10)
        kmeans.fit(np.array(nodes_pos_fit))
        labels = kmeans.predict(np.array(nodes_pos))
        regions_sumo = list()
        for region in range(self.N):
            nodes_cluster = [node for idx, node in enumerate(nodes) if labels[idx] == region]
            nodes_id_cluster = [node_id for idx, node_id in enumerate(nodes_id) if labels[idx] == region]
            nodes_pos_cluster = [node_pos for idx, node_pos in enumerate(nodes_pos) if labels[idx] == region]
            cluster_center = kmeans.cluster_centers_[region]
            centroids = kmeans.cluster_centers_
            # Adjacency matrix calculation
            distances = cdist(centroids, centroids, 'euclidean')
            self.adjacency_matrix = (distances < 3500).astype(int)
            edges_cluster = [edge for edge in edges if edge.getFromNode() in nodes_cluster or edge.getToNode() in nodes_cluster]
            # Find the closest node to the region center
            distance = np.linalg.norm(np.array(nodes_pos_cluster) - cluster_center, axis=1)
            min_idx = np.argmin(distance)
            node = nodes_cluster[min_idx]
            in_edges = [incoming for incoming in node.getIncoming()]
            out_edges = [outgoing for outgoing in node.getOutgoing()]
            # Append the region info
            regions_sumo.append({'id': nodes_id_cluster, 'position': np.array(nodes_pos_cluster),
                                 'id_center': node.getID(), 'position_center': nodes_pos_cluster[min_idx],
                                 'edges': edges_cluster, 'in_edges': in_edges, 'out_edges': out_edges,
                                 'taxis': []}
                                )

        # Cluster centers edges
        return regions_sumo, kmeans

    @staticmethod
    def filter_nodes_edges(nodes, edges):
        """
        Method to filter the nodes and the edges of the network
        """
        # Filter dead_end nodes
        nodes = [node for node in nodes if node.getType() != 'dead_end' and node.getType() != 'unregulated']
        # Edges filter
        edges = [edge for edge in edges if edge.getFromNode() in nodes and edge.getToNode() in nodes]
        edges = [edge for edge in edges if 'motorway' not in edge.getType()]  # Remove motorway edges
        # Nodes linked to removed edges filtered
        nodes = [
            node for node in nodes
            if sum(1 for outgoing in node.getOutgoing() if outgoing in edges) >= 2 and
               sum(1 for incoming in node.getIncoming() if incoming in edges) >= 2
        ]
        # Edges linked to removed nodes filtered
        edges = [edge for edge in edges if edge.getFromNode() in nodes and edge.getToNode() in nodes]
        return nodes, edges

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
        tree.write("data/lux/input/routes/taxi_initialization.rou.xml", xml_declaration=True, pretty_print=True)

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
                            depart_time = np.random.randint(t0, tf)  # Time instant at which the person appears in the network
                            for person_num in range(demand[i, j][t]):
                                if self.aggregated_demand:
                                    edge_o = self.regions_sumo[i]['out_edges'][np.random.randint(len(self.regions_sumo[i]['out_edges']))]
                                    edge_d = self.regions_sumo[j]['in_edges'][np.random.randint(len(self.regions_sumo[j]['in_edges']))]
                                else:
                                    edge_o = self.regions_sumo[i]['edges'][np.random.randint(len(self.regions_sumo[i]['edges']))]
                                    edge_d = self.regions_sumo[j]['edges'][np.random.randint(len(self.regions_sumo[j]['edges']))]
                                    while edge_o == edge_d:
                                        edge_d = self.regions_sumo[j]['edges'][np.random.randint(len(self.regions_sumo[j]['edges']))]  # In case the edge crosses the boarder between two regions
                                person_id = 'p' + str(t) + 'o' + str(i) + 'd' + str(j) + '#' + str(person_num)
                                traci.person.add(personID=person_id, edgeID=edge_o.getID(), depart=depart_time, pos=0)
                                traci.person.setColor(typeID=person_id, color=(237, 177, 32))
                                traci.person.appendDrivingStage(personID=person_id, toEdge=edge_d.getID(), lines='taxi')
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
                    prob = np.array([np.math.exp(-self.rebTime[i, j][0] * self.trip_length_preference) for j in J])
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

        #print(f'Total demand: {sum([demand[i, j][t] for i, j in demand for t in range(0, self.duration, self.tstep)])}')
        return trip_attr

    def get_price(self, t, trip):
        """
        Method to compute the price, given the time and length of the route
        """
        t = t / 60
        route = traci.simulation.findRoute(trip.fromEdge, trip.toEdge, 'taxi')
        length = sumolib.route.getLength(self.sumo_net, route.edges)
        if t <= 5:
            price = 2.75 + 2.86 * length / 1000  # Night tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
        else:
            price = 2.5 + 2.60 * length / 1000  # Daily tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
        return price

    def set_taxi_lines(self):
        """
        Method to upload the taxi lines generated in node_routes in sumo
        """
        for node in self.G.nodes:
            for edge_o in self.regions_sumo[node]['in_edges']:
                for edge_d in self.regions_sumo[node]['out_edges']:
                    route_id = edge_o.getID() + edge_d.getID() + 'init'
                    route = traci.simulation.findRoute(edge_o.getID(), edge_d.getID())
                    traci.route.add(route_id, route.edges)
                    if not route.edges:
                        raise Exception(f"{edge_o.getID()} and {edge_d.getID()} can't be connected")

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




class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None):
        super().__init__()
        self.env = env
        self.T = self.env.scenario.thorizon
        self.s_acc, self.s_dem = self.get_scaling_factors()     # ADDED
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1]*self.s_acc for n in self.env.region]).view(1, 1, self.env.nregion).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s_acc for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregion).float(),
            torch.tensor([[sum([(self.env.scenario.demand_input[i,j][t])*(self.env.price[i,j][t])*self.s_dem \
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregion).float()),
              dim=1).squeeze(0).view(1+self.T+self.T, self.env.nregion).T

        ###################
        # ADDED
        # Edge index self-connected tensor definition
        origin = []
        destination = []
        for o in range(self.env.scenario.adjacency_matrix.shape[0]):
            for d in range(self.env.scenario.adjacency_matrix.shape[1]):
                if self.env.scenario.adjacency_matrix[o, d] == 1:
                    origin.append(o)
                    destination.append(d)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        # edge_index = torch.cat([torch.tensor([self.env.region]), torch.tensor([self.env.region])])    # Just local region information
        ##################


        data = Data(x, edge_index)
        return data

    def get_scaling_factors(self):
        t0 = 0
        tf = self.env.scenario.duration
        time = [t for t in range(t0, tf)]
        acc_tot = (self.env.acc[0][0] * self.env.nregion)
        demand = self.env.scenario.demand_input
        price = self.env.scenario.price
        demand_max = max([max([demand[key][t] for key in demand]) for t in time])
        price_max = max([max([price[key][t] for key in price]) for t in time])
        return 2/acc_tot, 1/(1.2 * demand_max * price_max)
