from src.algos.base import BaseAlgorithm
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value, LpStatus, PULP_CBC_CMD
import traci
class DTV(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
        self.policy_name = kwargs.get('policy_name')

    def select_action(self, env):
        t = env.time

        accInitTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
        edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

        region = [i for (i, n) in accInitTuple]
        # add self loops to edgeAttr with time =0
        edgeAttr += [(i, i, 0) for i in region]
        time = {(i, j): t for (i, j, t) in edgeAttr}

        vehicles = {i: v for (i, v) in accInitTuple}

        #demand = env.waiting_passengers
        #print(demand)
        reservations = traci.person.getTaxiReservations(3)
        #print('reservations', reservations)
        #print('unserved_demand', [env.unserved_demand[i][t] for i in env.unserved_demand])
        #print('waiting_passengers', [env.waiting_passengers[i][t] for i in env.waiting_passengers])

        demand = {}
        for i in env.region: 
            demand[i] = 0
        for trip in reservations:
            persons = trip.persons[0]
            waiting_time = traci.person.getWaitingTime(persons)
            if waiting_time > env.max_waiting_time * 60:
                traci.person.remove(persons)
                persons = ''
            if not persons:
                continue
            o = int(persons[(persons.find('o') + 1):persons.find('d')])
            demand[o] += 1

        #print(demand.values())
        #assert sum(demand.values()) == sum([env.waiting_passengers[i][t] for i in env.waiting_passengers])
     
        num_vehicles = sum(vehicles.values())
        num_requests = sum(demand.values())

        vehicle_region = {}
        vehicle = 0
        for i in region:
            for _ in range(vehicles[i]):
                vehicle_region[vehicle] = i
                vehicle += 1

        # create the same for requests
        request_region = {}
        request = 0
        for i in demand.keys():
            for _ in range(int(demand[i])):
                request_region[request] = i
                request += 1
        
        # calculate time for each vehicle to each request according to the region
        time_vehicle_request = {}
        for vehicle in vehicle_region:
            for request in request_region:
                time_vehicle_request[vehicle, request] = time[
                    vehicle_region[vehicle], request_region[request]
                ]

        edge = [
            (vehicle, request) for vehicle in vehicle_region for request in request_region
        ]


        model = LpProblem("RebalancingFlowMinimization", LpMinimize)
        
        rebFlow = LpVariable.dicts("x", edge, 0, 1, LpBinary)


        model += lpSum(rebFlow[e] * time_vehicle_request[e] for e in edge)
    

        model += lpSum(rebFlow[e] for e in edge) == min(num_vehicles, num_requests)
        

        # only one vehicle can be assigned to one request
        for request in request_region:
            model += lpSum(rebFlow[v, request] for v in vehicle_region) <= 1
            #model.addConstr(quicksum(rebFlow[v, request] for v in vehicle_region) <= 1)

        # only one request can be assigned to one vehicle
        for vehicle in vehicle_region:
            model += lpSum(rebFlow[vehicle, k] for k in request_region) <= 1
            
        # Optimize the model
        status = model.solve(PULP_CBC_CMD(msg=False))
        #print objective value 
        # Check if the solution is optimal
        if LpStatus[status] == "Optimal":
            
            # get rebalancing flows 
            flows = {e: 0 for e in env.edges}
            for v in vehicle_region:
                for r in request_region:
                    if value(rebFlow[v, r]) == 1:
                        i = vehicle_region[v]
                        j = request_region[r]
                        flows[i, j] += 1

        reb_action = [flows[i, j] for i, j in env.edges]

        return reb_action
