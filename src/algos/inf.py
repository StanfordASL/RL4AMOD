from src.algos.base import BaseAlgorithm
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, value, LpStatus, PULP_CBC_CMD

class INF(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """

        self.max_reb = kwargs.get('max_reb')
        self.roh = kwargs.get('roh')
    
    def select_action(self, env):
        t = env.time

        accInitTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
        edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

        region = [i for (i, n) in accInitTuple]

        edgeAttr += [(i, i, 0) for i in region]

        time = {(i, j): t for (i, j, t) in edgeAttr}

        vehicles = {i: v for (i, v) in accInitTuple}

        vehicle_region = {}
        vehicle = 0
        for i in region:
            for _ in range(vehicles[i]):
                vehicle_region[vehicle] = i
                vehicle += 1

        open_requests = {}
        for i in env.region:
            open_requests[i] = 0
        for i, j in env.scenario.demand_input:
            for t in range(env.time + 1, env.time + 11):
                 open_requests[i] += env.scenario.demand_input[i, j][t]
        for i in env.region:
            open_requests[i] = round(open_requests[i] / 10)

        # calculate time for each vehicle to each region
        time_vehicle_region = {}
        for vehicle in vehicle_region:
            for r in region:
                time_vehicle_region[vehicle, r] = time[vehicle_region[vehicle], r]

        edge = [(vehicle, r) for vehicle in vehicle_region for r in region]

        model = LpProblem("RebalancingFlowMinimization", LpMaximize)
        
        rebFlow = LpVariable.dicts("x", edge, 0, 1, LpBinary)

        model += lpSum(rebFlow[e] * open_requests[e[1]] * (self.max_reb - time_vehicle_region[e]) for e in edge), "TotalRebalancingFlow"

        # Add constraints
        # only one vehicle can be assigned to one region
        for v in vehicle_region:
            model += lpSum(rebFlow[v, k] for k in region) <= 1
            for j in region:
                model += rebFlow[v, j] * (self.max_reb - time_vehicle_region[v, j]) >= 0

        for i in region:
            model += lpSum(rebFlow[v, i] * (self.max_reb - time_vehicle_region[v, i]) for v in vehicle_region) <= open_requests[i] * self.roh * (self.max_reb**2)
        # Optimize the model
        status = model.solve(PULP_CBC_CMD(msg=False))
   
        if LpStatus[status] == "Optimal":
            flows = {e: 0 for e in env.edges}
            for v in vehicle_region:
                for r in region:
                    if value(rebFlow[v, r]) == 1:
                        i = vehicle_region[v]
                        j = r
                        flows[i, j] += 1

            reb_action = [flows[i, j] for i, j in env.edges]

            return reb_action
