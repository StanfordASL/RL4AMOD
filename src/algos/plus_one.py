from src.algos.reb_flow_solver import solveRebFlow
from src.algos.base import BaseAlgorithm


class PlusOneBaseline(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        super().__init__()
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
        self.policy_name = kwargs.get('policy_name')

    def select_action(self, env):
        """
        Implements the Plus One (plus_one) baseline.
        """
  
        time = len(env.obs[0][3])-1
        action = [0] * env.nregion
        acc = [env.acc[region][time] for region in range(env.nregion)]
        for k in range(len(env.edges)):
            o, d = env.edges[k]
            pax_action = env.paxAction[k]
            if pax_action == 0:
                continue
            action[o] += 1

        for region_o in range(env.nregion):
            dacc = action[region_o]
            if dacc == 0:
                continue
            rebTime = dict()
            for region_d in range(env.nregion):
                if region_d == region_o:
                    continue
                rebTime[(region_o, region_d)] = env.rebTime[(region_o, region_d)][time - 1]
            rebTime = dict(sorted(rebTime.items(), key=lambda item: item[1]))
            flow = iter(rebTime)
            region_d = next(flow)[1]
            reb_taxi = 0
            while reb_taxi < dacc:
                if acc[region_d] > 0:
                    action[region_d] -= 1
                    acc[region_d] -= 1
                    acc[region_o] += 1
                    reb_taxi += 1
                else:
                    try:
                        region_d = next(flow)[1]
                    except:
                        break

        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        desired_acc = {env.region[i]: int(acc[i]) for i in range(len(env.region))}
        # solve minimum rebalancing distance problem
        reb_action = solveRebFlow(env, self.directory, desired_acc, self.cplexpath)

        return reb_action