from src.misc.utils import dictsum
from src.algos.reb_flow_solver import solveRebFlow
from src.algos.base import BaseAlgorithm


class EqualDistribution(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
        self.policy_name = kwargs.get('policy_name')

    def select_action(self, env):
        """
        Implements the Equal Distribution (ED) baseline for rebalancing.
        """
        action_rl = [1 / env.nregion for _ in range(env.nregion)]
        desired_acc = {
            env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time +1))
            for i in range(len(env.region))
        }
        reb_action = solveRebFlow(env, self.directory, desired_acc, self.cplexpath)
        return reb_action
