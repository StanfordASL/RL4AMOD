from src.misc.utils import dictsum
from src.algos.reb_flow_solver import solveRebFlow
from src.algos.base import BaseAlgorithm
from torch.distributions import Dirichlet
import torch

class RandomBaseline(BaseAlgorithm):
    def __init__(self, **kwargs):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        super().__init__()
        self.cplexpath = kwargs.get('cplexpath')
        self.directory = kwargs.get('directory')
    
    def select_action(self, env):
        """
        Implements the random baseline for rebalancing.
        """
        action_rl = torch.ones(env.nregion)
        action_rl = Dirichlet(action_rl).sample().tolist()
        desired_acc = {
            env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time +1))
            for i in range(len(env.region))
        }
        reb_action = solveRebFlow(env, self.directory, desired_acc, self.cplexpath)
        return reb_action
