from src.algos.base import BaseAlgorithm

class NoRebalanceBaseline(BaseAlgorithm):
    def __init__(self, **kwargs):

        super().__init__()

    def select_action(self, env):
        """
        Implements the No Rebalancing (no_reb) strategy.
        """
        reb_action = [0] * len(env.edges)
        return reb_action
