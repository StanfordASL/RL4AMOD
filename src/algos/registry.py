
from src.algos.sac import SAC
from src.algos.a2c import A2C
from src.algos.ed import EqualDistribution
from src.algos.plus_one import PlusOneBaseline
from src.algos.no_reb import NoRebalanceBaseline
from src.algos.random import RandomBaseline
from src.algos.MPC import MPC
from src.algos.dtv import DTV

MODEL_REGISTRY = {
    "sac": SAC,
    "a2c": A2C,
    "equal_distribution": EqualDistribution,
    "plus_one": PlusOneBaseline,
    "no_rebalancing": NoRebalanceBaseline,
    "random": RandomBaseline,
    "mpc": MPC, 
    "dtv": DTV, 
}

def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model or baseline: {model_name}")
    return MODEL_REGISTRY[model_name]
