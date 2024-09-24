import hydra
from omegaconf import DictConfig
import os 
import torch
import json
from hydra import initialize, compose
def setup_sumo(cfg):
    from src.envs.sim.sumo_env import Scenario, AMoD, GNNParser
    
    cfg = cfg.simulator
    cfg.simulator.cplexpath = cfg.model.cplexpath
    demand_file = f'src/envs/data/scenario_lux{cfg.num_regions}.json'
    aggregated_demand = not cfg.random_od
    scenario_path = 'src/envs/data/LuSTScenario/'
    net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')

    scenario = Scenario(num_cluster=cfg.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
                sumo_net_file=net_file, acc_init=cfg.acc_init, sd=cfg.seed, demand_ratio=cfg.demand_ratio,
                time_start=cfg.time_start, time_horizon=cfg.time_horizon, duration=cfg.duration,
                tstep=cfg.matching_tstep, max_waiting_time=cfg.max_waiting_time)
    env = AMoD(scenario, beta=cfg.beta)
    parser = GNNParser(env, T=cfg.time_horizon, json_file=demand_file)

    return env, parser

def setup_macro(cfg):
    from src.envs.sim.macro_env import Scenario, AMoD, GNNParser
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)
    
    cfg.simulator.cplexpath = cfg.model.cplexpath

    cfg = cfg.simulator
    city = cfg.city
     
    scenario = Scenario(
    json_file=f"src/envs/data/macro/scenario_{city}.json",
    demand_ratio=calibrated_params[city]["demand_ratio"],
    json_hr=calibrated_params[city]["json_hr"],
    sd=cfg.seed,
    json_tstep=cfg.json_tsetp,
    tf=cfg.max_steps,
    )
    env = AMoD(scenario, cfg = cfg, beta = calibrated_params[city]["beta"])
    parser = GNNParser(env, T=cfg.time_horizon, json_file=f"src/envs/data/macro/scenario_{city}.json")
    return env, parser

def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name
    cfg = cfg.model
    if model_name == "sac":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser).to(device)
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser).to(device)
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")

def train(config): 
    """
    for colab tutorial
    """

    with initialize(config_path="src/config"):
        cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration

    if cfg.simulator.name == "sumo":
        env, parser = setup_sumo(cfg)
    elif cfg.simulator.name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {cfg.simulator.name}")
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = setup_model(cfg, env, parser, device)
    model.learn(cfg)

@hydra.main(version_base=None, config_path="src/config/", config_name="config")
def main(cfg: DictConfig):
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        env, parser = setup_sumo(cfg)

    elif simulator_name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = setup_model(cfg, env, parser, device)

    model.learn(cfg)

if __name__ == "__main__":
    main()
