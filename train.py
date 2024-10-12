import hydra
from omegaconf import DictConfig
import os 
import torch
import json
from hydra import initialize, compose

def setup_sumo(cfg):
    from src.envs.sim.sumo_env import Scenario, AMoD, GNNParser
    cfg.simulator.cplexpath = cfg.model.cplexpath
    if not cfg.simulator.directory:
        cfg.simulator.directory = f"{cfg.model.name}/{cfg.simulator.city}"
    cfg = cfg.simulator
    scenario_path = 'src/envs/data'
    cfg.sumocfg_file = f'{scenario_path}/{cfg.city}/{cfg.sumocfg_file}'
    cfg.net_file = f'{scenario_path}/{cfg.city}/{cfg.net_file}'
    demand_file = f'src/envs/data/scenario_lux{cfg.num_regions}.json'
    aggregated_demand = not cfg.random_od

    scenario = Scenario(
        num_cluster=cfg.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
        sumo_net_file=cfg.net_file, acc_init=cfg.acc_init, sd=cfg.seed, demand_ratio=cfg.demand_ratio,
        time_start=cfg.time_start, time_horizon=cfg.time_horizon, duration=cfg.duration,
        tstep=cfg.matching_tstep, max_waiting_time=cfg.max_waiting_time
    )
    env = AMoD(scenario, cfg=cfg, beta=cfg.beta)
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
    if model_name == "sac" or model_name =="cql":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser).to(device)
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser).to(device)
    elif model_name == "iql":
        from src.algos.iql import IQL
        return IQL(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser).to(device)
    elif model_name == "bc":
        from src.algos.bc import BC
        return BC(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser).to(device)
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")

def setup_dataset(cfg, env, device):
    from src.algos.sac import ReplayData

    if cfg.simulator.name == "sumo":
        origin = []
        destination = []
        for o in range(env.scenario.adjacency_matrix.shape[0]):
            for d in range(env.scenario.adjacency_matrix.shape[1]):
                if env.scenario.adjacency_matrix[o, d] == 1:
                    origin.append(o)
                    destination.append(d)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])

    else: 
        with open(f"src/envs/data/macro/scenario_{cfg.simulator.city}.json", "r") as file:
            data = json.load(file)

        edge_index = torch.vstack(
            (
                torch.tensor([edge["i"] for edge in data["topology_graph"]]).view(1, -1),
                torch.tensor([edge["j"] for edge in data["topology_graph"]]).view(1, -1),
            )
        ).long()
    
    Dataset = ReplayData(device=device)
    Dataset.create_dataset(
        edge_index=edge_index,
        memory_path=cfg.model.data_path,
        rew_scale=cfg.model.rew_scale,
        size=cfg.model.samples_buffer,
    )

    return Dataset

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

def load_actor_weights(model, path):
    full_model_state = torch.load(f"ckpt/{path}.pth")

    actor_encoder_state = {
        k.replace("actor.", ""): v
        for k, v in full_model_state["model"].items()
        if "actor" in k
    }
    model.actor.load_state_dict(actor_encoder_state)
    return model

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

    if hasattr(cfg.model, "pretrained_path"):
        if cfg.model.pretrained_path is not None:
            model = load_actor_weights(model, cfg.model.pretrained_path)

    if hasattr(cfg.model, "data_path"):
        Dataset = setup_dataset(cfg, env, device)
        model.learn(cfg, Dataset)
    else:
        model.learn(cfg)

if __name__ == "__main__":
    main()
