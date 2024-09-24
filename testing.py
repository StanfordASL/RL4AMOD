import hydra
from omegaconf import DictConfig
import torch
import json
import numpy as np
from hydra import initialize, compose
from src.algos.registry import get_model

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
    if not cfg.simulator.directory:
        cfg.simulator.directory = f"{cfg.model.name}/{cfg.simulator.city}"
    cfg = cfg.simulator
    city = cfg.city
    scenario = Scenario(
        json_file=f"src/envs/data/macro/scenario_{city}.json",
        demand_ratio=calibrated_params[city]["demand_ratio"],
        json_hr=calibrated_params[city]["json_hr"],
        sd=cfg.seed,
        json_tstep=calibrated_params[city]["test_tstep"],
        tf=cfg.max_steps,
    )
    env = AMoD(scenario, cfg = cfg, beta = calibrated_params[city]["beta"])
    parser = GNNParser(env, T=cfg.time_horizon, json_file=f"src/envs/data/macro/scenario_{city}.json")
    return env, parser

def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name

    if model_name == "sac":
        from src.algos.sac import SAC
        model= SAC(env=env, input_size=cfg.model.input_size, cfg=cfg.model, parser=parser).to(device)
        # model.load_checkpoint(path=f"ckpt/{cfg.model.agent_name}.pth")
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}_best.pth")
        return model
    
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        model= A2C(env=env, input_size=cfg.model.input_size, parser=parser).to(device)
        model.load_checkpoint(path=f"ckpt/{cfg.model.checkpoint_path}_best.pth")
        return model
    
    else:
        model_class = get_model(model_name)
        
        model_kwargs = {
            "cplexpath": cfg.simulator.cplexpath,
            "directory": cfg.simulator.directory,
            "T": cfg.simulator.time_horizon,
            "policy_name": cfg.model.agent_name
        }
        
        return model_class(**model_kwargs)

def setup_net(cfg):
    if cfg.model.use_LSTM:
        from src.nets.actor import GNNActorLSTM as GNNActor
        from src.nets.critic import GNNCriticLSTM as GNNCritic
    else:
        from src.nets.actor import GNNActor
        from src.nets.critic import GNNCritic

        models = {
        "actor": GNNActor,
        "critic": GNNCritic,
        }
    return models


def test(config):
    '''
    for Colab tutorial
    '''
    with initialize(config_path="src/config"):
        cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration
        
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
    
    print(f'Testing model {cfg.model.name} on {cfg.simulator.name} environment')
    episode_reward, episode_served_demand, episode_rebalancing_cost, inflows = model.test(cfg.model.test_episodes, env)

    print('Mean Episode Profit ($): ', np.mean(episode_reward))
    print('Mean Episode Served Demand- Proit($): ', np.mean(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost))

    inflows = np.mean(inflows, axis=0)
    no_reb_reward = 27.6
    #no_reb_demand = 1599.6
    no_reb_demand = 27.6
    no_reb_cost = 0.0
    mean_reward = np.mean(episode_reward)
    mean_served_demand = np.mean(episode_served_demand)
    mean_rebalancing_cost = np.mean(episode_rebalancing_cost)

    mean_reward = round(mean_reward/1000,2)
    mean_served_demand = round(mean_served_demand/1000,2)
    mean_rebalancing_cost = round(mean_rebalancing_cost/1000,2)
    labels = ['Overall Profit', 'Served Demand Profit', 'Rebalancing Cost']
    rl_means = [mean_reward, mean_served_demand, mean_rebalancing_cost]
    
    no_control = [no_reb_reward, no_reb_demand, no_reb_cost]
    
    import matplotlib.pyplot as plt
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    #fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax1.bar(x - width/2, rl_means, width, label=cfg.model.name, color="#0072BD", capsize=5)
    rects2 = ax1.bar(x + width/2, no_control, width, label='No Control', color="#A2142F")
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('$, x10^3')
    ax1.set_title(f'Comparison of {cfg.model.name} vs No Control')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Function to add value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Adding value labels to each bar
    add_value_labels(rects1)
    add_value_labels(rects2)

    #plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #plt.show()
    

    open_reqest = {0: 0,
        1: 414.0,
        2: 0,
        3: 0,
        4: 0,
        5: 49756.49999999998,
        6: 9948.600000000006,
        7: 98.99999999999999,
        8: 198.00000000000003,
        9: 881.9999999999998,
        10: 1232.9999999999993,
        11: 6492.600000000001,
        12: 23293.80000000004,
        13: 170.99999999999997}
    
    #open_reqest = {k: v / max(open_reqest.values()) for k,v in open_reqest.items()}

    #inflows = inflows / max(inflows)

    labels = range(14)
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    r1 = np.arange(14)
    r2 = [x + width for x in r1]

    #fig, ax = plt.subplots(figsize=(8, 5))
    ax2.bar(r2, inflows, width, label='Rebalancing Flows', color="#0072BD")
    ax3 = ax2.twinx()  # Create a second y-axis
    ax3.bar(r1, open_reqest.values(), width, label='Profit', color="#A2142F")

    # Add labels and title to the second plot
    ax2.set_xlabel('Regions')
    ax2.set_ylabel('Flows', color="#0072BD")
    ax3.set_ylabel('Profit', color="#A2142F")
    ax2.set_title('Comparison of Incoming Rebalancing Flows vs Profit')
    ax2.set_xticks(r1)
    ax2.set_xticklabels(labels)
    ax2.tick_params(axis='y', labelcolor="#0072BD")
    ax3.tick_params(axis='y', labelcolor="#A2142F")
    #ax2.legend()
    #ax3.legend()

    plt.tight_layout()  
    plt.show()

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
    
    print('Testing...')
    episode_reward, episode_served_demand, episode_rebalancing_cost, episode_inflows = model.test(cfg.model.test_episodes, env)

    print('Mean Episode Profit ($): ', np.mean(episode_reward), 'Std Episode Reward: ', np.std(episode_reward))
    print('Mean Episode Served Demand($): ', np.mean(episode_served_demand), 'Std Episode Served Demand: ', np.std(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost), 'Std Episode Rebalancing Cost: ', np.std(episode_rebalancing_cost))

    ##TODO: ADD VISUALIZATION

if __name__ == "__main__":
    main()
    