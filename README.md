# Data-driven Methods for Network-level Control of Autonomous Mobility-on-Demand Systems Across Scales

<img align="center" src="figures/figure.png" width="1500"/></td> <br/>

Join us in moving the first steps toward the creation of publicly available benchmarks, datasets, and simulators for network-level coordination of MoD systems.

We aim to provide:
- Openly accessible simulation platforms across different fidelity levels (macroscopic, mesoscopic, microscopic)
- Calibrated scenarios
- Implementation of fleet coordination algorithms
- A common validation process that allows for direct comparison between different approaches (i.e., leaderboard-based)
- Standardized pipeline for contributed algorithms, scenarios, tools, etc.


## Prerequisites

* We recommend using CPLEX, however if you don't have access to a IBM cplex installation, we provide an alternative with the free python package PuLP. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)


* You will need to have a working installation of SUMO (Simulation of Urban MObility). It is an open source microscopic and continous multi-modal traffic simulation package used to handle large networks. The repo is based on the version 1.20.0. Find the information for SUMO installation [here](https://sumo.dlr.de/docs/Installing/index.html)


* To install all required dependencies, run
```
pip install -r requirements.txt
```
## Contents
`RL4AMOD`  
`├── testing.py`: test main file<br>
`├── train.py`: RL agents train main file <br>
`├── src/`  
`│   ├── algos/`  
`│   │   ├── sac.py`: Pytorch implementation of SAC<br>
`│   │   ├── a2c.py`: PyTorch implementation of A2C<br>
`│   │   ├── base.py`: Base class for controller<br>
`│   │   ├── ed.py`: Equal distribution controller<br>
`│   │   ├── `...<br>
`│   │   └── registry.py`: models registration function<br>
`│   ├── nets/`  
`│   │   ├── actor.py`: Pytorch implementation of a GNN-based actor<br>
`│   │   ├── critic.py`: Pytorch implementation of a GNN-based critic<br>
`│   ├── envs/`  
`│   │   ├── data/`: data for the macroscopic and mesoscopic scenarios<br>
`│   │   └── sim/`  
`│   │   │    ├── macro_env.py`: implementation of a macroscopic simulator for AMoD systems system<br>
`│   │   │    └── sumo_env.py`: implementation of a SUMO-based mesoscopic simulator for AMoD systems<br>
`│   ├── misc/`: helper functions<br>
`│   ├── config/`: default parameters for the simulator and the models<br>
`│   ├── cplex_mod/`: CPLEX formulation of Rebalancing, Matching and MPC problems<br>
`├── saved_files/`: directory for saving results, logging, etc<br>

##  Configuration parameters
To run a training or a testing, firstly the simulator and the model type must be selected with the config arguments
```
config arguments:
    simulator           simulator fidelity between 'macro' and 'sumo' (default: macro)
    model               selected model: choose between the models in src/algos/registry.py file (default: sac)
```
### Simulators parameters
You need to pass the following argument to set a simulator parameter:
```
simulator.{arg}=value
```
Use the following argument for macroscopic simulator:
```
simulator=macro
```
```
macro simulator arguments:
    seed                random seed (default: 10)
    demand_ratio        ratio of demand (default: 0.5)
    json_hr             hour of the day for JSON configuration (default: 7)
    json_tstep          minutes per timestep (default: 3 min)
    beta                cost of rebalancing (default: 0.5)
    city                city: defines the city (default: 'nyc_brooklyn')
    max_steps           number of steps per episode (default: T=20)
    time_horizon        time horizon for demand and arriving vehicle forecast (default: 6)
    directory           directory where to save files
```
Use the following argument for mesoscopic simulator:
```
simulator=sumo
```
```
sumo simulator arguments:
    sumocfg_file        define the SUMO configuration file
    net_file            define the city network file
    seed                random seed (default: 10)
    demand_ratio        demand ratio (default: 0.8)
    time_start          simulation start time in hours (default: 7)
    duration            episode duration in hours (default: 2 hr)
    time_horizon        matching steps in the future for demand and arriving vehicle forecast (default: 10 min)
    matching_tstep      minutes per timestep (default: 1 min)
    reb_tstep           minutes per timestep (default: 3 min)
    sumo_tstep          sumo time step (default: 1 s)
    max_waiting_time    maximum passengers waiting time for a ride (default: 10 min)
    beta                cost of rebalancing (default: 1)
    num_regions         number of regions for spatial aggregation (default: 8)
    random_od           demand aggregated in the centers of the regions (default: False)
    acc_init            initial number of taxis per region (default: 90)
    city                (default: 'lux')
    directory           defines directory where to save files
```
### Models parameters
You need to pass the following argument to set a simulator parameter:
```
model.{arg}=value
```
Use the following argument for a2c agent:
```
model=a2c
```
```
a2c model arguments:
    agent_name          agent name for training or evaluation (default: today's date + '_a2c_gnn')
    cplexpath           defines directory of the CPLEX installation
    directory           defines directory where to save files
    max_episodes        number of episodes to train agent (default: 16k)
    max_steps           number of steps per episode (default: T=120)
    no_cuda             disables CUDA training (default: true)
    batch_size          defines batch size (default: 100)
    p_lr                define policy learning rate (default: 1e-3)
    q_lr                defines q-value learning rate (default: 1e-3)
    hidden_size         defines hidden units in the MLP layers (default: 256)
    clip                clip value for gradient clipping (default: 500)
    checkpoint_path     path where to save model checkpoints (A2C)
```

Use the following argument for sac agent:
```
model=sac
```
```
sac model arguments:
    agent_name          agent name for training or evaluation (default: 'SAC'+'{city}')
    cplexpath           defines directory of the CPLEX installation
    max_episodes        number of episodes to train agent (default: 16k)
    no_cuda             disables CUDA training (default: true)
    batch_size          batch size (default: 100)
    p_lr                policy learning rate (default: 1e-3)
    q_lr                q-value learning rate (default: 1e-3)
    alpha               entropy coefficient (default: 0.3)
    auto_entropy        use automatic entropy tuning (default: false)
    hidden_size         hidden units in the MLP layers (default: 256)
    clip                clip value for gradient clipping (default: 500)
    checkpoint_path     path where to save model checkpoints (SAC)
    rew_scale           reward scale (default: 0.01)
    use_LSTM            use LSTM in the model (default: false)
    input_size          number of node features (defalut: 13)
    test_episodes       number of episodes to test agent (default 10)
```

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following. If model.cplexpath = None, the PulP solver will be automatically called. 
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```

## Examples

### Training and simulating an agent

1. To train an agent (with the default parameters) run the following:
```
python train.py  simulator.name=macro model=sac simulator.city=nyc_brooklyn model.checkpoint_path=SAC_custom
```

2. To evaluate a pretrained agent run the following:
```
python testing.py  simulator.name=macro model=sac simulator.city=nyc_brooklyn model.checkpoint_path=SAC_custom
```

### Testing baselines
2. To evaluate a baseline (e.g. equal_distribution) run the following (All available models can be found in *src/algos/registry.py*):
```
python testing.py  simulator.name=macro model.name=equal_distribution simulator.city=nyc_brooklyn
```

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact us at <gammelli@stanford.edu>, csasc@dtu.dk or <ltresca@stanford.edu>.