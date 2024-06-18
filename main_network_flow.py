import torch
from src.envs.network_flow_env import NetworkFlowEnv
import os
import random
import gymnasium as gym
from enum import Enum

from torch.utils.tensorboard import SummaryWriter
from gymnasium.envs.registration import register
from src.algos.a2c_stable_baselines import CustomMultiInputActorCriticPolicy
from src.algos.sac_stable_baselines import CustomSACPolicy
from src.envs.stable_baselines_env_wrapper import MyDummyVecEnv
from src.misc.utils import FeatureExtractor, RLAlgorithm
from src.algos.stable_baselines_mpnn import MPNNExtractor
from src.algos.stable_baselines_gcn import GCNExtractor
from src.algos.stable_baselines_mlp import MLPExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, SAC, PPO

# set this to the path to the saved checkpoint (e.g. network_flow_checkpoints/SAC/MPNN/100000_steps.zip) to resume
# from that checkpoint
CHECKPOINT_PATH = ""


device = "cpu"

random.seed(104)

class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, tensorboard_writer, eval_freq=1000, save_freq=10000, verbose=1,
                 rl_algorithm=RLAlgorithm.A2C, feature_extractor=FeatureExtractor.GCN):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tensorboard_writer = tensorboard_writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = f"./network_flow_checkpoints/{rl_algorithm.name}/{feature_extractor.name}/"
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 or self.num_timesteps == 1:
            validation_reward = self.evaluate_model()
            self.tensorboard_writer.add_scalar("Validation reward", validation_reward, self.num_timesteps)
            print("Validation reward is ", validation_reward)
        if self.num_timesteps % self.save_freq == 0:
            self.save_checkpoint()
        return True
    
    def save_checkpoint(self):
        model_path = os.path.join(self.save_path, f"{self.num_timesteps}_steps.zip")
        self.model.save(model_path)
        print(f"Saving model checkpoint to {model_path}")


    def evaluate_model(self):
        self.eval_env.env_method("set_start_to_end_test", True)
        obs = self.eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            episode_reward += reward
        self.eval_env.env_method("set_start_to_end_test", False)
        # uncomment this to visualize the path taken during training
        # self.eval_env.env_method("visualize_prediction", info[0]["true_shortest_path"], info[0]["path_followed"], info[0]["episode_reward"])
        return episode_reward

def run_training(feature_extractor, rl_algorithm, total_timesteps=20000):
    run_dir = os.path.join('network_flow_runs', f'{rl_algorithm.name}/{feature_extractor.name}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Register the environment
    register(id='CustomEnv-v0', entry_point=NetworkFlowEnv)

    # Create the environment
    env = MyDummyVecEnv([lambda: gym.make('CustomEnv-v0')])

    if feature_extractor == FeatureExtractor.GCN:
        features_extractor_class = GCNExtractor
    elif feature_extractor == FeatureExtractor.MPNN:
        features_extractor_class = MPNNExtractor
    else:
        features_extractor_class = MLPExtractor

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs={
            "hidden_features_dim": 11, # TODO: make this a cmdline argument
            "num_nodes": env.envs[0].nregion
        }
    )

    if rl_algorithm == RLAlgorithm.A2C:
        model = A2C(CustomMultiInputActorCriticPolicy,
                    env, policy_kwargs=policy_kwargs, verbose=1,
                    use_rms_prop=False, learning_rate=1e-4, ent_coef=0.01, device=device)
        eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = A2C.load(CHECKPOINT_PATH, env=env, device=device)
    elif rl_algorithm == RLAlgorithm.PPO:
        model = PPO(CustomMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs,
                    verbose=1, learning_rate=1e-4, ent_coef=0.01, device=device)
        eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = PPO.load(CHECKPOINT_PATH, env=env, device=device)
    else:
        model = SAC(CustomSACPolicy, env, policy_kwargs=policy_kwargs,
                    verbose=1, learning_rate=1e-4, ent_coef=0.01, device=device)
        eval_callback = EvaluationCallback(env, writer, eval_freq=100, save_freq=1000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = SAC.load(CHECKPOINT_PATH, env=env, device=device)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

for algorithm in [RLAlgorithm.SAC, RLAlgorithm.A2C, RLAlgorithm.PPO]:
    for extractor in [FeatureExtractor.MPNN, FeatureExtractor.GCN, FeatureExtractor.MLP]:
        if algorithm == RLAlgorithm.A2C or algorithm == RLAlgorithm.PPO:
            run_training(extractor, algorithm, 2000000)
        else:
            run_training(extractor, algorithm, 100000)

