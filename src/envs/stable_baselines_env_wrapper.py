
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from copy import deepcopy


class MyDummyVecEnv(DummyVecEnv):
    # modified implementation of stable baselines 3 DummyVecEnv
    # where terminal observation is saved
    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            self._save_obs(env_idx, obs)
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
