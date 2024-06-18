from typing import Tuple,  Dict, Optional
import torch
import torch.nn as nn

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from src.algos.dirichlet_distribution import DirichletDistribution

    
class CustomSACActor(Actor):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(CustomSACActor, self).__init__(
            *args, **kwargs
        )
        # override custom Gaussian distribution
        self.action_dist = DirichletDistribution(kwargs["action_space"].shape[1])
        self.last_linear_layer = nn.Linear(self.features_dim, 1)
    
    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Concentration
        """
        features = self.extract_features(obs, self.features_extractor)
        action_logits = self.last_linear_layer(features).squeeze(-1)
        return action_logits

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        action_logits = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(action_logits, deterministic=deterministic)

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(action_logits)

class CustomContinuousCritic(ContinuousCritic):
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(True):
            # include actions as input to feature extractor
            obs_copy = obs.copy()
            obs_copy["node_features"] = torch.cat([obs["node_features"], actions.unsqueeze(-1)], dim=2)
            features = torch.sum(self.extract_features(obs_copy, self.features_extractor), dim=1)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            # include actions as input to feature extractor
            obs_copy = obs.copy()
            obs_copy["node_features"] = torch.cat([obs["node_features"], actions.unsqueeze(-1)], dim=2)
            features = torch.sum(self.extract_features(obs_copy, self.features_extractor), dim=1)
        return self.q_networks[0](torch.cat([features, actions], dim=1), dim=1)

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomSACActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.critic = self.make_critic(features_extractor=self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs,
                                                                            action_dim=self.action_space.shape[1]))
        critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs,
                                                                              action_dim=self.action_space.shape[1]))
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)
    
    def _get_action_dist_from_latent(self, latent_pi):
        concentration = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(concentration)

