
from typing import List, Tuple, TypeVar
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import Distribution, sum_independent_dims

SelfDirichletDistribution = TypeVar("SelfDirichletDistribution", bound="DirichletDistribution")


from typing import List, Tuple, TypeVar
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import Distribution

SelfDirichletDistribution = TypeVar("SelfDirichletDistribution", bound="DirichletDistribution")

class DirichletDistribution(Distribution):
    """
    Dirichlet distribution
    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
       """
        Create the layer that represents the distribution:
        You can then get probabilities using a softplus.
        :return:
        """
       action_logits = nn.Identity()
       return action_logits

    def proba_distribution(self: SelfDirichletDistribution, action_logits: torch.Tensor) -> SelfDirichletDistribution:
        concentration = F.softplus(action_logits)
        concentration += torch.rand(concentration.shape) * 1e-20
        self.concentration = concentration
        # print("concentration shape ", concentration.shape)
        self.distribution = Dirichlet(concentration)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        # potentially try out rsample for stability
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.concentration / (self.concentration.sum(dim=1)[:, None])

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
