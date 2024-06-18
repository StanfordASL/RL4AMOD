from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, LeakyReLU
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    """
    def __init__(self, observation_space, hidden_features_dim: int = 1, action_dim: int = 0,
                 num_nodes: int = 1):
        node_features_dim = observation_space["node_features"].shape[1]
        super(MLPExtractor, self).__init__(observation_space, hidden_features_dim)
        self.lin1 = nn.Linear(num_nodes * (node_features_dim+action_dim), (num_nodes * hidden_features_dim))
        self.lin2 = nn.Linear(num_nodes * hidden_features_dim, num_nodes * hidden_features_dim)
        self.lin3 = nn.Linear(num_nodes * hidden_features_dim, num_nodes * hidden_features_dim)
        self.num_nodes = num_nodes
    
    def forward(self, observations) -> torch.Tensor:
        x = observations['node_features'].type(torch.FloatTensor)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  
        x = F.leaky_relu(self.lin1(x)) 
        x = F.leaky_relu(self.lin2(x)) 
        x = F.leaky_relu(self.lin3(x))  
        x = x.reshape(x.shape[0], self.num_nodes, -1)            
        return x
