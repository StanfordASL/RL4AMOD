import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GCNExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    """
    def __init__(self, observation_space, hidden_features_dim: int = 1, action_dim: int = 0,
                 num_nodes: int = 1):
        node_features_dim = observation_space["node_features"].shape[1]
        super(GCNExtractor, self).__init__(observation_space, hidden_features_dim)
        self.conv1 = GCNConv(node_features_dim+action_dim, node_features_dim+action_dim)
        self.lin1 = nn.Linear(node_features_dim+action_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
    
    def forward(self, observations) -> torch.Tensor:
        x = observations['node_features'].type(torch.FloatTensor)
        num_nodes = x.shape[1]
        edge_index = observations['edge_index'].type(torch.LongTensor)
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = x + batch.x
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = x.reshape(-1, num_nodes, x.shape[1])
        return x
