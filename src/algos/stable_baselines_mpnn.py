import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch.nn import Sequential as Seq, Linear, LeakyReLU
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class EdgeConv(MessagePassing):
    def __init__(self, node_size=4, edge_size=0, out_channels=4):
        super().__init__(aggr='min', flow="source_to_target")
        self.mlp = Seq(Linear(2 * node_size + edge_size, out_channels),
                       LeakyReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

class MPNNExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    """
    def __init__(self, observation_space, hidden_features_dim: int = 1, action_dim: int = 0,
                 num_nodes: int = 1):
        node_features_dim = observation_space["node_features"].shape[1]
        edge_features_dim = observation_space["edge_features"].shape[1]
        super(MPNNExtractor, self).__init__(observation_space, hidden_features_dim)
        self.conv1 = EdgeConv(node_features_dim + action_dim, edge_features_dim, hidden_features_dim)
        self.conv2 = EdgeConv(hidden_features_dim, 1, hidden_features_dim)
        self.conv3 = EdgeConv(hidden_features_dim, 1, hidden_features_dim)
        self.lin1 = nn.Linear(hidden_features_dim + node_features_dim + action_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
    
    def forward(self, observations) -> torch.Tensor:
        x = observations['node_features'].type(torch.IntTensor)
        num_nodes = x.shape[1]
        edge_index = observations['edge_index'].type(torch.LongTensor)
        edge_attr = observations['edge_features']
        data_list = [Data(x=x[i], edge_index=edge_index[i], edge_attr=edge_attr[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        x = F.leaky_relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = F.leaky_relu(self.conv2(x, batch.edge_index, batch.edge_attr))
        x = F.leaky_relu(self.conv3(x, batch.edge_index, batch.edge_attr))
        x = torch.cat([batch.x, x], dim=1)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = x.reshape(-1, num_nodes, x.shape[1])
        return x
