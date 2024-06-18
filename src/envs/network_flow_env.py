"""
Simple Network Flow Environment
-----------------------------------------
This file contains the specifications for the network flow simulator.
"""
from collections import defaultdict
import torch
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from matplotlib.lines import Line2D

# Function to generate a random connected graph that is not complete
def generate_connected_graph(num_nodes):
    G = nx.DiGraph()

    start_node = 0
    goal_node = 7
    other_nodes = [1, 2, 3, 4, 5, 6]

    # Add nodes to the graph
    G.add_nodes_from([start_node, goal_node] + other_nodes)

    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 4), (4, 1), (1, 5), (5, 1), (2, 4), (4, 2), (2, 5), (5, 2), (2, 6), (6, 2), (3, 5), (5, 3), (3, 6), (6, 3), (4, 7), (7, 4), (5, 7), (7, 5), (6, 7), (7, 6),
             (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7,7)]

    G.add_edges_from(edges)

    custom_pos = {
        0: (0, 1),
        1: (1, 0),
        2: (1, 1),
        3: (1, 2),
        4: (2, 0),
        5: (2, 1),
        6: (2, 2),
        7: (3, 1)
    }
    nx.draw(G, custom_pos, with_labels=True)
    return G

class NetworkFlowEnv(gym.Env):
    def __init__(
        self, num_nodes=10
    ):
        self.G = generate_connected_graph(num_nodes)
        self.region = list(self.G)  # set of nodes
        self.edges = list(self.G.edges)
        self.total_commodity = 1
        self.edge_index = self.get_edge_index()
        self.nregion = len(self.G)
        self.start_to_end_test = False
        
        self.time = 0  # current time
        self.acc = defaultdict(dict) # commodities at each node at each timestep
        self.max_steps = 10
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.nregion, 1), dtype=np.float32)
        self.observation_space_dict = {
            "node_features": gym.spaces.Box(low=0, high=self.total_commodity, shape=(self.nregion, 2), dtype=int),
            "edge_features": gym.spaces.Box(low=0, high=1, shape=(len(self.edges), 1), dtype=np.float32),
            "edge_index": gym.spaces.Box(low=0, high=self.nregion, shape=(2, len(self.edges)), dtype=int)
        }
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)


    def get_edge_index(self):
        edge_index = np.array(self.edges).T
        return edge_index

    def set_start_to_end_test(self, start_to_end_test):
        self.start_to_end_test = start_to_end_test

    def get_current_state(self):
        # current state (input to RL policy) includes current
        # commodity distribution (integer distribution that adds up to total commodity), 
        # where the goal node is,
        # travel times, 
        # and graph layout (self.edge_index) 
        node_data = torch.IntTensor([self.acc[i][self.time] for i in range(len(self.G.nodes))]).unsqueeze(1)
        node_data = torch.hstack([node_data, self.goal_node_feature[:, None]])
        return_val = {
            "node_features": node_data.numpy(),
            "edge_features": self.edge_data.numpy(),
            "edge_index": self.edge_index
        }
        return return_val
    
    def get_action_from_action_rl(self, action_rl):
        """
        Params:
            action_rl is action outputted by stable baselines RL policy
            It represents desired distribution of commodity
        Returns:
        action_dict mapping edge to action
        """
        highest_node_prob = 0
        selected_edge_index = -1
        cur_region = 0
        for n in self.region:
            if self.acc[n][self.time] == 1:
                cur_region = n
        action_dict = {}
        # Choose to move the commodity to the highest probability adjacent node
        # (can be same as current node because of self edges)
        for n, edge in enumerate(self.edges):
            (i,j) = edge
            # only consider adjacent nodes
            if i == cur_region:
                if action_rl[j] > highest_node_prob:
                    highest_node_prob = action_rl[j]
                    selected_edge_index = n
        action_dict[self.edges[selected_edge_index]] = 1
        return action_dict

    def step(self, action_rl):
        """
        Params:
            action_rl is action outputted by stable baselines RL policy
            It is desired distribution of commodity
        Returns:
        next state (matches self.observation_space)
        integer reward for that step
            reward is -travel_time,
            plus 1 if all commodity reaches goal
            or minus 1 if trajectory terminates early
        boolean indicating whether trajectory is over or not
            currently trajectory only ends when all commodity reaches goal
        boolean indicating whether trajectory was terminated early (happens after 10 steps)
        metadata
        """
        # select action based on action_rl
        action_rl = action_rl / np.sum(action_rl)
        action_dict = self.get_action_from_action_rl(action_rl)
        # copy commodity distribution from current time to next
        for n in self.region:
            self.acc[n][self.time + 1] = self.acc[n][self.time]
        # add flows to commodity distribution for next timestamp
        total_travel_time = 0
        cur_region = None
        for edge, flow in action_dict.items():
            (i, j) = edge
            if edge not in self.G.edges:
                continue
            # update the position of the commodities
            if flow > 0:
                self.acc[i][self.time + 1] -= flow
                self.acc[j][self.time + 1] += flow
                total_travel_time += self.G.edges[(i,j)]['time'] * flow
                cur_region = j
        # check that commodities were conserved (no node has negative commodity)
        for n in self.region:
            assert self.acc[n][self.time + 1] >= 0
        self.time += 1
        self.path_followed.append(cur_region)
        # return next state, reward, trajectory complete, trajectory terminated early, and trajectory info
        if self.acc[self.goal_node][self.time] == self.total_commodity:
            # return self.get_current_state(), -total_travel_time + 10, True
            self.episode_reward += (-total_travel_time + 1)
            return self.get_current_state(), -total_travel_time + 1, True, False, {"true_shortest_path": self.true_shortest_path,
                                                                                    "path_followed": self.path_followed,
                                                                                    "episode_reward": self.episode_reward}
        elif self.time == self.max_steps:
            self.episode_reward += (-total_travel_time - 1)
            return self.get_current_state(), -total_travel_time - 1, False, True,  {"true_shortest_path": self.true_shortest_path,
                                                                                    "path_followed": self.path_followed,
                                                                                    "episode_reward": self.episode_reward}
        else:
            self.episode_reward -= total_travel_time
            return self.get_current_state(), -total_travel_time, False, False, {}
            
    def reset(self, seed=None):
        # resets environment for next trajectory, randomly chooses
        # start and goal node and travel times
        # all commodity starts at start node
        super().reset(seed=seed)
        self.time = 0  # current time
        self.acc = defaultdict(dict) # maps nodes to time to amount of commodity at that node at that time

        for i in self.G.edges:
            (a, b) = i
            # self edges will always have travel time 1
            if a == b:
                self.G.edges[i]['originalTime'] = 1
            # all other edges have random travel time between 1 and 5
            else:
                self.G.edges[i]['originalTime'] = random.randint(1,5)
        if self.start_to_end_test:
            self.start_node, self.goal_node = 0, self.nregion - 1
        else:
            shortest_path_length = -1
            # make sure we never choose start and goal node next to each other
            while shortest_path_length < 3:
                self.start_node, self.goal_node = np.random.choice(self.region, 2, replace=False)
                shortest_path = nx.shortest_path(self.G, source=self.start_node, target=self.goal_node, weight='originalTime')
                shortest_path_length = len(shortest_path)
        self.goal_node_feature = torch.IntTensor([1 if i == self.goal_node else 0 for i in range(self.nregion)])
        for n in self.region:
            self.acc[n][0] = self.total_commodity if n == self.start_node else 0
        self.true_shortest_path = nx.shortest_path(self.G, source=self.start_node, target=self.goal_node, weight='originalTime')
        self.path_followed = [self.start_node]
        self.episode_reward = 0
        # normalize travel times by travel time of shortest path
        shortest_path_travel_time = 0
        for n in range(len(self.true_shortest_path) - 1):
            (a, b) = self.true_shortest_path[n], self.true_shortest_path[n + 1]
            shortest_path_travel_time += self.G.edges[(a,b)]['originalTime']

        for i in self.G.edges:
            self.G.edges[i]['time'] = self.G.edges[i]['originalTime'] / float(shortest_path_travel_time)
            (a, b) = i
        for n in range(len(self.true_shortest_path) - 1):
            (a, b) = self.true_shortest_path[n], self.true_shortest_path[n + 1]
        self.edge_data = torch.FloatTensor([self.G.edges[i,j]['time'] for i,j in self.edges]).unsqueeze(1)
        return self.get_current_state(), {}
    
    def visualize_prediction(self, true_shortest_path, path_followed, episode_reward):
        custom_pos = {
            0: (0, 1),
            1: (1, 0),
            2: (1, 1),
            3: (1, 2),
            4: (2, 0),
            5: (2, 1),
            6: (2, 2),
            7: (3, 1)
        }
        plt.clf()
        nx.draw(self.G, custom_pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, font_weight='bold')

        # Highlight the true shortest path
        nx.draw_networkx_edges(self.G, custom_pos, edgelist=list(zip(true_shortest_path[:-1], true_shortest_path[1:])), edge_color='green', width=3)

        # Highlight the calculated shortest path
        nx.draw_networkx_edges(self.G, custom_pos, edgelist=list(zip(path_followed[:-1], path_followed[1:])), edge_color='red', width=1)


        legend_handles = [
            Line2D([0], [0], color='red', lw=2),
            Line2D([0], [0], color='green', lw=2)
        ]
        # Add a legend
        plt.legend(legend_handles, ['Predicted Shortest Path', 'True Shortest Path'])
        plt.text(2.2, 0, f'Difference in reward: {round(episode_reward, 2)}', ha='left', va='top')

        plt.pause(1)

