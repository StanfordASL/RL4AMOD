import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import networkx as nx
sys.path.append(os.path.dirname(os.environ.get('SUMO_HOME')))
import sumolib
from collections import namedtuple
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from src.misc.utils import moving_average, moving_std


def plot_reward(log_train, agent):
    """
    Plot the reward trend, the reward mean value and mean standard deviation
    """
    if not os.path.exists(f"./images/scenario_lux/{agent}"):
        os.makedirs(f"./images/scenario_lux/{agent}")
    train_reward_mean = moving_average(log_train['train_reward'], window=10)
    train_reward_std = moving_std(log_train['train_reward'], window=10)
    episodes = list(range(len(log_train['train_reward'])))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, log_train['train_reward'], label='reward')
    plt.plot(episodes, train_reward_mean, label='mean')
    plt.xlabel('Episodes [-]')
    plt.ylabel('Reward [-]')
    plt.title('Training reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./images/scenario_lux/{agent}/reward_trend.png")

    # Reward mean and std
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, train_reward_mean, label='mean')
    plt.fill_between(episodes,
                     [mean + std for mean, std in zip(train_reward_mean, train_reward_std)],
                     [mean - std for mean, std in zip(train_reward_mean, train_reward_std)],
                     alpha=0.5,
                     label='std')
    plt.xlabel('Episodes [-]')
    plt.ylabel('Reward [-]')
    plt.title('Training reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./images/scenario_lux/{agent}/reward_stats.png")

    # Actor loss
    plt.figure(figsize=(10, 6))
    plt.plot(log_train['train_policy_losses'], label='mean')
    plt.xlabel('Steps [-]')
    plt.ylabel('Loss [-]')
    plt.title('Actor loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./images/scenario_lux/{agent}/actor_loss.png")

    # Critic loss
    plt.figure(figsize=(10, 6))
    plt.plot(log_train['train_value_losses'], label='mean')
    plt.xlabel('Steps [-]')
    plt.ylabel('Loss [-]')
    plt.title('Critic loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./images/scenario_lux/{agent}/critic_loss.png")


def scenario_analysis_plot(regions_graph, taxi_info, log, policy_name):
    """
    Function to plot all the statistics to analyze the scenario
    """
    nregion = len(regions_graph.nodes)
    # Vehicles per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['acc'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# vehicles')
    plt.title('Vehicles per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/veh_region.png")

    # Demand per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['demand'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# people')
    plt.title('People per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/people_region.png")

    # Total per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['demand_tot'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('Demand')
    plt.title('Total demand per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/demand_region.png")
 
    # Served demand per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['served_demand'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# people')
    plt.title('Served demand per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/demand_served_region.png")

    # Removed demand per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['removed_demand'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# people')
    plt.title('Removed demand per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/demand_removed_region.png")

    # Average waiting time per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['waiting_time'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('Time [s]')
    plt.title('Waiting time per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/waiting_time_region.png")
 
    # Rebalancing out per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['rebalancing_out'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# vehicle')
    plt.title('Outgoing rebalanced vehicles per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/reb_out_region.png")
 
    # Rebalancing in per region
    plt.figure(figsize=(10, 6))
    for idx in range(nregion):
        region = regions_graph.nodes[idx]
        plt.plot(region['rebalancing_in'], label='region' + str(idx))
    plt.xlabel('Time [s]')
    plt.ylabel('# vehicle')
    plt.title('Incoming rebalanced vehicles per region')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/reb_in_region.png")

    # Fleet utilization
    plt.figure(figsize=(10, 6))
    plt.plot(taxi_info['taxi_utilization'], label='utilization factor')
    plt.plot(taxi_info['taxi_rebalancing'], label='rebalancing factor')
    plt.plot(taxi_info['taxi_idle'], label='idle factor')
    plt.xlabel('Time [s]')
    plt.ylabel('Utilization factor [%]')
    plt.title('Fleet utilization')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/taxi_utilizaton.png")
 
    # Demand vs incoming vehicles histogram
    plt.figure(figsize=(10, 6))
    demand = []
    rebalancing_in = []
    tot_in = []
    for region in regions_graph.nodes:
        demand.append(regions_graph.nodes[region]['demand_tot'][-1])
        rebalancing_in.append(regions_graph.nodes[region]['rebalancing_in'][-1])
        tot_in.append(regions_graph.nodes[region]['matching_in'][-1] + regions_graph.nodes[region]['rebalancing_in'][-1])
    regions = [str(region) for region in np.arange(0, len(demand), 1).tolist()]
    bar_width = 0.35
    r1 = np.arange(len(regions))
    r2 = [x + bar_width for x in r1]
    plt.bar(r2, demand, color="#A2142F", width=bar_width, edgecolor='black', label='Demand')
    plt.bar(r1, tot_in, color="#0072BD", alpha=0.5, width=bar_width, edgecolor='black', label=r'Total in vehicles')
    plt.bar(r1, rebalancing_in, color="#0072BD",  hatch='//', width=bar_width, edgecolor='black', label=r'Total reb vehicles')
    plt.xlabel('Regions', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks([r + bar_width/2 for r in range(len(regions))], regions)
    plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    plt.savefig(f"./images/scenario_lux/{policy_name}/vehicles_vs_demand.png", dpi=500)
 
    # Profit
    plt.figure(figsize=(10, 6))
    plt.plot(log[-1]['test_revenue'], label='revenue')
    plt.plot(log[-1]['test_op_cost'], label='operating_cost')
    plt.plot(np.array(log[-1]['test_op_cost']) - np.array(log[-1]['test_reb_cost']), label='demand_cost')
    plt.plot(log[-1]['test_reb_cost'], label='rebalancing_cost')
    plt.xlabel('Time [m]')
    plt.ylabel('$')
    plt.title('Profit analysis')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./images/scenario_lux/{policy_name}/profit.png")

    # Histogram profit
    plt.figure(figsize=(10, 6))
    demand = []
    rebalancing_in = []
    tot_in = []
    label = ['reward', 'revenue', 'op cost', 'reb cost']
    data = [log[-1]['test_reward'][-1], log[-1]['test_revenue'][-1], log[-1]['test_op_cost'][-1], log[-1]['test_reb_cost'][-1]]
    bar_width = 0.35
    bars = plt.bar(label, data, color="#A2142F", width=bar_width, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 '%d' % int(height), ha='center', va='bottom')
    plt.ylabel('$')
    plt.title('Profit histogram')
    plt.savefig(f"./images/scenario_lux/{policy_name}/profit_histogram.png", dpi=500)
 

def graph_flow_plot(regions_graph, policy_name):
    """
    Function to plot the flow in the graph-format network
    """
    pos = {i: regions_graph.nodes[i]['pos'] for i in regions_graph.nodes}
    # Edge filter
    edges_reb = []
    edges_noreb = []
    colors_reb = []
    colors_noreb = []
    edges_matching = []
    edges_nomatching = []
    colors_matching = []
    colors_nomatching = []
    for u, v, data in regions_graph.edges()(data=True):
        if 'rebalancing' in data and data['rebalancing'] > 0:
            edges_reb.append((u, v))
            colors_reb.append(data['rebalancing'])
        else:
            edges_noreb.append((u, v))
            colors_noreb.append(data['rebalancing'])
        if 'matching' in data and data['matching'] > 0:
            edges_matching.append((u, v))
            colors_matching.append(data['matching'])
        else:
            edges_nomatching.append((u, v))
            colors_nomatching.append(data['matching'])
    # Rebalancing flow plot
    plt.figure(figsize=(10, 6))
    nx.draw(regions_graph, pos, with_labels=True,
            node_size=800, node_color="darkblue", font_size=12, font_color="white", edge_color="darkblue")
    nx.draw_networkx_edges(regions_graph, pos,
                           edgelist=edges_noreb, edge_color=colors_noreb,
                           edge_cmap=plt.cm.jet, width=1, arrowsize=10)
    nx.draw_networkx_edges(regions_graph, pos,
                           edgelist=edges_reb, edge_color=colors_reb,
                           edge_cmap=plt.cm.jet, width=2, arrowsize=25)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max(colors_reb)))
    sm.set_array([])
    plt.colorbar(sm, label="Rebalancing flow", ax=plt.gca())
    plt.suptitle(' Rebalancing flow')
    plt.savefig(f"./images/scenario_lux/{policy_name}/rebalancing_flow_graph.png")
 
    # Matching flow plot
    plt.figure(figsize=(10, 6))
    nx.draw(regions_graph, pos, with_labels=True,
            node_size=800, node_color="darkblue", font_size=12, font_color="white", edge_color="darkblue")
    nx.draw_networkx_edges(regions_graph, pos,
                           edgelist=edges_nomatching, edge_color=colors_nomatching,
                           edge_cmap=plt.cm.jet, width=1, arrowsize=10)
    nx.draw_networkx_edges(regions_graph, pos,
                           edgelist=edges_matching, edge_color=colors_matching,
                           edge_cmap=plt.cm.jet, width=2, arrowsize=25)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max(colors_matching)))
    sm.set_array([])
    plt.colorbar(sm, label="Matching flow", ax=plt.gca())
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.suptitle('Matching flow')
    plt.savefig(f"./images/scenario_lux/{policy_name}/matching_flow_graph.png")


def plot_net(net_file):
    """
    Function to plot the original network
    """
    net = sumolib.net.readNet(net_file)
    options = namedtuple
    options.defaultColor = '#000000'
    options.defaultWidth = .5
    shapes = []
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        c.append(options.defaultColor)
        w.append(options.defaultWidth)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    line_segments = LineCollection(shapes, linewidths=w, colors=c, linestyles='-')
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(True, True, True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./images/scenario_lux/net.png", dpi=500)


def plot_net_demand(net_file, scenario, regions_graph, policy_name):
    """
    Function to plot the original network
    """
    net = sumolib.net.readNet(net_file)
    options = namedtuple
    options.defaultColor = '#000000'
    options.defaultWidth = .5
    demand_tot = [regions_graph.nodes[node]['demand_tot'][-1] for node in regions_graph.nodes]
    norm = Normalize(vmin=min(demand_tot), vmax=max(demand_tot))
    cmap = cm.jet
    shapes = []
    colors = [cmap(norm(demand)) for demand in demand_tot]
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        region = scenario.cluster_alg.predict(np.array(e.getFromNode().getCoord()).reshape(1, -1))[0]
        c.append(colors[region])
        w.append(options.defaultWidth)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    line_segments = LineCollection(shapes, linewidths=w, colors=c, linestyles='-')
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(True, True, True)
    plt.xticks([])
    plt.yticks([])
    plt.title('Demand per region')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Demand')
    plt.savefig(f"./images/scenario_lux/{policy_name}/net_wdemand.png", dpi=500)


def plot_net_rebalancing(net_file, scenario, regions_graph, policy_name):
    """
    Function to plot the original network
    """
    net = sumolib.net.readNet(net_file)
    options = namedtuple
    options.defaultColor = '#000000'
    options.defaultWidth = .5
    demand_tot = [regions_graph.nodes[node]['rebalancing_in'][-1] for node in regions_graph.nodes]
    norm = Normalize(vmin=min(demand_tot), vmax=max(demand_tot))
    cmap = cm.jet
    shapes = []
    colors = [cmap(norm(demand)) for demand in demand_tot]
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        region = scenario.cluster_alg.predict(np.array(e.getFromNode().getCoord()).reshape(1, -1))[0]
        c.append(colors[region])
        w.append(options.defaultWidth)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    line_segments = LineCollection(shapes, linewidths=w, colors=c, linestyles='-')
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(True, True, True)
    plt.xticks([])
    plt.yticks([])
    plt.title('In rebalacing vehicles per region')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Vehicles')
    plt.savefig(f"./images/scenario_lux/{policy_name}/net_wreb.png", dpi=500)


def plot_net_matching(net_file, scenario, regions_graph, policy_name):
    """
    Function to plot the original network
    """
    net = sumolib.net.readNet(net_file)
    options = namedtuple
    options.defaultColor = '#000000'
    options.defaultWidth = .5
    demand_tot = [regions_graph.nodes[node]['matching_in'][-1] for node in regions_graph.nodes]
    norm = Normalize(vmin=min(demand_tot), vmax=max(demand_tot))
    cmap = cm.jet
    shapes = []
    colors = [cmap(norm(demand)) for demand in demand_tot]
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        region = scenario.cluster_alg.predict(np.array(e.getFromNode().getCoord()).reshape(1, -1))[0]
        c.append(colors[region])
        w.append(options.defaultWidth)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    line_segments = LineCollection(shapes, linewidths=w, colors=c, linestyles='-')
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(True, True, True)
    plt.xticks([])
    plt.yticks([])
    plt.title('In matching vehicles per region')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Vehicles')
    plt.savefig(f"./images/scenario_lux/{policy_name}/net_wmatching.png", dpi=500)


def plot_net_aggregated(net_file, scenario, policy_name):
    """
    Function to plot the aggregated network
    """
    # Net aggregated plot
    net = sumolib.net.readNet(net_file)
    options = namedtuple
    options.defaultColor = '#000000'
    options.defaultWidth = .5
    shapes = []
    colors = [
        "#FF5733",  # Arancione
        "#33FF57",  # Verde
        "#3357FF",  # Blu
        "#FF33A6",  # Rosa
        "#33FFF5",  # Ciano
        "#F5FF33",  # Giallo
        "#FF8C33",  # Arancione scuro
        "#8C33FF",  # Viola
        "#33FF8C",  # Verde menta
        "#FF3333",  # Rosso
        "#3333FF",  # Blu medio
        "#33FFA6",  # Verde acqua
        "#FF33F5",  # Magenta
        "#A6FF33",  # Verde lime
        "#F533FF",  # Fucsia
        "#33A6FF",  # Blu cielo
        "#FF3333",  # Rosso vivo
        "#33FF33",  # Verde chiaro
        "#FFA633",  # Arancione dorato
        "#FF33D4"   # Rosa intenso
    ]
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        region = scenario.cluster_alg.predict(np.array(e.getFromNode().getCoord()).reshape(1, -1))[0]
        c.append(colors[region])
        # c.append(options.defaultColor)
        w.append(options.defaultWidth)

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    line_segments = LineCollection(shapes, linewidths=w, colors=c, linestyles='-')
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(True, True, True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./images/scenario_lux/{policy_name}/net_aggregated.png", dpi=500)
