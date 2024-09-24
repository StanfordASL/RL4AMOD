import math

import numpy as np
import pandas as pd
import math
#from lxml import etree as ET
import networkx as nx



def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  


def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])


def moving_average(data, window=5):
    """
    Computes a moving average used for reward trace smoothing.
    """
    data = pd.Series(data)
    mov_data = data.rolling(window=window).mean()
    return list(mov_data)


def moving_std(data, window=5):
    """
    Computes a moving standard deviation used for reward trace smoothing.
    """
    data = pd.Series(data)
    mov_data = data.rolling(window=window).std()
    return list(mov_data)

"""
def sumo_scenario_analysis(xml_file, acc_init, nregions, center_pos, time_start=0, duration=2):
    
    Function to analyze the tripinfo xml file, output of sumo simulation
    
    # Initialization
    tripinfos_tree = ET.parse(xml_file)
    time_start = time_start * 60 * 60
    duration = duration * 60 * 60
    regions_graph = nx.complete_graph(nregions, create_using=nx.DiGraph())
    regions_info = list()
    taxi_info = dict()
    taxi_info['taxi_utilization'] = np.zeros(duration)
    taxi_info['taxi_rebalancing'] = np.zeros(duration)
    taxi_info['taxi_idle'] = np.zeros(duration)
    for region in range(nregions):
        regions_graph.nodes[region]['acc'] = np.ones(duration) * acc_init
        regions_graph.nodes[region]['pos'] = center_pos[region]
        regions_graph.nodes[region]['demand'] = np.zeros(duration)
        regions_graph.nodes[region]['demand_tot'] = np.zeros(duration)
        regions_graph.nodes[region]['demand_inst'] = np.zeros(duration)
        regions_graph.nodes[region]['served_demand'] = np.zeros(duration)
        regions_graph.nodes[region]['removed_demand'] = np.zeros(duration)
        regions_graph.nodes[region]['incoming'] = np.zeros(duration)
        regions_graph.nodes[region]['rebalancing_in'] = np.zeros(duration)
        regions_graph.nodes[region]['matching_in'] = np.zeros(duration)
        regions_graph.nodes[region]['rebalancing_out'] = np.zeros(duration)
        regions_graph.nodes[region]['waiting_time'] = np.zeros(duration)
        for acc in range(acc_init):
            taxi_id = 'taxi' + str(region) + '#' + str(acc)
            taxi_info[taxi_id] = np.ones(duration) * region
    # Edges features initialization
    for edge in regions_graph.edges:
        regions_graph.edges[edge]['matching'] = 0
        regions_graph.edges[edge]['rebalancing'] = 0

    # xml output analysis
    tripinfos = tripinfos_tree.getroot()
    for tripinfo in tripinfos.findall('.//personinfo'):
        # Info extraction
        person_id = tripinfo.attrib.get('id')
        time_insert = int(float(tripinfo.attrib.get('depart'))) - time_start
        if time_insert >= duration:
            continue
        ride = tripinfo.find('ride')
        taxi_id = ride.attrib.get('vehicle')
        time_depart = int(float(ride.attrib.get('depart'))) - time_start
        time_arrival = int(float(ride.attrib.get('arrival'))) - time_start
        waiting_time = int(float(ride.attrib.get('waitingTime')))
        # Route info
        node_o = int(person_id[(person_id.find('o') + 1):person_id.rfind('d')])
        node_d = int(person_id[(person_id.find('d') + 1):person_id.rfind('#')])
        # Trip analysis
        if 'rebalancing' in person_id:
            # Rebalanced vehicles
            regions_graph.nodes[node_o]['acc'][time_depart:] -= 1
            regions_graph.nodes[node_o]['rebalancing_out'][time_depart:] += 1
            regions_graph.edges[(node_o, node_d)]['rebalancing'] += 1
            if time_arrival >= 0:
                # Arrived taxi
                regions_graph.nodes[node_d]['acc'][time_arrival:] += 1
                regions_graph.nodes[node_d]['rebalancing_in'][time_arrival:] += 1
                taxi_info[taxi_id][time_depart:time_arrival] = -2
                taxi_info['taxi_utilization'][time_depart:time_arrival] += 1
                taxi_info['taxi_rebalancing'][time_depart:time_arrival] += 1
            else:
                # Not arrived taxi
                taxi_info[taxi_id][time_depart:] = -2
                taxi_info['taxi_utilization'][time_depart:] += 1
                taxi_info['taxi_rebalancing'][time_depart:] += 1
        else:
            # Passengers
            regions_graph.nodes[node_o]['demand'][time_insert:] += 1
            regions_graph.nodes[node_o]['demand_tot'][time_insert:] += 1
            regions_graph.nodes[node_o]['demand_inst'][time_insert] += 1
            if taxi_id != 'NULL':
                # Matched vehicles
                regions_graph.nodes[node_o]['acc'][time_depart:] -= 1
                regions_graph.nodes[node_o]['demand'][time_depart:] -= 1
                regions_graph.nodes[node_o]['served_demand'][time_depart:] += 1
                regions_graph.nodes[node_o]['waiting_time'][time_insert:(time_insert+waiting_time)] += np.arange(0, waiting_time, 1)
                regions_graph.edges[(node_o, node_d)]['matching'] += 1
                if time_arrival >= 0:
                    # Arrived taxi
                    regions_graph.nodes[node_d]['acc'][time_arrival:] += 1
                    regions_graph.nodes[node_d]['incoming'][time_arrival:] += 1
                    regions_graph.nodes[node_d]['matching_in'][time_arrival:] += 1
                    taxi_info[taxi_id][time_depart:time_arrival] = -1
                    taxi_info['taxi_utilization'][time_depart:time_arrival] += 1
                else:
                    # Not arrived taxi
                    taxi_info[taxi_id][time_depart:] = -1
                    taxi_info['taxi_utilization'][time_depart:] += 1
            else:
                if time_insert + waiting_time <= duration-1:
                    regions_graph.nodes[node_o]['waiting_time'][time_insert:(time_insert+waiting_time)] += np.arange(0, waiting_time, 1)
                    regions_graph.nodes[node_o]['demand'][(time_insert+waiting_time):] -= 1
                    regions_graph.nodes[node_o]['removed_demand'][(time_insert + waiting_time):] += 1
                else:
                    regions_graph.nodes[node_o]['waiting_time'][time_insert:] += np.arange(0, (duration - time_insert), 1)

    # Update idle taxi
    taxi_info['taxi_idle'] = acc_init*nregions - taxi_info['taxi_utilization']
    taxi_info['taxi_utilization'] /= acc_init * nregions
    taxi_info['taxi_rebalancing'] /= acc_init * nregions
    taxi_info['taxi_idle'] /= acc_init * nregions

    for region in regions_graph.nodes:
        regions_graph.nodes[region]['waiting_time'] /= regions_graph.nodes[region]['demand']
        regions_graph.nodes[region]['waiting_time'] = np.nan_to_num(regions_graph.nodes[region]['waiting_time'], nan=0)
    return regions_graph, taxi_info
"""