import json
import random
import os
import sys
import itertools
import math

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib
import numpy as np
import argparse
import os

from lxml import etree as ET
from sklearn.cluster import KMeans
from collections import defaultdict


class Scenario:
    """
    Class with the net and the clustering algorithm
    """

    def __init__(self, net, nregions, time_aggr=15):
        self.net = net
        self.nregions = nregions
        self.regions, self.kmeans = self.sumo_net_clustering()
        self.routes = self.get_routes()
        self.flows = list(itertools.product(range(nregions), range(nregions)))
        self.time_aggr = time_aggr
        self.data = {'regions': nregions, 'demand': list(), 'stat': defaultdict(dict), 'totalAcc': [0] * 24,
                     'totalDemand': [0] * int(24*60/self.time_aggr)}                # Time-aggregation of 15 mins
        for t in range(int(24*60/self.time_aggr)):
            self.data['demand'].append(defaultdict(dict))
        for (o, d) in self.flows:
            if o != d:
                self.data['stat'][(o, d)] = {
                    'travel_time': self.routes[(o, d)][2] / 60,
                    'price': 0,
                    'demand_tot': 0
                }
            else:
                self.data['stat'][(o, d)] = {
                    'travel_time': 0,
                    'price': 0,
                    'demand_tot': 0
                }
            for t in range(int(24*60/self.time_aggr)):              # Time-aggregation of 15 mins
                if o != d:
                    self.data['demand'][t][(o, d)] = {
                            'time_stamp': t,
                            'origin': o,
                            'destination': d,
                            'demand': 0
                        }
                else:
                    self.data['demand'][t][(o, d)] = {
                        'time_stamp': t,
                        'origin': o,
                        'destination': d,
                        'demand': 0
                    }
        self.root_traffic, self.root_demand, self.root_trips = xml_init()

    def sumo_net_clustering(self):
        """
        Function to cluster the junctions of a SUMO net, once a .sumocgf file has been started
        """
        # Nodes id and position
        nodes = self.net.getNodes()
        nodes_id = list()
        nodes_pos = list()
        for node in nodes:
            nodes_id.append(node.getID())
            nodes_pos.append(node.getCoord())

        # Clustering
        kmeans = KMeans(n_clusters=self.nregions, random_state=0, n_init=10)
        kmeans.fit(np.array(nodes_pos))
        labels = kmeans.labels_
        regions_sumo = list()
        for region in range(self.nregions):
            nodes_cluster = []
            nodes_id_cluster = []
            nodes_pos_cluster = []
            cluster_center = kmeans.cluster_centers_[region]
            for idx in range(len(nodes_pos)):
                if labels[idx] == region:
                    nodes_cluster.append(nodes[idx])
                    nodes_id_cluster.append(nodes_id[idx])
                    nodes_pos_cluster.append(nodes_pos[idx])

            distance = np.linalg.norm(np.array(nodes_pos_cluster) - cluster_center, axis=1)
            # Terminal edges correction and urban center selection
            is_urban = False
            discarded_node = list()
            while not is_urban:
                is_urban = True
                min_idx = np.argmin(distance)
                node = nodes_cluster[min_idx]
                in_edges = node.getIncoming()
                out_edges = node.getOutgoing()
                if node.getType() == 'dead_end':
                    distance[min_idx] = math.inf
                    is_urban = False
                    discarded_node.append(node.getID())
                    continue
                # Remove terminal edges
                in_edges = [in_edge for in_edge in in_edges if in_edge.getFromNode().getID() not in discarded_node and in_edge.getFromNode().getType() != 'dead_end']
                out_edges = [out_edge for out_edge in out_edges if out_edge.getToNode().getID() not in discarded_node and out_edge.getToNode().getType() != 'dead_end']
                # Discarded node close to terminal edges
                if len(in_edges) <= 1 or len(out_edges) <= 1:
                    distance[min_idx] = math.inf
                    discarded_node.append(node.getID())
                    is_urban = False
                    continue
                # Discard motorway junctions
                for in_edge in in_edges:
                    if 'motorway' in in_edge.getType():
                        distance[min_idx] = math.inf
                        discarded_node.append(node.getID())
                        is_urban = False

                for out_edge in out_edges:
                    if 'motorway' in out_edge.getType():
                        distance[min_idx] = math.inf
                        discarded_node.append(node.getID())
                        is_urban = False

            regions_sumo.append({'id': nodes_id_cluster, 'position': np.array(nodes_pos_cluster),
                                 'id_center': node.getID(), 'position_center': nodes_pos_cluster[min_idx],
                                 'in_edges': in_edges,
                                 'out_edges': out_edges}
                                )

        # Cluster centers edges
        return regions_sumo, kmeans

    def get_routes(self):
        """
        Method to compute the shortest route given the center of an aggregated net and save a .xml with the trip
        """
        # Cluster centers edges
        taxi_routes = defaultdict(tuple)
        routes = list(itertools.combinations(list(range(len(self.regions))), 2))
        for o, d in routes:
            taxi_routes[(o, d)] = (None, None, float('inf'))
            for edge_o in self.regions[o]['out_edges']:
                for edge_d in self.regions[d]['in_edges']:
                    route = self.net.getOptimalPath(edge_o, edge_d, fastest=True)
                    if route[1] < taxi_routes[(o, d)][2]:
                        length = sumolib.route.getLength(self.net, route[0])
                        spd_avg = length / route[1] * 3.6  # [km/h]
                        taxi_routes[(o, d)] = ((route[0][0].getID(), route[0][-1].getID()), route[0], route[1], length, spd_avg)
            # Add the way back
            edge_o = taxi_routes[(o, d)][0][1]
            edge_d = taxi_routes[(o, d)][0][0]
            # Edge correction for the backward road
            if edge_o[1].isdigit():
                edge_o = edge_o[1:]
            else:
                edge_o = edge_o[2:]
            edge_found = False
            for edge in self.regions[d]['out_edges']:
                if edge_o in edge.getID():
                    edge_o = edge.getID()
                    edge_found = True
                    break
            if not edge_found:
                edge_o = self.regions[d]['out_edges'][0].getID()

            if edge_d[1].isdigit():
                edge_d = edge_d[1:]
            else:
                edge_d = edge_d[2:]
            edge_found = False
            for edge in self.regions[o]['in_edges']:
                if edge_d in edge.getID():
                    edge_d = edge.getID()
                    edge_found = True
                    break
            if not edge_found:
                edge_d = self.regions[o]['in_edges'][0].getID()

            ids = [edge.getID() for edge in self.regions[d]['out_edges']]
            edge_o = self.regions[d]['out_edges'][ids.index(edge_o)]
            ids = [edge.getID() for edge in self.regions[o]['in_edges']]
            edge_d = self.regions[o]['in_edges'][ids.index(edge_d)]
            route = self.net.getOptimalPath(edge_o, edge_d, fastest=True)
            try:
                length = sumolib.route.getLength(self.net, route[0])
            except:
                ciao = 1
            spd_avg = length / route[1] * 3.6  # [km/h]
            try:
                taxi_routes[(d, o)] = ((route[0][0].getID(), route[0][-1].getID()), route[0], route[1], length, spd_avg)
            except Exception as e:
                raise RuntimeError(f"Route from {d} to {o} not found: please reduce the number of regions")

        taxi_routes = defaultdict(tuple, sorted(taxi_routes.items(), key=lambda x: x[0]))
        return taxi_routes
    

    def is_demand(self, prob=0.15):
        """
        Function to set a route as a travel demand
        """
        return True if random.random() < prob else False

    def set_root(self, xml_file='data/LuSTScenario/input/routes/local.0.rou.xml', demand='True'):
        """
        Function to split the initial xml file in demand and traffic xml files
        """
        xml_original = ET.parse(xml_file)
        root = xml_original.getroot()
        is_demand = True
        person_num = 0
        if demand == 'False':
            is_demand = False
        for vehicle in root.findall('vehicle'):
            depart = vehicle.attrib['depart']
            if demand == 'Random':
                is_demand = self.is_demand()
            if is_demand:
                time = int(float(depart)) // (60*self.time_aggr)
                edges = vehicle.find('route').attrib['edges']
                edges = edges.split(" ")
                edge_o = self.net.getEdge(edges[0])
                edge_d = self.net.getEdge(edges[-1])
                o_coord = edge_o.getFromNode().getCoord()
                d_coord = edge_d.getToNode().getCoord()
                o = self.kmeans.predict(np.array(o_coord).reshape(1, -1))[0]
                d = self.kmeans.predict(np.array(d_coord).reshape(1, -1))[0]
                if o != d:
                    if time <= 5:
                        price = 2.75 + 2.86 * self.routes[(o, d)][3] / 1000     # Night tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
                    else:
                        price = 2.5 + 2.60 * self.routes[(o, d)][3] / 1000  # Daily tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
                    self.data['demand'][time][(o, d)]['demand'] += 1
                    self.data['stat'][(o, d)]['demand_tot'] += 1
                    self.data['stat'][(o, d)]['price'] = price

                    # Create the passengers xml file
                    person_id = 'p' + str(time) + 'o' + str(o) + 'd' + str(d) + '#' + str(person_num)
                    person = ET.SubElement(self.root_trips, 'person')
                    person.set('id', person_id)
                    person.set('depart', str(depart))
                    ride = ET.SubElement(person, 'ride')
                    ride.set('from', edge_o.getID())
                    ride.set('to', edge_d.getID())
                    ride.set('lines', 'taxi')
                    person_num += 1

                else:
                    length = sumolib.route.getLength(self.net, edges)
                    if time <= 5:
                        price = 2.75 + 2.86 * length / 1000     # Night tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
                    else:
                        price = 2.5 + 2.60 * length / 1000  # Daily tariff (https://www.bettertaxi.com/taxi-fare-calculator/luxemburg/)
                    self.data['demand'][time][(o, d)]['demand'] += 1
                    self.data['stat'][(o, d)]['demand_tot'] += 1
                    self.data['stat'][(o, d)]['travel_time'] += (length / (50/3.6*60))
                    self.data['stat'][(o, d)]['price'] += price

                self.root_demand.append(vehicle)
                """
                # Create the passengers xml file
                person_id = 'p' + str(time) + 'o' + str(o) + 'd' + str(d) + '#' + str(person_num)
                person = ET.SubElement(self.root_trips, 'person')
                person.set('id', person_id)
                person.set('depart', str(depart))
                ride = ET.SubElement(person, 'ride')
                ride.set('from', edge_o.getID())
                ride.set('to', edge_d.getID())
                ride.set('lines', 'taxi')
                person_num += 1
                """
                self.data['totalDemand'][int(float(depart)) // (60 * self.time_aggr)] += 1
            else:
                self.root_traffic.append(vehicle)
            self.data['totalAcc'][int(float(depart)) // (60*60)] += 1

    def set_json(self):
        """
        Json file generation given self.data
        """
        data = {'regions': self.nregions, 'demand': list(), 'totalAcc': list()}
        for (o, d) in self.flows:
            for t in range(int(24*60//self.time_aggr)):
                flow = self.data['demand'][t][(o, d)]
                demand_tot = self.data['totalDemand'][t]
                if flow['demand'] == 0:
                    flow['demand'] = 0.0001
                for tt in range(self.time_aggr):
                    time_stamp = t * self.time_aggr + tt
                    data['demand'].append({
                            'time_stamp': int(time_stamp),
                            'origin': int(o),
                            'destination': int(d),
                            'demand': float(flow['demand'] / self.time_aggr),
                            'travel_time': float(self.data['stat'][(o, d)]['travel_time']),
                            'price': float(self.data['stat'][(o, d)]['price'])
                        })

        for hour in range(len(self.data['totalAcc'])):
            data['totalAcc'].append({
                'hour': int(hour),
                'acc': float(self.data['totalAcc'][hour])
            })
        return data


def xml_init():
    """
    Cml files import and initalization
    """
    root_traffic = ET.Element('routes')
    root_demand = ET.Element('routes')
    root_trips = ET.Element('routes')
    return root_traffic, root_demand, root_trips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='json generation')

    # Simulator parameters
    parser.add_argument('--num_regions', type=int, default=8, metavar='S',
                        help='Number of regions to aggregate the demand (default: 8)')
    parser.add_argument('--time_aggregation', type=int, default=15, metavar='S',
                        help='NTime interval for network information aggregation (default: 15 min)')

    args = parser.parse_args()
    scenario = Scenario(
        net=sumolib.net.readNet('data/LuSTScenario/input/lust_meso.net.xml'),
        nregions=args.num_regions, time_aggr=args.time_aggregation
    )

    xml_exist = os.path.exists('data/LuSTScenario/input/routes/local.rou.xml')
    print(f'demand.rou.xml and local/rou.xml e xist: {xml_exist}')
    if xml_exist:
        xml_files = [
            'data/LuSTScenario/input/routes/local.rou.xml',
            'data/LuSTScenario/input/routes/demand.rou.xml'
        ]
    else:
        xml_files = [
            'data/LuSTScenario/input/routes/local.0.rou.xml',
            'data/LuSTScenario/input/routes/local.1.rou.xml',
            'data/LuSTScenario/input/routes/local.2.rou.xml',
        ]
    for xml in xml_files:
        if xml == 'data/LuSTScenario/input/routes/local.rou.xml':
            scenario.set_root(xml, 'False')
        elif xml == 'data/LuSTScenario/input/routes/demand.rou.xml':
            scenario.set_root(xml, 'True')
        else:
            scenario.set_root(xml, 'Random')

    # Update o==d pairs price and travel time
    for (o, d) in scenario.flows:
        if o != d:
            continue
        price = scenario.data['stat'][(o, d)]['price']
        travel_time = scenario.data['stat'][(o, d)]['travel_time']
        demand = scenario.data['stat'][(o, d)]['demand_tot']
        scenario.data['stat'][(o, d)]['travel_time'] = travel_time / demand
        scenario.data['stat'][(o, d)]['price'] = price / demand

    if not xml_exist:
        tree_demand = ET.ElementTree(scenario.root_demand)
        tree_demand.write(f'data/LuSTScenario/input/routes/demand.rou.xml', pretty_print=True)
        tree_trips = ET.ElementTree(scenario.root_trips)
        tree_trips.write(f'data/LuSTScenario/input/routes/trips.rou.xml', pretty_print=True)
        tree_traffic = ET.ElementTree(scenario.root_traffic)
        tree_traffic.write(f'data/LuSTScenario/input/routes/local.rou.xml', pretty_print=True)
    data = scenario.set_json()
    with open(f'data/scenario_lux{args.num_regions}_taggr{args.time_aggregation}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
