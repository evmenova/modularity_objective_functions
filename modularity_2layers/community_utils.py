import warnings

warnings.filterwarnings('ignore')
from collections import Counter, OrderedDict
import community
import json
import networkx as nx
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode, plot

import collections
import kaleido
import os
import sklearn
from louvain_modified import louvain_modified_algorithm as LM
import leidenalg

init_notebook_mode(connected=True)

from itertools import combinations, product
from numpy.linalg import norm
from scipy.spatial.distance import cosine, minkowski, jaccard, hamming
from tqdm import tqdm_notebook, tqdm
import random
from functools import partial
from multiprocessing import cpu_count
import multiprocessing as mp

import igraph as ig
import random

def from_nx_to_ig_graph(nx_graph, node_list):
    g = ig.Graph()
    g.add_vertices(len(node_list))
    g.vs['name'] = list(node_list)
    g.add_edges(list(nx_graph.edges()))
    g.es['weight'] = list(nx.get_edge_attributes(nx_graph, 'weight').values())
#     g.vs['label'] = list(nx_graph.nodes())
    return g

def from_ig_to_nx_partition_multiplex(partition_ig, id_name_dict):
    partition_nx = {id_name_dict[node]: com for node, com in enumerate(partition_ig)}
#     partition_nx = partition_ig
    return partition_nx


def from_ig_to_nx_partition_ordinary(partition_ig, id_name_dict):
    com_number = 0
    nodes = []
    coms = []
    for item in partition_ig:
        for node in item:
            nodes.append(node)
            coms.append(com_number)
        com_number += 1
    partition_nx = {id_name_dict[node]: com for node, com in zip(nodes, coms)}
    return partition_nx


def read_edge_list_layer(file_name):
    edge_list = []
    file = open(file_name)
    for line in file:
        spl_line = line.split()
        if spl_line:
            main_node = spl_line[0]
            neigh_list = spl_line[2:]
            triples = list(product([main_node], neigh_list, [1]))
            edge_list.extend(triples)
    return edge_list


def normalize_to_fixed_weight_ig(graph, total_weight):
    new_graph = graph.copy()
    graph_weights = graph.es['weight']
    sum_weight = np.sum(graph_weights)
    weight_values = [weight / sum_weight * total_weight for weight in graph_weights]
    new_graph.es['weight'] = weight_values
    return new_graph


def normalize_to_fixed_weight_nx(graph, total_weight):
    whole_weight = 0
    for item in graph.edges(data=True):
        whole_weight += item[2]['weight']
    
    for edge in graph.edges(data=True):
        graph[edge[0]][edge[1]]['weight'] = edge[2]['weight'] / whole_weight * total_weight
    
    print('The whole weight has been equalt to', whole_weight, '. Now it is equal to', check_the_whole_weight(graph))
    return


def read_raw_files(ef_name, sf_name, lf_name):
    partition = {'ef': {}, 'sf': {}, 'lf': {}}
    raw_file = {key: open(file,'r') for key,file in zip(partition.keys(), [ef_name, sf_name, lf_name])}
    for key in partition.keys():
        for line in raw_file[key]:
            partition[key] = eval(line)
    for f in raw_file.values():
        f.close()
    return partition


def read_raw_file(file_name):
    raw_file = open(file_name,'r')
    for line in raw_file:
        partition = eval(line)
    raw_file.close()
    return partition


def check_the_whole_weight(graph):
    check = 0
    for edge in graph.edges(data=True):
        check += edge[2]['weight']
    return check


def normalize(graph):
    whole_weight = 0
    for item in graph.edges(data=True):
        whole_weight += item[2]['weight']
    
    for edge in graph.edges(data=True):
        graph[edge[0]][edge[1]]['weight'] = edge[2]['weight'] / whole_weight
    
    print('The whole weight has been equalt to', whole_weight, '. Now it is equal to', check_the_whole_weight(graph))
    return


def save(path, what):
    with open(path, 'w') as f:
        print(what, file=f)


def late_fusion_transient_absorption(G_old, best_partit):
    G_late = G_old.copy()
    comm_absor = {}  # to collect all the weights that have been absorbed for each community
    
    #Remove edges between communities and save the removed data to comm_absor. The half of an edge for each community
    for edge in list(G_late.edges(data=True)):
        if best_partit[edge[0]] != best_partit[edge[1]]:
            for absorbed_edge in [edge[0], edge[1]]:
                if best_partit[absorbed_edge] in comm_absor.keys():
                    comm_absor[best_partit[absorbed_edge]] += edge[2]['weight'] / 2
                else:
                    comm_absor[best_partit[absorbed_edge]] = edge[2]['weight'] / 2
            G_late.remove_edge(edge[0], edge[1])
            
    #Construct the node list of each community
    node_list_comm = {com: [] for com in np.unique(list(best_partit.values()))}
    for node,com in best_partit.items():
        node_list_comm[com].append(node)

    #Calculate the weight to be distributed within community per each node
    for com,val in comm_absor.items():
        nodes_number = len(node_list_comm[com])
        if nodes_number != 1:
            weight_for_comm = val / (nodes_number * (nodes_number - 1))
            for node1 in node_list_comm[com]:  #Distribute the absorbed edges
                for node2 in node_list_comm[com]:
                    if node1 != node2:
                        if G_late.has_edge(node1, node2):
                            G_late[node1][node2]['weight'] = G_late.get_edge_data(node1, node2)['weight'] + weight_for_comm
                        else:
                            G_late.add_edge(node1, node2, weight = weight_for_comm)
    normalize_to_fixed_weight_nx(G_late, 1)
    return G_late


def combine(graph_layers, alphas):
    '''Make a graph out of graph layers using balance parameters alpha'''
    first_element = next(iter(graph_layers.values()))
    node_list = first_element.nodes()
    adj_matrix_new = np.zeros(shape=(len(node_list), len(node_list)))
    for layer_name, graph in graph_layers.items():
        temp_adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list, weight='weight').todense()
        adj_matrix_new += alphas[layer_name] * temp_adj_matrix
    new_graph = nx.from_numpy_matrix(adj_matrix_new)
    nx.relabel.relabel_nodes(new_graph, {val: name for val, name in enumerate(node_list)}, False)
    return new_graph


def SF(str_graph, att_graph, runs, alpha_grid):
    partition = {alpha: [] for alpha in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        for simulation in tqdm_notebook(range(runs), leave=False):
            partition[key_alpha].append(LM.best_partition(str_graph, att_graph, alpha=key_alpha[0]))
    return partition


def EF(str_graph, att_graph, runs, alpha_grid):
    partition = {alpha: [] for alpha in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(['str', 'att'], key_alpha)}
        ef_graph = combine({'str': str_graph, 'att': att_graph}, alphas_dict)
        for simulation in tqdm_notebook(range(runs), leave=False):
            partition[key_alpha].append(community.best_partition(ef_graph))
    return partition


def LF(str_graph, att_graph, runs, alphas_grid):
    best_partit = {'str': community.best_partition(str_graph),
                   'att': community.best_partition(att_graph)}
    late_graph = {'str': late_fusion_transient_absorption(str_graph, best_partit['str']),
                  'att': late_fusion_transient_absorption(att_graph, best_partit['att'])}
    return EF(late_graph['str'], late_graph['att'], runs, alphas_grid), best_partit['str'], best_partit['att']


def SF_leiden(graph_layers_ig, runs, alpha_grid, id_name_dict):
    partition = {alphas: [] for alphas in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
#         graph_layers_merging = {name: normalize_to_fixed_weight_ig(graph_layers_ig[name], total_weight) 
#                                 for name, total_weight in zip(graph_layers_ig.keys(), key_alpha)}
        for simulation in range(runs):
            membership, improvement = leidenalg.find_partition_multiplex(list(graph_layers_ig.values()), 
                                                                         leidenalg.ModularityVertexPartition,
                                                                         n_iterations=1, seed=random.randint(0, 1000))
            nx_partit = from_ig_to_nx_partition_multiplex(membership, id_name_dict)
            partition[key_alpha].append(nx_partit)
    return partition


def EF_leiden(graph_layers_nx, runs, alpha_grid, id_name_dict, node_list):
    partition = {alpha: [] for alpha in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers_nx.keys(), key_alpha)}
        ef_graph_nx = combine_nx(graph_layers_nx, alphas_dict, node_list)
        ef_graph_ig = from_nx_to_ig_graph(ef_graph_nx, node_list)
        for simulation in range(runs):
            membership = leidenalg.find_partition(ef_graph_ig, leidenalg.ModularityVertexPartition, n_iterations=1, 
                                                  seed=random.randint(0, 1000))
            nx_partit = from_ig_to_nx_partition_ordinary(membership, id_name_dict)
            partition[key_alpha].append(nx_partit)
    return partition


def LF_leiden(graph_layers_nx, runs, alphas_grid, id_name_dict, node_list):
    names = graph_layers_nx.keys()
    graph_layers_ig = {name: from_nx_to_ig_graph(graph_layers_nx[name], node_list) for name in names}
    best_partit_ig = {name: leidenalg.find_partition(graph_layers_ig[name], 
                                                     leidenalg.ModularityVertexPartition, 
                                                     n_iterations=1,
                                                     seed=random.randint(0, 1000)) 
                      for name in names}
    best_partit_nx = {name: from_ig_to_nx_partition_ordinary(best_partit_ig[name], id_name_dict) for name in names}
    late_graph_layers_nx = {name: late_fusion_transient_absorption(graph_layers_nx[name], best_partit_nx[name]) for name in names}
    return EF_leiden(late_graph_layers_nx, runs, alphas_grid, id_name_dict, node_list), best_partit_nx


def combine_ig(graph_layers, alphas):
    '''Make a graph out of graph layers using balance parameters alpha'''
    adj_matricies = {name: np.array(graph_layers[name].get_adjacency(attribute='weight').data) for name in graph_layers.keys()}
    adj_matrix_resulted = np.sum([np.multiply(adj_matricies[name], alphas[name]) for name in graph_layers.keys()], axis=0)
    new_graph = ig.Graph.Adjacency(adj_matrix_resulted)
    return new_graph


def combine_nx(graph_layers, alphas, node_list):
    '''Make a graph out of graph layers using balance parameters alpha'''
    adj_matrix_new = np.zeros(shape=(len(node_list), len(node_list)))
    for layer_name, graph in graph_layers.items():
        temp_adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list, weight='weight').todense()
        adj_matrix_new += alphas[layer_name] * temp_adj_matrix
    new_graph = nx.from_numpy_matrix(adj_matrix_new)
    nx.relabel.relabel_nodes(new_graph, {val: name for val, name in enumerate(node_list)}, copy=False)
    return new_graph


def take_first(dictionary):
    return next(iter(dictionary))

def leiden(graph_nx, simulations_number):
    node_list = take_first(graph_nx.values()).nodes()
    graph_ig = {key: from_nx_to_ig_graph(graph_nx[key], node_list) for key in graph_nx.keys()}
    id_name_dict = {key: {x.index: x['name'] for x in graph_ig[key].vs} for key in graph_nx.keys()}
    
    v0 = None
    for x in id_name_dict.values():
        if v0 is None:
            v0 = x
        else:
            assert x == v0, f'{x}!={v0}'
    
    id_name_dict = list(id_name_dict.values())[0]
    
    alphas_grid = [(val, 1-val) for val in np.linspace(0, 1, 21)]
    #  Calculate partitions
    sf_partit = SF_leiden(graph_ig, simulations_number, alphas_grid, id_name_dict)
    ef_partit = EF_leiden(graph_nx, simulations_number, alphas_grid, id_name_dict, node_list)
    lf_partit, best_partit = LF_leiden(graph_nx, simulations_number, alphas_grid, id_name_dict, node_list)
    
#     best_partit = {k: val for k, val in zip(new_graph_nx.keys(), [str_partit, att_partit])}
    partition = {'sf': sf_partit, 'ef': ef_partit, 'lf': lf_partit}
    
    Q_sf = {method: compute_experiment('sf', partition[method], graph_nx, node_list) for method in ['sf', 'ef', 'lf']}
    Q_ef = {method: compute_experiment('ef', partition[method], graph_nx, node_list) for method in ['sf', 'ef', 'lf']}
    
    short_graph_layers_lf = {item[0]: late_fusion_transient_absorption(item[1], best_partit[item[0]]) 
                             for item in graph_nx.items()}
    Q_lf = {method: compute_experiment('lf', partition[method], short_graph_layers_lf, node_list) for method in ['sf', 'ef', 'lf']}
    
    return Q_sf, Q_ef, Q_lf, partition


def compute_experiment(objective_function, partition, graph_layers_nx, node_list):
    x, y = [], []
    for alpha, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers_nx.keys(), alpha)}
        if objective_function == 'sf':
            for part in tqdm_notebook(runs, total=len(runs), leave=False):
                y.append(compute_composite_modularity(graph_layers_nx, alphas_dict, part))
                x.append(alpha)

        if objective_function == 'ef' or objective_function == 'lf':
            G_wb = combine_nx(graph_layers_nx, alphas_dict, node_list)
            for part in tqdm_notebook(runs, total=len(runs), leave=False):
                y.append(community.modularity(part, G_wb, weight='weight'))
                x.append(alpha)
    return x, y


def compute_composite_modularity(graph_layers, alphas, partit):
    components = [community.modularity(partit, graph_layers[layer]) * alphas[layer] for layer in graph_layers.keys()]
    return np.sum(components)


def define_clusters(partition):
    clust_dict = {com: [] for com in pd.unique(list(partition.values()))}
    for node, com in partition.items():
        clust_dict[com].append(node)
    return clust_dict


def define_delta_2layers(str_graph, att_graph, alpha, partition):
    graph = {'str': str_graph, 'att': att_graph}
    clusters = define_clusters(partition)
    result = 0
    for com, nodes in clusters.items():
        com_degree = {'str': 0, 'att': 0}
        sum1, sum2 = 0, 0
        for key in com_degree.keys():
            com_degree[key] = np.sum([item for name, item in graph[key].degree(nodes, 'weight')])
            sum1 += alpha[key] * com_degree[key]**2
            sum2 += alpha[key] * com_degree[key]
        result += sum1 - sum2**2
    return result / 4


def delta_experiment(partition, str_graph, att_graph):
    x, y = [], []
    for alpha, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        for part in tqdm_notebook(runs, total=len(runs), leave=False):
            y.append(define_delta_2layers(str_graph, att_graph, {'str': alpha[0], 'att': alpha[1]}, part))
            x.append(alpha)
    return x, y


def delta_new_experiment(partition, new_graph_layers_set):
    x, y = [], []
    for alpha, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        for part in tqdm_notebook(runs, total=len(runs), leave=False):
            for graph_layers in new_graph_layers_set:
                y.append(define_delta_2layers(graph_layers['str'], graph_layers['att'], {'str': alpha[0], 'att': alpha[1]}, part))
                x.append(alpha)
    return x, y


def one_layer_theta(graph, new_graph, partition):
    sum1, sum2 = 0, 0
    communities = set(partition.values())
    for com in communities:
        com_nodes = [key for key, val in partition.items() if val == com]
        sum21, sum22 = 0, 0
        for node in com_nodes:
            for node2 in com_nodes:
                if node != node2:
                    sum1 += new_graph.get_edge_data(node, node2, default={'weight': 0})['weight']
                    sum1 -= graph.get_edge_data(node, node2, default={'weight': 0})['weight']
            sum21 += float(new_graph.degree(node, weight='weight'))
            sum22 += float(graph.degree(node, weight='weight'))
        sum2 += sum21**2 - sum22**2
    return 0.5 * sum1 - 0.25 * sum2


def global_theta(graph_layers, new_graph_layers, partition, alphas):
    '''Lemma 1'''
    theta = 0
    for layer in graph_layers.keys():
        theta += alphas[layer] * one_layer_theta(graph_layers[layer], new_graph_layers[layer], partition)
    return theta


def theta_experiment(partition, best_partition_phase1, graph_layers):
    x, y = [], []
    for alphas, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        for part in tqdm_notebook(runs, total=len(runs), leave=False):
            new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name], 
                                                                             best_partition_phase1[layer_name]) 
                                for layer_name in graph_layers.keys()}
            y.append(global_theta(graph_layers, new_graph_layers, part, {'str': alphas[0], 'att': alphas[1]}))
            x.append(alphas)
    return x, y


def plot_dots(xy, title, file_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[val1 for val1, val2 in xy['ef'][0]], y=xy['ef'][1], 
                              mode="markers", 
                              name='Early Fusion', 
                              marker=dict(size=15), 
                              line=dict(color='olivedrab', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[val1 for val1, val2 in xy['sf'][0]], y=xy['sf'][1], 
                              mode="markers", 
                              name='Simultaneous Fusion', 
                              marker=dict(size=10), 
                              line=dict(color='deepskyblue', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[val1 for val1, val2 in xy['lf'][0]], y=xy['lf'][1], 
                              mode="markers", 
                              name='Late Fusion', 
                              marker=dict(size=7), 
                              line=dict(color='tomato', width=2, dash='dash')))
    fig.update_layout(xaxis=dict(title=r'$\alpha$'), yaxis=dict(title=title))
    fig.update_layout(showlegend=False)
    fig.write_image(file_name)
    fig.show()
    
    
def is_good(p, comms):
    for i in range(comms):
        if i not in p:
            return False
    return True


def define_partitions(n):
    """n - number of nodes in graph"""
    partitions = []
    for i in range(n):
        coms_set = [list(p) for p in product(range(i + 1), repeat=n) if is_good(p, i + 1)]
        for coms in coms_set:
            partitions.append({dot: com for dot, com in enumerate(coms)})
    return partitions


def set_weights(graph, adj_matrix=None):
    for node1, node2 in list(graph.edges):
        if adj_matrix is None:
            graph.add_edge(node1, node2, weight=1)
        else:
            graph.add_edge(node1, node2, weight=adj_matrix[node1, node2])
    return graph


def define_the_best_partitions_ef(graph):
    # define partition diversity
    nodes_number = graph.number_of_nodes()
    diversity_partitions = define_partitions(nodes_number)
    
    # define the best partitions in partition diversity
    result_partitions = []
    stored_mod = -1
    for partit in diversity_partitions:
        current_mod = community.modularity(partit, graph)
        if current_mod > stored_mod:  # replace if current modularity is bigger then the stored one
            result_partitions = [partit]
            stored_mod = current_mod
        elif current_mod == stored_mod:  # add partition if the modularity is the same as the stored one
            result_partitions.append(partit)
    return result_partitions


def define_the_best_partitions_sf(graph_layers, alphas_dict):
    # define partition diversity
    nodes_number = np.max([gr.number_of_nodes() for gr in list(graph_layers.values())])
    diversity_partitions = define_partitions(nodes_number)
    
    # define the best partitions in partition diversity
    result_partitions = []
    stored_mod = -1
    for partit in diversity_partitions:
        current_mod = np.sum([alphas_dict[layer_name] * community.modularity(partit, graph_layers[layer_name]) 
                              for layer_name in graph_layers.keys()])
        if current_mod > stored_mod:  # replace if current modularity is bigger then the stored one
            result_partitions = [partit]
            stored_mod = current_mod
        elif current_mod == stored_mod:  # add partition if the modularity is the same as the stored one
            result_partitions.append(partit)
    return result_partitions


def brute_force(method_name, graph_layers, alphas_grid):
    result_partitions = {alphas: None for alphas in alphas_grid}
    
    # define the best partitions on the 1st phase of lf method
    if method_name == 'lf':
        best_partit_phase1 = {layer: define_the_best_partitions_ef(graph_layers[layer]) for layer in graph_layers.keys()}
        partition_combinations_phase1 = list(product(*best_partit_phase1.values()))  # tuples of partitions
    
    for alphas in tqdm_notebook(alphas_grid, leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), alphas)}
        if method_name == 'lf':
            result_partitions_step = []
            for partit_phase1 in partition_combinations_phase1:
                par_phase1_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), partit_phase1)}
                new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name], par_phase1_dict[layer_name]) 
                                    for layer_name in graph_layers.keys()}
                weighted_graph = combine(new_graph_layers, alphas_dict)
                result_partitions_step.extend(define_the_best_partitions_ef(weighted_graph))
        elif method_name == 'ef':
            weighted_graph = combine(graph_layers, alphas_dict)
            result_partitions_step = define_the_best_partitions_ef(weighted_graph)
        elif method_name == 'sf':
            result_partitions_step = define_the_best_partitions_sf(graph_layers, alphas_dict)
            
        result_partitions[alphas] = result_partitions_step.copy()
    return result_partitions


def define_modularity_set(method_name, partitions_set, graph_layers):
    x, y = [], []
    for alpha in tqdm_notebook(partitions_set.keys(), leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), alpha)}
        for partit in partitions_set[alpha]:
            # LF
            if method_name == 'lf':
                best_partit_phase1 = {layer: define_the_best_partitions_ef(graph_layers[layer]) for layer in graph_layers.keys()}
                partition_combinations_phase1 = list(product(*best_partit_phase1.values()))
                for partit_phase1 in partition_combinations_phase1:
                    par_phase1_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), partit_phase1)}
                    new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name], par_phase1_dict[layer_name]) 
                                        for layer_name in graph_layers.keys()}
                    weighted_graph = combine(new_graph_layers, alphas_dict)
                    y.append(community.modularity(partit, weighted_graph))
                    x.append(alpha)
                    
            # EF
            elif method_name == 'ef':
                weighted_graph = combine(graph_layers, alphas_dict)
                y.append(community.modularity(partit, weighted_graph))
                x.append(alpha)
            
            # SF
            if method_name == 'sf':
                y.append(np.sum([alphas_dict[layer_name] * community.modularity(partit, graph_layers[layer_name]) 
                         for layer_name in graph_layers.keys()]))
                x.append(alpha)
    return x, y