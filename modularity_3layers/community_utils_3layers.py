import numpy as np
import pandas as pd
import random
from itertools import product

import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

import networkx as nx
import igraph as ig
import community
from louvain_modified import louvain_modified_algorithm as LM
from leidenalg import Optimiser
import leidenalg

from tqdm import tqdm_notebook
# import warnings
from plotly.offline import init_notebook_mode

# warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)


def read_raw_file(file_name):
    raw_file = open(file_name, 'r')
    for line in raw_file:
        partition = eval(line)
    raw_file.close()
    return partition


def from_nx_to_ig_graph(nx_graph, node_list):
    g = ig.Graph()
    g.add_vertices(len(node_list))
    g.vs['name'] = list(node_list)
    g.add_edges(list(nx_graph.edges()))
    g.es['weight'] = list(nx.get_edge_attributes(nx_graph, 'weight').values())
    return g


def from_ig_to_nx_partition_multiplex(partition_ig, graph_ig):
    partition_nx = {list(graph_ig.vs)[node]['name']: com for node, com in enumerate(partition_ig)}
    return partition_nx


def from_ig_to_nx_partition_ordinary(partition_ig, graph_ig):
    com_number = 0
    nodes = []
    coms = []
    for item in partition_ig:
        for node in item:
            nodes.append(node)
            coms.append(com_number)
        com_number += 1
    partition_nx = {list(graph_ig.vs)[node]['name']: com for node, com in zip(nodes, coms)}
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


def normalize_to_fixed_weight_nx(old_graph, total_weight, show=False):
    graph = old_graph.copy()
    whole_weight = 0
    for item in graph.edges(data=True):
        whole_weight += item[2]['weight']
    
    for edge in graph.edges(data=True):
        if whole_weight * total_weight > 0:
            graph[edge[0]][edge[1]]['weight'] = edge[2]['weight'] / whole_weight * total_weight
        else:
            graph[edge[0]][edge[1]]['weight'] = 0
    
    if show:
        print('The whole weight has been equalt to', whole_weight, '. Now it is equal to', check_the_whole_weight(graph))
    return graph


def check_the_whole_weight(graph):
    check = 0
    for edge in graph.edges(data=True):
        check += edge[2]['weight']
    return check


def combine_nx(graph_layers, alphas, alpha_order, node_list):
    """Make a graph out of graph layers using balance parameters alpha"""
    adj_matrix_new = np.zeros(shape=(len(node_list), len(node_list)))
    for layer_name, graph in graph_layers.items():
        temp_adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list, weight='weight').todense()
        adj_matrix_new += alphas[alpha_order[layer_name]] * temp_adj_matrix
    new_graph = nx.from_numpy_matrix(adj_matrix_new)
    nx.relabel.relabel_nodes(new_graph, {val: name for val, name in enumerate(node_list)}, copy=False)
    return new_graph


def take_first(dictionary):
    return next(iter(dictionary))


def SF_leiden(graph_layers_nx, runs, alpha_grid, alpha_order):
    node_list = take_first(graph_layers_nx.values()).nodes()
    partition = {alphas: [] for alphas in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        # new_layers_nx = graph_layers_nx.copy()
        # new_layers_nx = {name: normalize_to_fixed_weight_nx(new_layers_nx[name], key_alpha[alpha_order[name]])
        #                  for name in new_layers_nx.keys()}
        graph_layers_ig = {key: from_nx_to_ig_graph(graph_layers_nx[key], node_list) for key in graph_layers_nx.keys()}
        key_list = [key_alpha[alpha_order[name]] for name in graph_layers_ig.keys()]
        for simulation in range(runs):
            membership, improvement = find_partition_multiplex_layers(list(graph_layers_ig.values()),
                                                                           key_list,
                                                                           leidenalg.ModularityVertexPartition,
                                                                           n_iterations=1,
                                                                           seed=random.randint(0, 1000))
            nx_partit = from_ig_to_nx_partition_multiplex(membership, graph_layers_ig[take_first(graph_layers_ig)])
            partition[key_alpha].append(nx_partit)
    return partition


def EF_leiden(graph_layers_nx, runs, alpha_grid, alpha_order):
    node_list = take_first(graph_layers_nx.values()).nodes()
    partition = {alpha: [] for alpha in alpha_grid}
    for alpha in tqdm_notebook(partition.keys(), leave=False):
        ef_graph_nx = combine_nx(graph_layers_nx, alpha, alpha_order, node_list)
        ef_graph_ig = from_nx_to_ig_graph(ef_graph_nx, node_list)
        for simulation in range(runs):
            weights = [item['weight'] for item in list(ef_graph_ig.es)]
            membership = leidenalg.find_partition(ef_graph_ig, leidenalg.ModularityVertexPartition, n_iterations=1, 
                                                  seed=random.randint(0, 1000),
                                                  weights=weights)
            nx_partit = from_ig_to_nx_partition_ordinary(membership, ef_graph_ig)
            partition[alpha].append(nx_partit)
    return partition


def LF_leiden(graph_layers_nx, runs, alphas_grid, alpha_order):
    node_list = take_first(graph_layers_nx.values()).nodes()
    names = graph_layers_nx.keys()
    graph_layers_ig = {name: from_nx_to_ig_graph(graph_layers_nx[name], node_list) for name in names}
    best_partit_ig = {name: leidenalg.find_partition(graph_layers_ig[name], 
                                                     leidenalg.ModularityVertexPartition, 
                                                     n_iterations=1,
                                                     seed=random.randint(0, 1000),
                                                     weights=[item['weight'] for item in list(graph_layers_ig[name].es)])
                      for name in names}
    best_partit_nx = {name: from_ig_to_nx_partition_ordinary(best_partit_ig[name], graph_layers_ig[take_first(graph_layers_ig)]) 
                      for name in names}
    late_graph_layers_nx = {name: late_fusion_transient_absorption(graph_layers_nx[name], best_partit_nx[name]) for name in names}
    return EF_leiden(late_graph_layers_nx, runs, alphas_grid, alpha_order), best_partit_nx


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
        weight_for_comm = val / (nodes_number * (nodes_number - 1))
        for node1 in node_list_comm[com]:  #Distribute the absorbed edges
            for node2 in node_list_comm[com]:
                if node1 != node2:
                    if G_late.has_edge(node1, node2):
                        G_late[node1][node2]['weight'] = G_late.get_edge_data(node1, node2)['weight'] + weight_for_comm
                    else:
                        G_late.add_edge(node1, node2, weight = weight_for_comm)
    return G_late


def SF_louvain(graph, runs, alpha_grid, alpha_order):
    partition = {alpha: [] for alpha in alpha_grid}
    graph_list = list(graph.values())
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        for simulation in tqdm_notebook(range(runs), leave=False):
            partition[key_alpha].append(LM.best_partition(graph_list[0], graph_list[1], alpha=key_alpha[0]))
    return partition


def EF_louvain(graph_layers_nx, runs, alpha_grid, alpha_order):
    node_list = take_first(graph_layers_nx.values()).nodes()
    partition = {alpha: [] for alpha in alpha_grid}
    for key_alpha in tqdm_notebook(partition.keys(), leave=False):
        alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers_nx.keys(), key_alpha)}
        ef_graph_nx = combine_nx(graph_layers_nx, key_alpha, alpha_order, node_list)
        for simulation in range(runs):
            partition[key_alpha].append(community.best_partition(ef_graph_nx))
    return partition


def LF_louvain(graph_layers_nx, runs, alphas_grid, alpha_order):
    names = graph_layers_nx.keys()
    best_partit_nx = {name: community.best_partition(graph_layers_nx[name]) for name in names}
    late_graph_layers_nx = {name: late_fusion_transient_absorption(graph_layers_nx[name], best_partit_nx[name]) for name in names}
    return EF_louvain(late_graph_layers_nx, runs, alphas_grid, alpha_order), best_partit_nx


def compute_experiment(objective_function, partition, graph_layers_nx, node_list, alpha_order):
    x, y = [], []
    for alpha, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
#         alphas_dict = {layer_name: alpha_order[layer_name] for layer_name in graph_layers_nx.keys()}
        if objective_function == 'sf':
            for part in runs:
                y.append(compute_composite_modularity(graph_layers_nx, alpha, alpha_order, part))
                x.append(alpha)

        if objective_function == 'ef' or objective_function == 'lf':
            G_wb = combine_nx(graph_layers_nx, alpha, alpha_order, node_list)
            for part in runs:
                y.append(community.modularity(part, G_wb, weight='weight'))
                x.append(alpha)
    return x, y


def compute_composite_modularity(graph_layers, alphas, alpha_order, partit):
    non_zero_layers = [name for name in graph_layers.keys() if alphas[alpha_order[name]] != 0]
    components = [community.modularity(partit, graph_layers[layer]) * alphas[alpha_order[layer]] for layer in non_zero_layers]
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


def define_delta(graph, alpha, alpha_order, partition):
    clusters = define_clusters(partition)
    result = 0
    for com, nodes in clusters.items():
        com_degree = {key: 0 for key in graph.keys()}
        sum1, sum2 = 0, 0
        for key in com_degree.keys():
            com_degree[key] = np.sum([item for name, item in graph[key].degree(nodes, 'weight')])
            sum1 += alpha[alpha_order[key]] * com_degree[key]**2
            sum2 += alpha[alpha_order[key]] * com_degree[key]
        result += sum1 - sum2**2
    return result / 4


def delta_experiment(partition, graph_layers, alpha_order):
    x, y = [], []
    for alpha, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        for part in runs:
            y.append(define_delta(graph_layers, alpha, alpha_order, part))
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


def global_theta(graph_layers, new_graph_layers, partition, alphas, alpha_order):
    """Lemma 1"""
    theta = 0
    for layer in graph_layers.keys():
        theta += alphas[alpha_order[layer]] * one_layer_theta(graph_layers[layer], new_graph_layers[layer], partition)
    return theta


def theta_experiment(partition, best_partition_phase1, graph_layers, alpha_order):
    x, y = [], []
    for alphas, runs in tqdm_notebook(partition.items(), total=len(partition), leave=False):
        for part in runs:
            new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name], 
                                                                             best_partition_phase1[layer_name]) 
                                for layer_name in graph_layers.keys()}
            y.append(global_theta(graph_layers, new_graph_layers, part, alphas, alpha_order))
            x.append(alphas)
    return x, y


def plot_dots_2d(xy, title, file_name):
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

    
def data_fixed_alpha(data, alpha_number, alpha_value):
    result = {key: None for key in data.keys()}
    for target_function in data.keys():
        alphas = np.array(data[target_function][0])  # mask
        modularities = np.array(data[target_function][1])
        alpha_left = alphas[alphas[:, alpha_number-1] == alpha_value]
        alphas_left_2d = np.delete(alpha_left, alpha_number-1, axis=1)
        modularity_left = modularities[alphas[:, alpha_number-1] == alpha_value]
        result[target_function] = [alphas_left_2d, modularity_left]
    return result


def plot_dots(xy, title, file_name):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(projection='3d')
    x, y, z, color = [], [], [], []
    for target_function, col in zip(xy.keys(), ['deepskyblue', 'olivedrab', 'tomato']):
        x.extend([val1 for val1, val2, val3 in xy[target_function][0]])
        y.extend([val2 for val1, val2, val3 in xy[target_function][0]])
        z.extend(xy[target_function][1])
        color.extend([col] * len(xy[target_function][1]))
        
    ax.scatter(x, y, z, marker='o', s=3, c=color)
    
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(title)

    plt.show()
    fig.savefig(file_name, bbox_inches='tight')
    
    
def plot_surface(xy, title, file_name):
    fig = plt.figure()
    ax = Axes3D(fig)
    for target_function, col in zip(xy.keys(), ['deepskyblue', 'olivedrab', 'tomato']):
        x = [val1 for val1, val2, val3 in xy[target_function][0]]
        y = [val2 for val1, val2, val3 in xy[target_function][0]]
        z = xy[target_function][1]
        surf = ax.plot_trisurf(x, y, z, linewidth=0.1, alpha=0.5, color=col) #cmap=cm.jet,

    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(title)

    
#     cset = ax.contourf(x, y, z, zdir='z', offset=-100, cmap=cm.coolwarm)
#     cset = ax.contourf(x, y, z, zdir='x', offset=-40, cmap=cm.coolwarm)
#     cset = ax.contourf(x, y, z, zdir='y', offset=40, cmap=cm.coolwarm)
    
    plt.show()
    fig.savefig(file_name)

    
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


def brute_force(method_name, graph_layers, alphas_grid, alpha_order, node_list):
    result_partitions = {alphas: None for alphas in alphas_grid}
    
    # define the best partitions on the 1st phase of lf method
    if method_name == 'lf':
        best_partit_phase1 = {layer: define_the_best_partitions_ef(graph_layers[layer]) for layer in graph_layers.keys()}
        partition_combinations_phase1 = list(product(*best_partit_phase1.values()))  # tuples of partitions
    
    for alphas in tqdm_notebook(alphas_grid, leave=False):
        alphas_dict = {layer_name: alphas[alpha_order[layer_name]] for layer_name in graph_layers.keys()}
        if method_name == 'lf':
            result_partitions_step = []
            for partit_phase1 in partition_combinations_phase1:
                par_phase1_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), partit_phase1)}
                new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name],
                                                                                 par_phase1_dict[layer_name])
                                    for layer_name in graph_layers.keys()}
                weighted_graph = combine_nx(new_graph_layers, alphas, alpha_order, node_list)
                result_partitions_step.extend(define_the_best_partitions_ef(weighted_graph))
        elif method_name == 'ef':
            weighted_graph = combine_nx(graph_layers, alphas, alpha_order, node_list)
            result_partitions_step = define_the_best_partitions_ef(weighted_graph)
        elif method_name == 'sf':
            result_partitions_step = define_the_best_partitions_sf(graph_layers, alphas_dict)
            
        result_partitions[alphas] = result_partitions_step.copy()
    return result_partitions


def define_modularity_set(method_name, partitions_set, graph_layers, alpha_order, node_list):
    x, y = [], []
    for alpha in tqdm_notebook(partitions_set.keys(), leave=False):
        # alphas_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), alpha)}
        for partit in partitions_set[alpha]:
            # LF
            if method_name == 'lf':
                best_partit_phase1 = {layer: define_the_best_partitions_ef(graph_layers[layer]) for layer in graph_layers.keys()}
                partition_combinations_phase1 = list(product(*best_partit_phase1.values()))
                for partit_phase1 in partition_combinations_phase1:
                    par_phase1_dict = {layer_name: value for layer_name, value in zip(graph_layers.keys(), partit_phase1)}
                    new_graph_layers = {layer_name: late_fusion_transient_absorption(graph_layers[layer_name], par_phase1_dict[layer_name]) 
                                        for layer_name in graph_layers.keys()}
                    weighted_graph = combine_nx(new_graph_layers, alpha, alpha_order, node_list)
                    y.append(community.modularity(partit, weighted_graph))
                    x.append(alpha)
                    
            # EF
            elif method_name == 'ef':
                weighted_graph = combine_nx(graph_layers, alpha, alpha_order, node_list)
                y.append(community.modularity(partit, weighted_graph))
                x.append(alpha)
            
            # SF
            if method_name == 'sf':
                y.append(np.sum([alpha[alpha_order[layer_name]] * community.modularity(partit, graph_layers[layer_name])
                         for layer_name in graph_layers.keys()]))
                x.append(alpha)
    return x, y


def plot_ticks(start, stop, tick, n):
    r = np.linspace(0, 1, n+1)
    x = start[0] * (1 - r) + stop[0] * r
    x = np.vstack((x, x + tick[0]))
    y = start[1] * (1 - r) + stop[1] * r
    y = np.vstack((y, y + tick[1]))
    plt.plot(x, y, 'k', lw=1)

    
def ternary_plot(test_data, vmin, vmax):
    n = 5
    tick_size = 0.1
    margin = 0.05

    # define corners of triangle    
    left = np.r_[0, 0]
    right = np.r_[1, 0]
    top = np.r_[0.5, np.sqrt(3)*0.576]
    triangle = np.c_[left, right, top, left]

    # define corners of triangle    
    left = np.r_[0, 0]
    right = np.r_[1, 0]
    top = np.r_[0.5, np.sqrt(3)*0.576]
    triangle = np.c_[left, right, top, left]

    # define vectors for ticks
    bottom_tick = 0.8264*tick_size * (right - top) / n
    right_tick = 0.8264*tick_size * (top - left) / n
    left_tick = 0.8264*tick_size * (left - right) / n

    #Define twin axis
    #ax = plt.gca()
    fig, ax = plt.subplots()
    plot_ticks(left, right, bottom_tick, n)
    plot_ticks(right, top, right_tick, n)
    plot_ticks(left, top, left_tick, n)
    #ax2 = ax.twinx()

    # barycentric coords: (a,b,c)
    a=test_data[:,0]
    b=test_data[:,1]
    c=test_data[:,2]

    # values is stored in the last column
    v = test_data[:,-1]

    # translate the data to cartesian corrds
    x = 0.5 * ( 2.*b+c ) / ( a+b+c )
    y = 0.576*np.sqrt(3) * c / (a+b+c)


    # create a triangulation out of these points
    T = tri.Triangulation(x,y)

    # plot the contour
    plt.tricontourf(x,y,T.triangles,v,cmap='viridis',vmin=vmin, vmax=vmax)


    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.576]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    #plotting the mesh and caliberate the axis
    plt.triplot(trimesh,'k--')
    #plt.title('Binding energy peratom of Al-Ti-Ni clusters')
    ax.set_xlabel(r'$\alpha_2 - \alpha_1$',fontsize=12,color='black')
    ax.set_ylabel(r'$\alpha_1 - \alpha_3$',fontsize=12,color='black')
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\alpha_2 - \alpha_3$',fontsize=12,color='black')
    plt.gcf().text(0.07, 0.05, r'$\alpha_1$', fontsize=12,color='black')
    plt.gcf().text(0.93, 0.05, r'$\alpha_2$', fontsize=12,color='black')
    plt.gcf().text(0.5, 0.9, r'$\alpha_3$', fontsize=12,color='black')

    #set scale for axis
    ax.set_xlim(1, 0)
    ax.set_ylim(0, 1)
    ax2.set_ylim(1, 0)

    cax = plt.axes([0.75, 0.55, 0.055, 0.3])
    plt.colorbar(cax=cax,format='%.3f')
    plt.savefig("AID.png", dpi=1000)
    plt.show()
    
    
def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def find_partition_multiplex_layers(graphs, layer_weights, partition_type, n_iterations=2, seed=None, **kwargs):
    partitions = []
    for graph in graphs:
        partitions.append(partition_type(graph, **kwargs))
    optimiser = Optimiser()

    if not seed is None:
        optimiser.set_rng_seed(seed)

    improvement = optimiser.optimise_partition_multiplex(partitions, layer_weights, n_iterations)

    return partitions[0].membership, improvement