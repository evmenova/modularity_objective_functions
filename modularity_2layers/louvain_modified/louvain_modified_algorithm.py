# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

from .louvain_modified_status import Status

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001
# __MIN = 0.0

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition which directly combines partition_at_level and
    generate_dendrogram to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph_s, graph_a, alpha, weight='weight', with_diff=False):
    """Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> part = best_partition(G)
    >>> modularity(part, G)
    """
    if graph_s.is_directed() or graph_a.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc_s, inc_a = dict([]), dict([])
    deg_s, deg_a = dict([]), dict([])
    links_s = graph_s.size(weight=weight)
    links_a = graph_a.size(weight=weight)
    if links_s == 0 or links_a == 0:
        raise ValueError("A graph without link has an undefined modularity")

    res_s = cycle(graph_s, weight, deg_s, inc_s, links_s, partition)
    res_a = cycle(graph_a, weight, deg_a, inc_a, links_a, partition)
    res = res_s * alpha + res_a * (1-alpha)

    size_s = graph_s.size(weight=weight)
    size_a = graph_a.size(weight=weight)

    if with_diff:
        mod_diff = 0.
        for com in set(partition.values()):
            diff_in = 0.
            for node in graph_a.nodes():
                if partition[node] == com:
                    diff_in += graph_s.degree(node) / size_s - graph_a.degree(node) / size_a
            mod_diff += diff_in
        mod_diff = mod_diff ** 2
        res += alpha * (1 - alpha) * mod_diff

    return res


def cycle(graph, weight, deg, inc, links, partition):
    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.
    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
                (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph_s,
                   graph_a,
                   alpha,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   with_diff=False):
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices

    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain algorithm.

    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean, optional
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    See Also
    --------
    generate_dendrogram to obtain all the decompositions levels

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>>  #Basic usage
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> part = best_partition(G)

    >>> #other example to display a graph with its community :
    >>> #better with karate_graph() as defined in networkx examples
    >>> #erdos renyi don't have true community structure
    >>> G = nx.erdos_renyi_graph(30, 0.05)
    >>> #first compute the best partition
    >>> partition = community.best_partition(G)
    >>>  #drawing
    >>> size = float(len(set(partition.values())))
    >>> pos = nx.spring_layout(G)
    >>> count = 0.
    >>> for com in set(partition.values()) :
    >>>     count += 1.
    >>>     list_nodes = [nodes for nodes in partition.keys()
    >>>                                 if partition[nodes] == com]
    >>>     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()
    """
    dendo = generate_dendrogram(graph_s,
                                graph_a,
                                alpha,
                                with_diff,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph_s,
                        graph_a,
                        alpha,
                        with_diff=False,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    if graph_s.is_directed() or graph_a.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    # if graph1.number_of_edges() == 0:
    #     part = dict([])
    #     for i, node in enumerate(graph1.nodes()):
    #         part[node] = i
    #     return [part]

    current_graph_s = graph_s.copy()
    current_graph_a = graph_a.copy()
    status_s = Status()
    status_a = Status()
    status_s.init(current_graph_s, weight, part_init)
    status_a.init(current_graph_a, weight, part_init)
    status_list = list()
    __one_level(current_graph_s, current_graph_a, status_s, status_a, alpha, weight, resolution, random_state,
                with_diff=with_diff, partition=part_init)
    new_mod_s = __modularity(status_s, resolution)
    new_mod_a = __modularity(status_a, resolution)
    new_mod = new_mod_s * alpha + new_mod_a * (1-alpha)
    partition = __renumber(status_s.node2com)  # ----------------------------------------------------------------------- начальный RENUMBER
    status_list.append(partition)
    mod = new_mod
    current_graph_s = induced_graph(partition, current_graph_s, weight)
    current_graph_a = induced_graph(partition, current_graph_a, weight)
    status_s.init(current_graph_s, weight)
    status_a.init(current_graph_a, weight)

    while True:
        __one_level(current_graph_s, current_graph_a, status_s, status_a, alpha, weight, resolution, random_state,
                    with_diff=with_diff, partition=partition)
        new_mod_s = __modularity(status_s, resolution)
        new_mod_a = __modularity(status_a, resolution)
        new_mod = new_mod_s * alpha + new_mod_a * (1-alpha)
        if new_mod - mod <= __MIN:
            break
        partition = __renumber(status_s.node2com)  # ------------------------------------------------------------------- RENUMBER
        status_list.append(partition)
        mod = new_mod
        current_graph_s = induced_graph(partition, current_graph_s, weight)
        current_graph_a = induced_graph(partition, current_graph_a, weight)
        status_s.init(current_graph_s, weight)
        status_a.init(current_graph_a, weight)
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def __one_level(graph_s, graph_a, status_s, status_a, alpha, weight_key, resolution, random_state, with_diff, partition):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod_s = __modularity(status_s, resolution)
    cur_mod_a = __modularity(status_a, resolution)
    cur_mod = cur_mod_s * alpha + cur_mod_a * (1-alpha)
    new_mod = cur_mod

    size_s = graph_s.size(weight=weight_key)
    size_a = graph_a.size(weight=weight_key)

    if with_diff:
        mod_diff = 0.
        for com in set(partition.values()):
            diff_in = 0.
            for node in graph_a.nodes():
                if partition[node] == com:
                    diff_in += graph_s.degree(node) / size_s - graph_a.degree(node) / size_a
            mod_diff += diff_in
        mod_diff = mod_diff ** 2

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph_a.nodes(), random_state):
            com_node = status_a.node2com[node]  # ---------------------------------------------------------------------- без разницы
            degc_totw_s = status_s.gdegrees.get(node, 0.) / (status_s.total_weight * 2.)  # NOQA
            degc_totw_a = status_a.gdegrees.get(node, 0.) / (status_a.total_weight * 2.)
            neigh_communities_s = __neighcom(node, graph_s, status_s, weight_key)
            neigh_communities_a = __neighcom(node, graph_a, status_a, weight_key)

            s_neighbors = neigh_communities_s.keys()
            a_neighbors = neigh_communities_a.keys()
            intersection_neighbors = [value for value in s_neighbors if value in a_neighbors]
            a_neighbors_only = [value for value in a_neighbors if value not in intersection_neighbors]
            neigh_communities = neigh_communities_s.copy()
            neigh_communities.update({key: 0. for key in a_neighbors_only})

            remove_cost_s = - resolution * neigh_communities_s.get(com_node, 0.) + \
                (status_s.degrees.get(com_node, 0.) - status_s.gdegrees.get(node, 0.)) * degc_totw_s
            remove_cost_a = - resolution * neigh_communities_a.get(com_node, 0.) + \
                (status_a.degrees.get(com_node, 0.) - status_a.gdegrees.get(node, 0.)) * degc_totw_a

            __remove(node, com_node,
                     neigh_communities_s.get(com_node, 0.), status_s)
            __remove(node, com_node,
                     neigh_communities_a.get(com_node, 0.), status_a)
            best_com = com_node
            best_increase = 0

            for com, dnc_s in __randomize(neigh_communities.items(), random_state):
                dnc_a = neigh_communities_a.get(com, 0.)
                incr_s = remove_cost_s + resolution * dnc_s - \
                         status_s.degrees.get(com, 0.) * degc_totw_s
                incr_a = remove_cost_a + resolution * dnc_a - \
                         status_a.degrees.get(com, 0.) * degc_totw_a
                incr = incr_s * alpha + incr_a * (1-alpha)

                if with_diff:
                    new_mod_diff = 0.
                    for com in set(partition.values()):
                        diff_in = 0.
                        for node in graph_a.nodes():
                            if partition[node] == com:
                                diff_in += graph_s.degree(node) / size_s - graph_a.degree(node) / size_a
                        new_mod_diff += diff_in
                    new_mod_diff = new_mod_diff ** 2
                    incr_d = new_mod_diff - mod_diff
                    incr += alpha * (1-alpha) * incr_d

                if incr > best_increase:
                    best_increase = incr
                    best_com = com

            __insert(node, best_com,
                     neigh_communities_s.get(best_com, 0.), status_s)
            __insert(node, best_com,
                     neigh_communities_a.get(best_com, 0.), status_a)
            if best_com != com_node:
                modified = True

        new_mod_s = __modularity(status_s, resolution)
        new_mod_a = __modularity(status_a, resolution)
        new_mod = new_mod_s * alpha + new_mod_a * (1-alpha)

        if with_diff:
            new_mod_diff = mod_diff
            new_mod += new_mod_diff * alpha * (1-alpha)

        if new_mod - cur_mod <= __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links - ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items


def differential_part_sq(partition, graph_s, graph_a, alpha, weight='weight'):
    res = 0.
    s_size = graph_s.size(weight=weight)
    a_size = graph_a.size(weight=weight)
    for com in set(partition.values()):
        diff = 0.
        for node in graph_s.nodes():
            if partition[node] == com:
                k_iS = graph_s.degree(node, weight=weight) / s_size
                k_iA = graph_a.degree(node, weight=weight) / a_size
                diff += k_iS - k_iA
        res += diff ** 2 / 4
    return res


def differential_part_full(partition, graph_s, graph_a, alpha, weight='weight'):
    res = 0.
    s_size = graph_s.size(weight=weight)
    a_size = graph_a.size(weight=weight)
    for node in graph_s:
        for neighbor in graph_s:
            if partition[node] == partition[neighbor]:
                k_iS = graph_s.degree(node, weight=weight) / s_size
                k_iA = graph_a.degree(node, weight=weight) / a_size
                k_jS = graph_s.degree(neighbor, weight=weight) / s_size
                k_jA = graph_a.degree(neighbor, weight=weight) / a_size
                res += (k_iS - k_iA) * (k_jS - k_jA)
    res = res / 4
    return res