import networkx as nx
import numpy as np
from louvain_modified import louvain_modified_algorithm
from community import community_louvain
from tqdm import tqdm, notebook
import plotly.graph_objects as go

n = 4
m = 4
k = 4
c = 4
d = 4
r = 4

M = np.bmat([[np.ones(n)-np.eye(n), np.zeros((n,m))], [np.zeros((m,n)), np.ones(m)-np.eye(m)]])
K = np.bmat([[np.ones(n)-np.eye(n), np.zeros((n,m)), np.zeros((n,k))],
             [np.zeros((m,n)), np.ones(m)-np.eye(m), np.zeros((m,k))],
             [np.zeros((k,n)), np.zeros((k,m)), np.ones(k)-np.eye(k)]])
C = np.bmat([[np.ones(n)-np.eye(n), np.zeros((n,m)), np.zeros((n,k)), np.zeros((n,c))],
             [np.zeros((m,n)), np.ones(m)-np.eye(m), np.zeros((m,k)), np.zeros((m,c))],
             [np.zeros((k,n)), np.zeros((k,m)), np.ones(k)-np.eye(k), np.zeros((k,c))],
             [np.zeros((c,n)), np.zeros((c,m)), np.zeros((c,k)), np.ones(c)-np.eye(c)]])
D = np.bmat([[np.ones(n)-np.eye(n), np.zeros((n,m)), np.zeros((n,k)), np.zeros((n,c)), np.zeros((n,d))],
             [np.zeros((m,n)), np.ones(m)-np.eye(m), np.zeros((m,k)), np.zeros((m,c)), np.zeros((m,d))],
             [np.zeros((k,n)), np.zeros((k,m)), np.ones(k)-np.eye(k), np.zeros((k,c)), np.zeros((k,d))],
             [np.zeros((c,n)), np.zeros((c,m)), np.zeros((c,k)), np.ones(c)-np.eye(c), np.zeros((c,d))],
             [np.zeros((d,n)), np.zeros((d,m)), np.zeros((d,k)), np.zeros((d,c)), np.ones(d)-np.eye(d)]])
R = np.bmat([[np.ones(n)-np.eye(n), np.zeros((n,m)), np.zeros((n,k)), np.zeros((n,c)), np.zeros((n,d)), np.zeros((n,r))],
             [np.zeros((m,n)), np.ones(m)-np.eye(m), np.zeros((m,k)), np.zeros((m,c)), np.zeros((m,d)), np.zeros((m,r))],
             [np.zeros((k,n)), np.zeros((k,m)), np.ones(k)-np.eye(k), np.zeros((k,c)), np.zeros((k,d)), np.zeros((k,r))],
             [np.zeros((c,n)), np.zeros((c,m)), np.zeros((c,k)), np.ones(c)-np.eye(c), np.zeros((c,d)), np.zeros((c,r))],
             [np.zeros((d,n)), np.zeros((d,m)), np.zeros((d,k)), np.zeros((d,c)), np.ones(d)-np.eye(d), np.zeros((d,r))],
             [np.zeros((r,n)), np.zeros((r,m)), np.zeros((r,k)), np.zeros((r,c)), np.zeros((r,d)), np.ones(r)-np.eye(r)]])

matrix = R
print('Matrix')
print(matrix)
structural_graph = nx.from_numpy_matrix(matrix)
partition = community_louvain.best_partition(structural_graph)
Q = community_louvain.modularity(partition, structural_graph)
print('Modularity = ', Q)