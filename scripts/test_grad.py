import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from scipy.sparse.csgraph import shortest_path
import matplotlib.colors as mcol
from matplotlib import cm
import torch
import time

from fgw.barycenter import fgw_barycenters


def find_thresh(C, inf=0.5, sup=3, step=10):
    """ Trick to find the adequate thresholds from where value of the C matrix are considered close enough to say that nodes are connected
        The threshold is found by a linesearch between values "inf" and "sup" with "step" thresholds tested.
        The optimal threshold is the one which minimizes the reconstruction error between the shortest_path matrix coming from the thresholded adjacency matrix
        and the original matrix.
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix to threshold
    inf : float
          The beginning of the linesearch
    sup : float
          The end of the linesearch
    step : integer
            Number of thresholds tested
    """
    dist = []
    search = np.linspace(inf, sup, step)
    for thresh in search:
        Cprime = sp_to_adjacency(C, 0, thresh)
        SC = shortest_path(Cprime, method='D')
        SC[SC == float('inf')] = 100
        dist.append(np.linalg.norm(SC - C))
    return search[np.argmin(dist)], dist


def sp_to_adjacency(C, threshinf=0.2, threshsup=1.8):
    """ Thresholds the structure matrix in order to compute an adjacency matrix.
    All values between threshinf and threshsup are considered representing connected nodes and set to 1. Else are set to 0
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
        The structure matrix to threshold
    threshinf : float
        The minimum value of distance from which the new value is set to 1
    threshsup : float
        The maximum value of distance from which the new value is set to 1
    Returns
    -------
    C : ndarray, shape (n_nodes,n_nodes)
        The threshold matrix. Each element is in {0,1}
    """
    H = np.zeros_like(C)
    np.fill_diagonal(H, np.diagonal(C))
    C = C - H
    C = np.minimum(np.maximum(C, threshinf), threshsup)
    C[C == threshsup] = 0
    C[C != 0] = 1

    return C


def build_noisy_circular_graph(N=20, mu=0, sigma=0.3, with_noise=False, structure_noise=False, p=None):
    """ Create a noisy circular graph
    """
    g = nx.Graph()
    g.add_nodes_from(list(range(N)))
    for i in range(N):
        noise = float(np.random.normal(mu, sigma, 1))
        if with_noise:
            g.add_node(i, attr_name=math.sin((2 * i * math.pi / N)) + noise)
        else:
            g.add_node(i, attr_name=math.sin(2 * i * math.pi / N))
        g.add_edge(i, i + 1)
        if structure_noise:
            randomint = np.random.randint(0, p)
            if randomint == 0:
                if i <= N - 3:
                    g.add_edge(i, i + 2)
                if i == N - 2:
                    g.add_edge(i, 0)
                if i == N - 1:
                    g.add_edge(i, 1)
    g.add_edge(N, 0)
    noise = float(np.random.normal(mu, sigma, 1))
    if with_noise:
        g.add_node(N, attr_name=math.sin((2 * N * math.pi / N)) + noise)
    else:
        g.add_node(N, attr_name=math.sin(2 * N * math.pi / N))
    return g


def graph_colors(nx_graph, vmin=0, vmax=7):
    cnorm = mcol.Normalize(vmin=vmin, vmax=vmax)
    cpick = cm.ScalarMappable(norm=cnorm, cmap='viridis')
    cpick.set_array([])
    val_map = {}
    for k, v in nx.get_node_attributes(nx_graph, 'attr_name').items():
        val_map[k] = cpick.to_rgba(v)
    colors = []
    for node in nx_graph.nodes():
        colors.append(val_map[node])
    return colors

np.random.seed(30)
X0 = []
for k in range(9):
    X0.append(build_noisy_circular_graph(np.random.randint(15, 25), with_noise=True, structure_noise=True, p=3))



##############################################################################
# Barycenter computation
# ----------------------
device = 'cuda'
Cs = [shortest_path(nx.adjacency_matrix(x).todense()) for x in X0]
Cs = [torch.from_numpy(x).double().to(device) for x in Cs]
Cs = [C / C.max() for C in Cs]  # NOTE: avoid numerical issues
ps = [torch.ones(len(x.nodes())).double().to(device) / len(x.nodes()) for x in X0]
Ys = [np.array([v for (k, v) in nx.get_node_attributes(x, 'attr_name').items()]).reshape(-1, 1) for x in X0]
Ys = [torch.from_numpy(x).double().to(device) for x in Ys]
sizebary = 15  # we choose a barycenter with 15 nodes
p = torch.ones(sizebary).double().to(device) / sizebary

# require grads
for C in Cs:
    C.requires_grad = True
for pt in ps:
    pt.requires_grad = True
for Y in Ys:
    Y.requires_grad = True
p.requires_grad = True

# NOTE: remember to set random seed
A, C, log = fgw_barycenters(sizebary, Ys, Cs, ps, epsilon=0.5, warmstartT=True, seed=int(time.time()), loss_fun='kl_loss',
                            alpha=0.95, tol=1e-3, max_iter=100, solver='BAPG', log=True, verbose=True, numItermax=20)

print(A)
print(C)
