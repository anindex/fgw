import networkx
from networkx.generators.community import stochastic_block_model as sbm
from time import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from fgw.bregman import fgw

np.random.seed(0)
max_iter = 200
numItermax = 10
N2 = 20  # 2 communities
N3 = 30  # 3 communities
p2 = [[1., 0.1],
      [0.1, 0.9]]
p3 = [[1., 0.1, 0.],
      [0.1, 0.95, 0.1],
      [0., 0.1, 0.9]]
G2 = sbm(seed=0, sizes=[N2 // 2, N2 // 2], p=p2)
G3 = sbm(seed=0, sizes=[N3 // 3, N3 // 3, N3 // 3], p=p3)
part_G2 = [G2.nodes[i]['block'] for i in range(N2)]
part_G3 = [G3.nodes[i]['block'] for i in range(N3)]

C2 = networkx.to_numpy_array(G2)
C3 = networkx.to_numpy_array(G3)


# We add node features with given mean - by clusters
# and inversely proportional to clusters' intra-connectivity

F2 = np.zeros((N2, 1))
for i, c in enumerate(part_G2):
    F2[i, 0] = np.random.normal(loc=c, scale=0.01)

F3 = np.zeros((N3, 1))
for i, c in enumerate(part_G3):
    F3[i, 0] = np.random.normal(loc=2. - c, scale=0.01)

# Compute pairwise euclidean distance between node features
M = (F2 ** 2).dot(np.ones((1, N3))) + np.ones((N2, 1)).dot((F3 ** 2).T) - 2 * F2.dot(F3.T)

h2 = np.ones(C2.shape[0]) / C2.shape[0]
h3 = np.ones(C3.shape[0]) / C3.shape[0]

alpha = 0.5

device = torch.device('cuda')
M = torch.from_numpy(M).float().to(device)
C2 = torch.from_numpy(C2).float().to(device)
C3 = torch.from_numpy(C3).float().to(device)
h2 = torch.from_numpy(h2).float().to(device)
h3 = torch.from_numpy(h3).float().to(device)

# Proximal Point algorithm with Kullback-Leibler as proximal operator
print('Proximal Point Algorithm \n')
start_ppa = time()
T_ppa, log_ppa = fgw(
    M, C2, C3, h2, h3, 'square_loss', alpha=alpha, epsilon=1., solver='PPA',
    tol=1e-9, log=True, verbose=True, warmstart=False, max_iter=max_iter, numItermax=numItermax)
end_ppa = time()
time_ppa = 1000 * (end_ppa - start_ppa)

# Projected Gradient algorithm with entropic regularization
print('Projected Gradient Descent \n')
start_pgd = time()
T_pgd, log_pgd = fgw(
    M, C2, C3, h2, h3, 'square_loss', alpha=alpha, epsilon=1., solver='PGD',
    tol=1e-9, log=True, verbose=True, warmstart=False, max_iter=max_iter, numItermax=numItermax)
end_pgd = time()
time_pgd = 1000 * (end_pgd - start_pgd)

# Alternated Bregman Projected Gradient algorithm with Kullback-Leibler as proximal operator
print('Bregman Alternated Projected Gradient \n')
start_bapg = time()
T_bapg, log_bapg = fgw(
    M, C2, C3, h2, h3, 'square_loss', alpha=alpha, epsilon=1., solver='BAPG',
    tol=1e-9, marginal_loss=True, verbose=True, log=True, max_iter=max_iter, numItermax=numItermax)
end_bapg = time()
time_bapg = 1000 * (end_bapg - start_bapg)

fgw_ppa = log_ppa['fgw_dist'].cpu().numpy()
fgw_pgd = log_pgd['fgw_dist'].cpu().numpy()
fgw_bapg = log_bapg['fgw_dist'].cpu().numpy()

print(f'FGW PPA: Time optim {time_ppa}ms, fgw_dist: {fgw_ppa}')
print(f'FGW PGD: Time optim {time_pgd}ms, fgw_dist: {fgw_pgd}')
print(f'FGW BAPG: Time optim {time_bapg}ms, fgw_dist: {fgw_bapg}')

# compute OT sparsity level
T_ppa_sparsity = 100 * (T_ppa == 0.).sum() / (N2 * N3)
T_pgd_sparsity = 100 * (T_pgd == 0.).sum() / (N2 * N3)
T_bapg_sparsity = 100 * (T_bapg == 0.).sum() / (N2 * N3)

T_ppa_sparsity = T_ppa_sparsity.cpu().numpy()
T_pgd_sparsity = T_pgd_sparsity.cpu().numpy()
T_bapg_sparsity = T_bapg_sparsity.cpu().numpy()

# Methods using Sinkhorn/Bregman projections tend to produce feasibility errors on the
# marginal constraints

err_ppa = torch.norm(T_ppa.sum(1) - h2) + torch.norm(T_ppa.sum(0) - h3)
err_pgd = torch.norm(T_pgd.sum(1) - h2) + torch.norm(T_pgd.sum(0) - h3)
err_bapg = torch.norm(T_bapg.sum(1) - h2) + torch.norm(T_bapg.sum(0) - h3)

err_ppa = err_ppa.cpu().numpy()
err_pgd = err_pgd.cpu().numpy()
err_bapg = err_bapg.cpu().numpy()

T_ppa = T_ppa.cpu().numpy()
T_pgd = T_pgd.cpu().numpy()
T_bapg = T_bapg.cpu().numpy()

# Add weights on the edges for visualization later on
weight_intra_G2 = 5
weight_inter_G2 = 0.5
weight_intra_G3 = 1.
weight_inter_G3 = 1.5

weightedG2 = networkx.Graph()
part_G2 = [G2.nodes[i]['block'] for i in range(N2)]

for node in G2.nodes():
    weightedG2.add_node(node)
for i, j in G2.edges():
    if part_G2[i] == part_G2[j]:
        weightedG2.add_edge(i, j, weight=weight_intra_G2)
    else:
        weightedG2.add_edge(i, j, weight=weight_inter_G2)

weightedG3 = networkx.Graph()
part_G3 = [G3.nodes[i]['block'] for i in range(N3)]

for node in G3.nodes():
    weightedG3.add_node(node)
for i, j in G3.edges():
    if part_G3[i] == part_G3[j]:
        weightedG3.add_edge(i, j, weight=weight_intra_G3)
    else:
        weightedG3.add_edge(i, j, weight=weight_inter_G3)


def draw_graph(G, C, nodes_color_part, Gweights=None,
               pos=None, edge_color='black', node_size=None,
               shiftx=0, seed=0):

    if (pos is None):
        pos = networkx.spring_layout(G, scale=1., seed=seed)

    if shiftx != 0:
        for k, v in pos.items():
            v[0] = v[0] + shiftx

    alpha_edge = 0.7
    width_edge = 1.8
    if Gweights is None:
        networkx.draw_networkx_edges(G, pos, width=width_edge, alpha=alpha_edge, edge_color=edge_color)
    else:
        # We make more visible connections between activated nodes
        n = len(Gweights)
        edgelist_activated = []
        edgelist_deactivated = []
        for i in range(n):
            for j in range(n):
                if Gweights[i] * Gweights[j] * C[i, j] > 0:
                    edgelist_activated.append((i, j))
                elif C[i, j] > 0:
                    edgelist_deactivated.append((i, j))

        networkx.draw_networkx_edges(G, pos, edgelist=edgelist_activated,
                                     width=width_edge, alpha=alpha_edge,
                                     edge_color=edge_color)
        networkx.draw_networkx_edges(G, pos, edgelist=edgelist_deactivated,
                                     width=width_edge, alpha=0.1,
                                     edge_color=edge_color)

    if Gweights is None:
        for node, node_color in enumerate(nodes_color_part):
            networkx.draw_networkx_nodes(G, pos, nodelist=[node],
                                         node_size=node_size, alpha=1,
                                         node_color=node_color)
    else:
        scaled_Gweights = Gweights / (0.5 * Gweights.max())
        nodes_size = node_size * scaled_Gweights
        for node, node_color in enumerate(nodes_color_part):
            networkx.draw_networkx_nodes(G, pos, nodelist=[node],
                                         node_size=nodes_size[node], alpha=1,
                                         node_color=node_color)
    return pos


def draw_transp_colored_GW(G1, C1, G2, C2, part_G1, p1, p2, T,
                           pos1=None, pos2=None, shiftx=4, switchx=False,
                           node_size=70, seed_G1=0, seed_G2=0):
    starting_color = 0
    # get graphs partition and their coloring
    part1 = part_G1.copy()
    unique_colors = ['C%s' % (starting_color + i) for i in np.unique(part1)]
    nodes_color_part1 = []
    for cluster in part1:
        nodes_color_part1.append(unique_colors[cluster])

    nodes_color_part2 = []
    # T: getting colors assignment from argmin of columns
    for i in range(len(G2.nodes())):
        j = np.argmax(T[:, i])
        nodes_color_part2.append(nodes_color_part1[j])
    pos1 = draw_graph(G1, C1, nodes_color_part1, Gweights=p1,
                      pos=pos1, node_size=node_size, shiftx=0, seed=seed_G1)
    pos2 = draw_graph(G2, C2, nodes_color_part2, Gweights=p2, pos=pos2,
                      node_size=node_size, shiftx=shiftx, seed=seed_G2)

    for k1, v1 in pos1.items():
        max_Tk1 = np.max(T[k1, :])
        for k2, v2 in pos2.items():
            if (T[k1, k2] > 0):
                plt.plot([pos1[k1][0], pos2[k2][0]],
                        [pos1[k1][1], pos2[k2][1]],
                        '-', lw=0.7, alpha=min(T[k1, k2] / max_Tk1 + 0.1, 1.),
                        color=nodes_color_part1[k1])
    return pos1, pos2


node_size = 40
fontsize = 13
seed_G2 = 0
seed_G3 = 4

plt.figure(2, figsize=(15, 3.5))
plt.clf()

plt.subplot(131)
plt.axis('off')

plt.title('(PPA) FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s' % (
    np.round(fgw_ppa, 3), str(np.round(T_ppa_sparsity, 2)) + ' %',
    np.round(err_ppa, 4), str(np.round(time_ppa, 2)) + ' ms'), fontsize=fontsize)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2, C2, weightedG3, C3, part_G2, p1=T_ppa.sum(1), p2=T_ppa.sum(0),
    T=T_ppa, shiftx=1.5, node_size=node_size, seed_G1=seed_G2, seed_G2=seed_G3)

plt.subplot(132)
plt.axis('off')

plt.title('(PGD) Entropic FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s' % (
    np.round(fgw_pgd, 3), str(np.round(T_pgd_sparsity, 2)) + ' %',
    np.round(err_pgd, 4), str(np.round(time_pgd, 2)) + ' ms'), fontsize=fontsize)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2, C2, weightedG3, C3, part_G2, p1=T_pgd.sum(1), p2=T_pgd.sum(0),
    T=T_pgd, pos1=pos1, pos2=pos2, shiftx=0., node_size=node_size, seed_G1=0, seed_G2=0)

plt.subplot(133)
plt.axis('off')

plt.title('(BAPG) FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s' % (
    np.round(fgw_bapg, 3), str(np.round(T_bapg_sparsity, 2)) + ' %',
    np.round(err_bapg, 4), str(np.round(time_bapg, 2)) + ' ms'), fontsize=fontsize)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2, C2, weightedG3, C3, part_G2, p1=T_bapg.sum(1), p2=T_bapg.sum(0),
    T=T_bapg, pos1=pos1, pos2=pos2, shiftx=0., node_size=node_size, seed_G1=0, seed_G2=0)

plt.tight_layout()

plt.show()