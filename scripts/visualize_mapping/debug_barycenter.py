"""
Debug the log of input/output fgw_barycenter to find out the mapping graph nodes
"""
import torch
import sys
sys.path.append(".")
from fgw.barycenter import fgw_barycenters, fgw_barycenters_BAPG
import networkx
import numpy as np
import matplotlib.pyplot as plt
import ot


def draw_graph(G, C, Gweights=None,
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

    return pos


def draw_transp_colored_GW(G1, C1, G2, C2, p1, p2, T,
                           pos1=None, pos2=None, shiftx=4,
                           node_size=70, seed_G1=0, seed_G2=0):

    pos1 = draw_graph(G1, C1, Gweights=p1,
                      pos=pos1, node_size=node_size, shiftx=0, seed=seed_G1)
    pos2 = draw_graph(G2, C2, Gweights=p2, pos=pos2,
                      node_size=node_size, shiftx=shiftx, seed=seed_G2)

    # for k1, v1 in pos1.items():
    #     max_Tk1 = np.max(T[k1, :])
    #     for k2, v2 in pos2.items():
    #         if (T[k1, k2] > 0):
    #             plt.plot([pos1[k1][0], pos2[k2][0]],
    #                     [pos1[k1][1], pos2[k2][1]],
    #                     '-', lw=0.7, alpha=min(T[k1, k2] / max_Tk1 + 0.1, 1.))
    return pos1, pos2

debug_dict = torch.load("./scripts/visualize_mapping/debug_barycenter_jan_28.pt")

Gs = []
for i in range(len(debug_dict["Cs"])):
    Gs.append(networkx.Graph(debug_dict["Cs"][i].cpu().numpy()))

max_node = 70
Ys = [Y[:max_node, :max_node] for Y in debug_dict["Ys"]] 
Cs = [C[:max_node, :max_node] for C in debug_dict["Cs"]]
ps = [p[:max_node] for p in debug_dict["ps"]]
ps = [p / p.sum() for p in ps]

F_bary, C_bary, log = fgw_barycenters(N=debug_dict['N'], Ys=Ys, Cs=Cs, ps=ps, lambdas=debug_dict["lambdas"], warmstartT=True, symmetric=False, method='sinkhorn_log',
                                alpha=0.5, solver='PGD', fixed_structure=True, fixed_features=False, epsilon=0.025, p=None, loss_fun='square_loss', max_iter=30, tol=1e-6,
                                numItermax=30, stopThr=5e-3, verbose=True, log=True, init_C=Cs[0], init_X=None, random_state=None)

# F_bary, C_bary, log = ot.gromov.fgw_barycenters(N=debug_dict["N"], Ys=debug_dict["Ys"], Cs=debug_dict["Cs"], ps=debug_dict["ps"], lambdas=debug_dict["lambdas"], warmstartT=True, symmetric=True,
#                                 alpha=0.5, fixed_structure=True, fixed_features=False, p=None, loss_fun='square_loss', max_iter=20, tol=1e-3,
#                                 verbose=True, log=True, init_C=debug_dict["Cs"][1], init_X=None, random_state=None)

# F_bary, C_bary, log = fgw_barycenters_BAPG(N=debug_dict['N'], Ys=Ys, Cs=Cs, ps=ps, lambdas=debug_dict["lambdas"], warmstartT=True, 
#                                         alpha=0.5, fixed_structure=False, fixed_features=False, epsilon=0.001, p=None, loss_fun='kl_loss', max_iter=50, tol=1e-5, rho=5,
#                                         verbose=True, log=True, init_C=Cs[0], init_X=None, random_state=None)

C_bary = C_bary.cpu().numpy()
G_bary = networkx.Graph(C_bary)
T = [t.cpu().numpy() for t in log["T"]]
p_bary = log["p"].cpu().numpy()
G = Gs[0]
Cs = debug_dict["Cs"][0].cpu().numpy()
ps = debug_dict["ps"][0].cpu().numpy()

# visualize the mapping
# pos1, pos2 = draw_transp_colored_GW(G, Cs, G_bary, C_bary, ps, p_bary, T[0], node_size=50)
# plt.show()

# visualize 5 couplings
plt.figure(figsize=(20, 10))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    # draw only max values
    # T_max = np.zeros_like(T[i])
    # T_max[np.arange(T[i].shape[0]), np.argmax(T[i], axis=1)] = 1
    T_max = T[i]
    plt.imshow(T_max)
plt.colorbar()
plt.show()

