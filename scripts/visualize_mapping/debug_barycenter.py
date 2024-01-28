"""
Debug the log of input/output fgw_barycenter to find out the mapping graph nodes
"""
import torch
import sys
sys.path.append(".")
from src.model.fgw.barycenter import fgw_barycenters
from src.model.fgw_v2.barycenter import fgw_barycenters as fgw_barycenters_v2

debug_dict = torch.load("/home/taing/workspace/2d_3d_molecular/share_dir/debug_barycenter_jan_28.pt")
F_bary, C_bary = fgw_barycenters( N=debug_dict["N"], Ys=debug_dict["Ys"], Cs=debug_dict["Cs"], ps=debug_dict["ps"], lambdas=debug_dict["lambdas"], alpha=0.5, fixed_structure=True, fixed_features=False, epsilon=0.1, p=None, loss_fun='square_loss', max_iter=30, tol=1e-2, rank=None, bapg=False, rho=0.1, numItermax=100, verbose=True, log=False, init_C=debug_dict["Cs"][0], init_X=None, random_state=None)
# list_num = list(range(5, 10, 1))
# for i in range(len(list_num)):
#     max_iter = list_num[i]
#     numItermax = list_num[i]
F_bary, C_bary = fgw_barycenters_v2(N=debug_dict["N"], Ys=debug_dict["Ys"], Cs=debug_dict["Cs"], ps=debug_dict["ps"], lambdas=debug_dict["lambdas"], warmstartT=True, symmetric=True, method='sinkhorn_log',
                                alpha=0.5, solver='PGD', fixed_structure=True, fixed_features=False, epsilon=0.1, p=None, loss_fun='square_loss', max_iter=10, tol=1e-2,
                                numItermax=2, stopThr=1e-2, verbose=True, log=False, init_C=debug_dict["Cs"][0], init_X=None, random_state=None)
print(f"F_bary: {F_bary} | C_bary: {C_bary}")
