import torch
from fgw.barycenter import fgw_barycenters, fgw_barycenters_BAPG
import ot
debug_dict = torch.load("./scripts/debug_barycenter.pt")

N = debug_dict["N"]
Ys = debug_dict['Ys']
Cs = debug_dict['Cs']
ps = debug_dict['ps']
lambdas = debug_dict["lambdas"]


# =============================================================================
# ground truth

F_bary, C_bary = ot.gromov.fgw_barycenters(
    N=N, Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, loss_fun='square_loss', max_iter=30, tol=1e-2, warmstartT=True, symmetric=True,
    verbose=False, log=False, init_C=Cs[0], init_X=None, random_state=None)



# =============================================================================
# Our code
F_bary1, C_bary2 = fgw_barycenters(N=N, Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, warmstartT=True, symmetric=True, method='sinkhorn_log',
                                 alpha=0.5, solver='PPA', fixed_structure=True, fixed_features=False, epsilon=0.1, p=None, loss_fun='kl_loss', max_iter=20, tol=1e-2, 
                                 numItermax=5, stopThr=1e-2, verbose=False, log=False, init_C=Cs[0], init_X=None, random_state=None)


# =============================================================================
# FGWMixup code

F_bary2, C_bary2 = fgw_barycenters_BAPG(N=N, Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, warmstartT=True, 
                                        alpha=0.5, fixed_structure=True, fixed_features=False, epsilon=0.1, p=None, loss_fun='kl_loss', max_iter=20, tol=1e-2, rho=5,
                                        verbose=True, log=False, init_C=Cs[0], init_X=None, random_state=None)


# compare F_bary and F_bary2
def frob_norm(A, B):
    return torch.norm(A - B, p='fro').item() / sum(A.shape)

print('Frob norm between FGW C++ and our code: ', frob_norm(F_bary, F_bary1))
print('Frob norm between FGW C++ and FGWMixup: ', frob_norm(F_bary, F_bary2))
