import torch
from fgw.barycenter import fgw_barycenters, fgw_barycenters_BAPG
import time
debug_dict = torch.load("./scripts/debug_barycenter.pt")
print(debug_dict['N'])
print(debug_dict['Ys'].min(), debug_dict['Ys'].max())
print(debug_dict['Cs'][0].min(), debug_dict['Cs'][0].max())

def normalize(C):
    m, M = C.min(), C.max()
    return (C - m) / (M - m)

# normalized
# Cs = [normalize(C) for C in debug_dict['Cs']]  # NOTE: avoid numerical issues
# Ys = [normalize(Y) for Y in debug_dict['Ys']]  # NOTE: avoid numerical issues # TODO: check if this makes sense data-wise in your experiments
# Ys = normalize(debug_dict['Ys'])
# Ys = debug_dict['Ys'].to(torch.float64)
# Cs = [C.to(torch.float64) for C in debug_dict['Cs']]
# ps = [p.to(torch.float64) for p in debug_dict['ps']]
# lambdas = debug_dict["lambdas"].to(torch.float64)
N = debug_dict["N"]
Ys = debug_dict['Ys']
Cs = debug_dict['Cs']
ps = debug_dict['ps']
lambdas = debug_dict["lambdas"]
p = torch.ones(N) / N
p = p.to(Cs[0].device)
for C in Cs:
    C.requires_grad = True
for pt in ps:
    pt.requires_grad = True
for Y in Ys:
    Y.requires_grad = True
p.requires_grad = True
start = time.time()
F_bary, C_bary, log = fgw_barycenters(N=N, Ys=Ys, Cs=Cs, ps=ps, p=p, lambdas=lambdas, warmstartT=True, symmetric=True, method='sinkhorn_log',
                                 alpha=0.5, solver='PGD', fixed_structure=False, fixed_features=False, epsilon=0.15, loss_fun='square_loss', max_iter=30, tol=1e-2, 
                                 numItermax=20, stopThr=1e-2, verbose=True, log=True, init_C=Cs[0], init_X=None, random_state=None)
print("FGW Sinkhorn Time elapsed: ", time.time() - start)
# print(F_bary)
# print(C_bary)
# print(log['T'])
# F_bary, C_bary = fgw_barycenters_BAPG(N=debug_dict["N"], Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, warmstartT=True,
#                                       alpha=0.5, fixed_structure=True, fixed_features=False, epsilon=0.05, p=None, loss_fun='square_loss', max_iter=30, tol=1e-2, rho=1, init_C=Cs[0],
#                                       verbose=True
#                                       )

# print(F_bary.min(), F_bary.max())
