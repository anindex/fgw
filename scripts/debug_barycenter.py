import torch
from fgw.barycenter import fgw_barycenters
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
Ys = debug_dict['Ys'].to(torch.float64)
Cs = [C.to(torch.float64) for C in debug_dict['Cs']]
ps = [p.to(torch.float64) for p in debug_dict['ps']]
lambdas = debug_dict["lambdas"].to(torch.float64)

F_bary, C_bary = fgw_barycenters(N=debug_dict["N"], Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, warmstartT=True, symmetric=True,
                                 alpha=0.5, solver='PGD', fixed_structure=True, fixed_features=False, epsilon=0.15, p=None, loss_fun='square_loss', max_iter=30, tol=1e-2, 
                                 numItermax=20, stopThr=1e-2, verbose=True, log=False, init_C=Cs[0], init_X=None, random_state=None)

print(F_bary.min(), F_bary.max())