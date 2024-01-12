import torch

from .bregman import fgw
from .utils import dist, update_feature_matrix, update_square_loss, update_kl_loss


def fgw_barycenters(
        N, Ys, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        epsilon=0.1, symmetric=True, alpha=0.5, max_iter=1000, tol=1e-9,
        solver='BAPG', stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, init_Y=None, fixed_structure=False,
        fixed_features=False, seed=0, **kwargs):
    
    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    if stop_criterion not in ['barycenter', 'loss']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")
    
    if solver not in ['PGD', 'PPA', 'BAPG']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA', 'BAPG']." % solver)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S
    
    if p is None:
        p = torch.ones(N).to(Cs[0]) / N

    d = Ys[0].shape[1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            torch.manual_seed(seed)
            xalea = torch.randn(N, 2).to(Cs[0])
            C = dist(xalea, xalea)
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise ValueError('If Y is fixed it must be initialized')
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = torch.zeros((N, d)).to(ps[0])

        else:
            Y = init_Y

    Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

    if warmstartT:
        T = [None] * S

    cpt = 0

    if stop_criterion == 'barycenter':
        inner_log = False
        err_feature = 1e15
        err_structure = 1e15
        err_rel_loss = 0.

    else:
        inner_log = True
        err_feature = 0.
        err_structure = 0.
        curr_loss = 1e15
        err_rel_loss = 1e15

    if log:
        log_ = {}
        if stop_criterion == 'barycenter':
            log_['err_feature'] = []
            log_['err_structure'] = []
            log_['Ts_iter'] = []
        else:
            log_['loss'] = []
            log_['err_rel_loss'] = []

    while ((err_feature > tol or err_structure > tol or err_rel_loss > tol) and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev = C
            Yprev = Y
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [fgw(Ms[s], C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, alpha,
                       T[s], max_iter, 1e-4, solver=solver, verbose=False, log=inner_log, **kwargs) for s in range(S)]

        else:
            res = [fgw(Ms[s], C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, alpha,
                       None, max_iter, 1e-4, solver=solver, verbose=False, log=inner_log, **kwargs) for s in range(S)]

        if stop_criterion == 'barycenter':
            T = res
        else:
            T = [output[0] for output in res]
            curr_loss = torch.sum([output[1]['fgw_dist'] for output in res])

        # update barycenters
        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p).T
            Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == 'square_loss':
                C = update_square_loss(p, lambdas, T, Cs)

            elif loss_fun == 'kl_loss':
                C = update_kl_loss(p, lambdas, T, Cs)

        # update convergence criterion
        if stop_criterion == 'barycenter':
            err_feature, err_structure = 0., 0.
            if not fixed_features:
                err_feature = torch.norm(Y - Yprev)
            if not fixed_structure:
                err_structure = torch.norm(C - Cprev)
            if log:
                log_['err_feature'].append(err_feature)
                log_['err_structure'].append(err_structure)
                log_['Ts_iter'].append(T)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err_structure))
                print('{:5d}|{:8e}|'.format(cpt, err_feature))
        else:
            err_rel_loss = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0. else torch.nan
            if log:
                log_['loss'].append(curr_loss)
                log_['err_rel_loss'].append(err_rel_loss)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err_rel_loss))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p
        log_['Ms'] = Ms

        return Y, C, log_
    else:
        return Y, C


# def fused_ACC_torch(M, A, B, a=None, b=None, X=None, alpha=0, epoch=2000, eps=1e-5, rho=1e-1):
#     if a is None:
#         a = torch.full((A.shape[0],), 1.0 / A.shape[0]).to(A)
#     else:
#         a = a[:, None].to(A)
        
#     if b is None:
#         b = torch.full((B.shape[0],), 1.0 / B.shape[0]).to(B)
#     else:
#         b = b[:, None].to(B)
    
#     if X is None:
#         X = a @ b.T
#     obj_list = []
#     for ii in range(epoch):
#         X = X + 1e-10
#         X_prev = X
#         grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
#         X = torch.exp(grad / rho)*X
#         X = X * (a / (X @  torch.ones_like(b)))
#         grad = 4 * alpha * A @ X @ B - (1 - alpha) * M
#         X = torch.exp(grad / rho) * X
#         X = X * (b.T / (X.T @ torch.ones_like(a)).T)
#         if ii > 0 and ii % 10 == 0:
#             objective = torch.trace(((1 - alpha) * M - 2 * alpha * A @ X @ B) @ X.T)
#             if len(obj_list) > 0 and torch.abs((objective - obj_list[-1]) / obj_list[-1]) < eps:
#                 # print('iter:{}, smaller than eps'.format(ii))
#                 break
#             obj_list.append(objective)
#     return X, obj_list


# def fgw_barycenters_other(
#     N, Ys, Cs, ps, lambdas, alpha, fixed_structure=False, fixed_features=False,
#     p=None, loss_fun='square_loss', max_iter=100, tol=1e-9, rank=None, bapg=False, rho=0.1,
#     verbose=False, log=False, init_C=None, init_X=None, seed=0
# ):
#     '''https://github.com/ArthurLeoM/FGWMixup/blob/main/src/FGW_barycenter.py'''
#     S = len(Cs)
#     d = Ys[0].shape[1]  # dimension on the node features
#     if p is None:
#         p = torch.ones(N).to(Cs[0]) / N

#     if fixed_structure:
#         if init_C is None:
#             raise ValueError('If C is fixed it must be initialized')
#         else:
#             C = init_C
#     else:
#         if init_C is None:
#             torch.manual_seed(seed)
#             xalea = torch.randn(N, 2).to(Cs[0])
#             C = dist(xalea, xalea)
#             C /= C.max()
#             C = torch.from_numpy(C).to(p)
#         else:
#             C = init_C

#     if fixed_features:
#         if init_X is None:
#             raise ValueError('If X is fixed it must be initialized')
#         else:
#             X = init_X
#     else:
#         if init_X is None:
#             X = torch.zeros((N, d)).to(p)
#         else:
#             X = init_X

#     T = [torch.outer(p, q) for q in ps]

#     Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

#     cpt = 0
#     err_feature = 1
#     err_structure = 1

#     if log:
#         log_ = {}
#         log_['err_feature'] = []
#         log_['err_structure'] = []
#         log_['Ts_iter'] = []
#         log_['dists_iter'] = []

#     while((err_feature > tol or err_structure > tol) and cpt < max_iter):
#         Cprev = C
#         Xprev = X

#         if not fixed_features:
#             Ys_temp = [y.T for y in Ys]
#             X = update_feature_matrix(lambdas, Ys_temp, T, p).T

#         Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

#         if not fixed_structure:
#             if loss_fun == 'square_loss':
#                 C = update_square_loss(p, lambdas, T, Cs)

#             elif loss_fun == 'kl_loss':
#                 C = update_kl_loss(p, lambdas, T, Cs)

#         # if rank is not None:
#         #     T = []
#         #     dists = []
#         #     for s in range(S):
#         #         cur_dist, cur_T = entropic_low_rank_fgw(Ms[s], C, Cs[s], p, ps[s], alpha, 
#         #                             rank=rank, gamma=100, reg=0, max_iter=max_iter, tol=1e-7, random_state=random_state)
#         #         T.append(cur_T)
#         #         dists.append(cur_dist)
        
#         if bapg:
#             T = []
#             dists = []
#             for s in range(S):
#                 cur_T, cur_dist = fused_ACC_numpy(Ms[s], C, Cs[s], p, ps[s], alpha=alpha, epoch=300, eps=1e-5, rho=rho)
#                 T.append(cur_T)
#                 c1 = np.dot(C*C, np.outer(p, np.ones_like(ps[s]))) + np.dot(np.outer(np.ones_like(p), ps[s]), Cs[s]*Cs[s])
#                 res = np.trace(np.dot(c1.T, cur_T))
#                 dists.append(cur_dist[-1] + alpha * res)
                
#         else:
#             T = []
#             dists = []
#             for s in range(S):
#                 cur_T, cur_log = fused_gromov_wasserstein(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
#                                         numItermax=300, stopThr=1e-5, verbose=False, log=True)
#                 T.append(cur_T)
#                 dists.append(cur_log['fgw_dist'])
#             # dists = [fused_gromov_wasserstein2(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
#             #                             numItermax=max_iter, stopThr=1e-5, verbose=False) for s in range(S)]
#         # print(f'Solve FGW time @ iter {cpt}: {time.time() - time_start}')

#         # T is N,ns
#         err_feature = nx.norm(X - nx.reshape(Xprev, (N, d))) / nx.norm(nx.reshape(Xprev, (N, d)))
#         err_structure = nx.norm(C - Cprev) / nx.norm(Cprev)
#         # print(err_feature, err_structure)
#         if log:
#             log_['err_feature'].append(err_feature)
#             log_['err_structure'].append(err_structure)
#             log_['Ts_iter'].append(T)
#             log_['dists_iter'].append(dists)

#         if verbose:
#             if cpt % 200 == 0:
#                 print('{:5s}|{:12s}'.format(
#                     'It.', 'Err') + '\n' + '-' * 19)
#             # if cpt % 10 == 0:
#             #     print('Matrix C: ', C)
#             #     print('Matrix C non-zeros: {:.4f}, entry_sum: {:.4f}'.format(np.sum(C!=0), np.sum(np.abs(C))))
#             #     print('Matrix Cprev: ', Cprev)
#             #     print('Matrix Delta: ', C - Cprev)
#             print('{:5d}|{:8e}|'.format(cpt, err_structure))
#             print('{:5d}|{:8e}|'.format(cpt, err_feature))

#         cpt += 1

#     all_time = time.time() - time_start
#     print(f'----Avg Solve FGW time @ iter {cpt}: {all_time / cpt}')
#     # print(C)
#     if log:
#         log_['T'] = T  # from target to Ys
#         log_['p'] = p
#         log_['Ms'] = Ms
#         log_['dists'] = dists

#     if log:
#         return X, C, log_, cpt, all_time
#     else:
#         return X, C, cpt, all_time