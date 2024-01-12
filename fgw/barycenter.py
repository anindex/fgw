import torch

from .bregman import fgw
from .utils import dist, update_feature_matrix, update_square_loss, update_kl_loss


def entropic_fused_gromov_barycenters(
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

    d = Ys[0].shape[1]  # dimension on the node features

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise ValueError('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            torch.seed(seed)
            xalea = torch.randn(N, 2)
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
                       T[s], max_iter, 1e-4, solver=solver, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]

        else:
            res = [fgw(Ms[s], C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, alpha,
                       None, max_iter, 1e-4, solver=solver, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]

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
