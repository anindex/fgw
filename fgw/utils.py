import torch


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):

    if loss_fun == 'square_loss':
        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * torch.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-15)
    else:
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    constC1 = f1(C1) @ p.reshape(-1, 1) @ torch.ones((1, q.shape[-1])).to(q)
    constC2 = torch.ones((p.shape[-1], 1)).to(p) @ q.reshape(1, -1) @ f2(C2).T
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):

    A = - hC1 @ T @ hC2.T
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):

    tens = tensor_product(constC, hC1, hC2, T)
    return torch.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):

    return 2 * tensor_product(constC, hC1, hC2, T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * T[s] @ Cs[s] @ T[s].T for s in range(len(T))
    ])

    ppt = torch.outer(p, p)
    return tmpsum / ppt


def batch_update_square_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 14 in [12]
    tmpsum = torch.einsum('sij,sjk->sik', T, Cs)
    tmpsum = torch.einsum('sij,sjk->sik', tmpsum, T.transpose(1, 2))
    tmpsum = (lambdas[:, None, None] * tmpsum).sum(dim=0)

    ppt = torch.outer(p, p)
    return tmpsum / ppt


def update_kl_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 15 in [12]
    tmpsum = sum([
        lambdas[s] * (T[s] @ torch.log(torch.clamp(Cs[s], min=1e-15)) @ T[s].T) for s in range(len(T))
    ])

    ppt = torch.outer(p, p)
    return torch.exp(tmpsum / ppt)


def batch_update_kl_loss(p, lambdas, T, Cs):

    # Correct order mistake in Equation 15 in [12]
    tmpsum = torch.einsum('sij,sjk->sik', T, torch.log(torch.clamp(Cs, min=1e-15)))
    tmpsum = torch.einsum('sij,sjk->sik', tmpsum, T.transpose(1, 2))
    tmpsum = (lambdas[:, None, None] * tmpsum).sum(dim=0)

    ppt = torch.outer(p, p)
    return torch.exp(tmpsum / ppt)


def update_feature_matrix(lambdas, Ys, Ts, p):

    p = 1. / p

    tmpsum = sum([
        lambdas[s] * (Ys[s] @ Ts[s].T) * p[None, :]
        for s in range(len(Ts))
    ])

    return tmpsum


def batch_update_feature_matrix(lambdas, Ys, Ts, p):
    tmpsum = lambdas[:, None, None] * torch.einsum('sij,sjk->sik', Ts, Ys)
    tmpsum = tmpsum.sum(dim=0) / p[:, None]

    return tmpsum


def init_matrix_semirelaxed(C1, C2, p, loss_fun='square_loss'):

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * torch.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-15)
    else:
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    constC = f1(C1) @ p.reshape(-1, 1) @ torch.ones((1, C2.shape[-1])).to(C2)

    hC1 = h1(C1)
    hC2 = h2(C2)
    fC2t = f2(C2).T
    return constC, hC1, hC2, fC2t


def dist(x1, x2=None, metric='sqeuclidean', p=2, w=None):

    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    else:
        # TODO: implement other metric with torch
        raise NotImplementedError()


def euclidean_distances(X, Y, squared=False):

    a2 = torch.einsum('ij,ij->i', X, X)
    b2 = torch.einsum('ij,ij->i', Y, Y)

    c = -2 * (X @ Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = torch.clamp(c, min=0)

    if not squared:
        c = torch.sqrt(c)

    if X is Y:
        c = c * (1 - torch.eye(X.shape[0]).to(c))

    return c




class Epsilon:
    """Epsilon scheduler for Sinkhorn and Sinkhorn Step."""

    def __init__(
        self,
        target: float = None,
        scale_epsilon: float = None,
        init: float = 1.0,
        decay: float = 1.0
    ):
        self._target_init = target
        self._scale_epsilon = scale_epsilon
        self._init = init
        self._decay = decay

    @property
    def target(self) -> float:
        """Return the final regularizer value of scheduler."""
        target = 5e-2 if self._target_init is None else self._target_init
        scale = 1.0 if self._scale_epsilon is None else self._scale_epsilon
        return scale * target

    def at(self, iteration: int = 1) -> float:
        """Return (intermediate) regularizer value at a given iteration."""
        if iteration is None:
            return self.target
        # check the decay is smaller than 1.0.
        decay = min(self._decay, 1.0)
        # the multiple is either 1.0 or a larger init value that is decayed.
        multiple = max(self._init * (decay ** iteration), 1.0)
        return multiple * self.target

    def done(self, eps: float) -> bool:
        """Return whether the scheduler is done at a given value."""
        return eps == self.target

    def done_at(self, iteration: int) -> bool:
        """Return whether the scheduler is done at a given iteration."""
        return self.done(self.at(iteration))
    