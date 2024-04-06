import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from PEACE_ALG.sqrtm import sqrtm


def gamma_tb(mathcal_Z, iters=50):
    d = mathcal_Z.shape[1]
    x = nn.Parameter(torch.zeros(d))
    optim = Adam([x], lr=1e-2)
    l = torch.softmax(x, dim=-1)
    X = torch.eye(d).clone().detach().float()
    outers = torch.bmm(X.unsqueeze(2), X.unsqueeze(1))

    for i in range(iters):
        A = torch.sum(l.view(-1, 1, 1) * outers, dim=0).requires_grad_()
        # draw eta
        eta = np.random.randn(d)
        # compute z that attains max
        A_sqrt_inv = sqrtm(torch.inverse(A).requires_grad_()).requires_grad_()
        _, max_z = gamma_est(mathcal_Z, A_sqrt_inv.clone().detach().numpy(), eta)
        loss = (A_sqrt_inv @ torch.tensor(max_z).float()) @ torch.tensor(eta).float()
        loss.backward()
        optim.step()
        optim.zero_grad()
        l = torch.softmax(x, dim=-1)
    ldetach = l.detach_().numpy()
    A = torch.sum(l.view(-1, 1, 1) * outers, dim=0)
    A_sqrt_inv = sqrtm(torch.inverse(A)).detach().numpy()
    total = 0
    iters = iters
    for _ in range(iters):
        eta = np.random.randn(d)
        val, _ = gamma_est(mathcal_Z, A_sqrt_inv, eta)
        total += val
    return ldetach, (total / iters) ** 2


def gamma_est(Z, A_sqrt_inv, eta):
    scores = Z @ A_sqrt_inv @ eta
    idx = np.argmax(scores)
    z_1 = Z[idx, :]
    return scores[idx], z_1


def _rounding(design, num_samples):
    """
    Routine to convert design to allocation over num_samples following rounding procedures in Pukelsheim.
    """
    num_support = (design > 0).sum()
    support_idx = np.where(design > 0)[0]
    support = design[support_idx]
    n_round = np.ceil((num_samples - 0.5 * num_support) * support)
    while n_round.sum() - num_samples != 0:
        if n_round.sum() < num_samples:
            idx = np.argmin(n_round / support)
            n_round[idx] += 1
        else:
            idx = np.argmax((n_round - 1) / support)
            n_round[idx] -= 1

    allocation = np.zeros(len(design))
    allocation[support_idx] = n_round
    return allocation.astype(int)
