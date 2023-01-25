import torch
from torch.nn import functional as F

EPS = 1.e-6


def pdiff(x):
    return x[:, None, ...] - x[None, :, ...]


def _compute_collision_w(p_ij, v_ij, params):
    beta = params['beta']
    v_ij_n = v_ij / (v_ij.norm(dim=-1, keepdim=True) + EPS)
    p_ij_n = p_ij / (p_ij.norm(dim=-1, keepdim=True) + EPS)
    wf = (0.5 * (1 - (v_ij_n * p_ij_n).sum(dim=-1))) ** beta
    res = wf
    return res


def collision_term(p_i, v_i, params, learnable_hparams=None):
    if p_i.shape[0] == 1:
        return torch.tensor(1.0).to(p_i.device)
    if learnable_hparams is not None:
        sigma_d = torch.abs(learnable_hparams['sigma_d']) + params.get('sigma_d_min', 0)
    else:
        sigma_d = params['sigma_d']
    use_w = params.get('use_w', True)
    loss_reduce = params.get('loss_reduce', 'sum')
    ind = torch.triu_indices(p_i.shape[0], p_i.shape[0], offset=1)
    p_ij = pdiff(p_i)[ind[0], ind[1]]
    v_ij = pdiff(v_i)[ind[0], ind[1]]
    # diff = p_ij.norm(dim=-1)[ind[0], ind[1]] - F.pdist(p_i)
    if use_w:
        w = _compute_collision_w(p_ij, v_ij, params)
    else:
        w = 1.0
    energy = torch.exp(-0.5 * p_ij.norm(dim=-1)**2 / sigma_d**2)
    col = w * energy
    loss = col.sum() if loss_reduce == 'sum' else col.mean()
    return loss


def compute_grad_feature(state, params, learnable_hparams=None):
    with torch.enable_grad():
        state.requires_grad_(True)
        p_i, v_i = state[..., :2], state[..., 2:]
        col = collision_term(p_i, v_i, params, learnable_hparams)
        if col.requires_grad:
            grad = torch.autograd.grad(col, state)[0]
        else:
            grad = torch.zeros_like(state)
    return grad