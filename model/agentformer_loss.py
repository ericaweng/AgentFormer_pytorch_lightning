import torch
from .sfm import collision_term


def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum() 
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_recon_sfm(data, cfg, learnable_hparams=None):
    sfm_params = cfg.get('sfm_params', data['cfg'].sfm_params)
    pred = data['train_dec_motion']
    pre_motion_orig = data['pre_motion'].transpose(0, 1)
    vel_pred = pred - torch.cat([pre_motion_orig[:, [-1]], pred[:, :-1]], dim=1)
    loss_unweighted = 0
    for i in range(pred.shape[1]):
        pos = pred[:, i]
        vel = vel_pred[:, i]
        col = collision_term(pos, vel, sfm_params, learnable_hparams)
        loss_unweighted += col
    loss_unweighted /= pred.shape[1]
    if sfm_params.get('learnable_hparams', False):
        loss = loss_unweighted * torch.abs(learnable_hparams['recon_weight']) - torch.abs(learnable_hparams['recon_weight']) / 2#** cfg['bias']
    else:
        loss = loss_unweighted * cfg['weight'] #- cfg.get('bias', 0)
    return loss, loss_unweighted


def compute_sample_sfm(data, cfg, learnable_hparams=None):
    sfm_params = cfg.get('sfm_params', data['cfg'].sfm_params)
    pred = data['infer_dec_motion']
    sample_num = pred.shape[1]
    pre_motion_orig = data['pre_motion'].transpose(0, 1).unsqueeze(1).repeat((1, sample_num, 1, 1))
    vel_pred = pred - torch.cat([pre_motion_orig[:, :, [-1]], pred[:, :, :-1]], dim=2)
    loss_unweighted = 0
    for i in range(pred.shape[2]):
        pos = pred[:, :, i]
        vel = vel_pred[:, :, i]
        col = collision_term(pos, vel, sfm_params, learnable_hparams)
        loss_unweighted += col
    loss_unweighted /= pred.shape[2]
    if sfm_params.get('learnable_hparams', False):
        loss = loss_unweighted * torch.abs(learnable_hparams['sample_weight']) - torch.abs(learnable_hparams['sample_weight']) / 2
    else:
        loss = loss_unweighted * cfg['weight'] #- cfg.get('bias', 0)
    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss,
    'recon_sfm': compute_recon_sfm,
    'sample_sfm': compute_sample_sfm
}