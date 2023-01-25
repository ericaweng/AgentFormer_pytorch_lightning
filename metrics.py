import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from functools import partial


def point_to_segment_dist_old(x1, y1, x2, y2, p1, p2):
    """
    Calculate the closest distance between start(p1, p2) and a line segment with two endpoints (x1, y1), (x2, y2)
    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((p1 - x1, p2 - y1), axis=-1)

    u = ((p1 - x1) * px + (p2 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest start to (p1, p2) on the line segment
    x = x1 + u * px
    y = y1 + u * py
    return np.linalg.norm((x - p1, y - p2), axis=-1)


def get_collisions_mat_old_torch(pred_traj_fake, threshold):
    """threshold: radius + discomfort distance of agents"""
    pred_traj_fake = pred_traj_fake.transpose(1, 0)
    ts, num_peds, _ = pred_traj_fake.shape
    collision_mat = torch.full((ts, num_peds, num_peds), False)
    collision_mat_vals = torch.full((ts, num_peds, num_peds), np.inf)
    # test initial timesteps
    for ped_i, x_i in enumerate(pred_traj_fake[0]):
        for ped_j, x_j in enumerate(pred_traj_fake[0]):
            if ped_i == ped_j:
                continue
            closest_dist = torch.norm(x_i - x_j) - threshold * 2
            if closest_dist < 0:
                collision_mat[0, ped_i, ped_j] = True
            collision_mat_vals[0, ped_i, ped_j] = closest_dist

    # test t-1 later timesteps
    for t in range(ts - 1):
        for ped_i, ((ped_ix, ped_iy), (ped_ix1, ped_iy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
            for ped_j, ((ped_jx, ped_jy), (ped_jx1, ped_jy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
                if ped_i == ped_j:
                    continue
                px = ped_ix - ped_jx
                py = ped_iy - ped_jy
                ex = ped_ix1 - ped_jx1
                ey = ped_iy1 - ped_jy1
                closest_dist = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                if closest_dist < 0:
                    collision_mat[t+1, ped_i, ped_j] = True
                collision_mat_vals[t + 1, ped_i, ped_j] = closest_dist

    return torch.any(torch.any(collision_mat, dim=0), dim=0), collision_mat


def get_collisions_mat_old(pred_traj_fake, threshold):
    """pred_traj_fake: shape (num_peds, num_samples, ts, 2)
    threshold: radius + discomfort distance of agents"""
    pred_traj_fake = pred_traj_fake.transpose(1, 0, 2)
    ts, num_peds, _ = pred_traj_fake.shape
    collision_mat = np.full((ts, num_peds, num_peds), False)
    collision_mat_vals = np.full((ts, num_peds, num_peds), np.inf)
    # test initial timesteps
    for ped_i, x_i in enumerate(pred_traj_fake[0]):
        for ped_j, x_j in enumerate(pred_traj_fake[0]):
            if ped_i == ped_j:
                continue
            closest_dist = np.linalg.norm(x_i - x_j) - threshold * 2
            if closest_dist < 0:
                collision_mat[0, ped_i, ped_j] = True
            collision_mat_vals[0, ped_i, ped_j] = closest_dist

    # test t-1 later timesteps
    for t in range(ts - 1):
        for ped_i, ((ped_ix, ped_iy), (ped_ix1, ped_iy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
            for ped_j, ((ped_jx, ped_jy), (ped_jx1, ped_jy1)) in enumerate(zip(pred_traj_fake[t], pred_traj_fake[t+1])):
                if ped_i == ped_j:
                    continue
                px = ped_ix - ped_jx
                py = ped_iy - ped_jy
                ex = ped_ix1 - ped_jx1
                ey = ped_iy1 - ped_jy1
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist_old(px, py, ex, ey, 0, 0) - threshold * 2
                if closest_dist < 0:
                    collision_mat[t+1, ped_i, ped_j] = True
                collision_mat_vals[t + 1, ped_i, ped_j] = closest_dist

    return np.any(np.any(collision_mat, axis=0), axis=0), collision_mat  # collision_mat_pred_t_bool


def compute_ADE(pred_arr, gt_arr, return_sample_vals=False, **kwargs):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist.mean(axis=-1)  # samples
        ade += dist.min(axis=0)  # (1, )
    ade /= len(pred_arr)
    if return_sample_vals:
        return ade, dist / len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr, return_sample_vals=False, **kwargs):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist[..., -1]  # samples
        fde += dist.min(axis=0)  # (1, )
    fde /= len(pred_arr)
    if return_sample_vals:
        return fde, dist / len(pred_arr)
    return fde


def _lineseg_dist(a, b):
    """
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    """
    # reduce computation
    if np.all(a == b):
        return np.linalg.norm(-a, axis=1)

    # normalized tangent vector
    d = (b - a) / np.linalg.norm(b - a, axis=-1, keepdims=True)

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros_like(t)], axis=0)

    # perpendicular distance component
    c = np.cross(-a, d, axis=-1)

    ans = np.hypot(h, np.abs(c))

    # edge case where agent stays still
    ts_pairs_where_a_eq_b = np.where(np.all(a == b, axis=-1))
    assert np.all(np.all(a == b, axis=-1) == np.isnan(ans))
    ans[ts_pairs_where_a_eq_b] = np.linalg.norm(-a, axis=1)[ts_pairs_where_a_eq_b]

    return ans


def _get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return np.concatenate([np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds - ped_i - 1, 1)) - traj[:, ped_i + 1:]
                           for ped_i in range(num_peds)], axis=1)


def _get_diffs_gt(traj, gt_traj):
    """same order of ped pairs as pdist"""
    num_peds = traj.shape[1]
    return np.stack([
            np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds, 1)) - gt_traj
            for ped_i in range(num_peds)
    ],
            axis=1)


def check_collision_per_sample(sample_idx, sample, gt_arr, ped_radius=0.1):
    """sample: (num_peds, ts, 2) and same for gt_arr"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    gt_arr = gt_arr.transpose(1, 0, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    # pred
    # Get collision for timestep=0
    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)
    # gt
    collision_0_gt = cdist(sample[0], gt_arr[0]) < ped_radius
    np.fill_diagonal(collision_0_gt, False)
    ped_pair_diffs_gt = _get_diffs_gt(sample, gt_arr)
    pxy_gt = ped_pair_diffs_gt[:-1].reshape(-1, 2)
    exy_gt = ped_pair_diffs_gt[1:].reshape(-1, 2)
    collision_t_gt = _lineseg_dist(pxy_gt, exy_gt).reshape(ts - 1, num_peds, num_peds) < ped_radius * 2
    for ped_mat in collision_t_gt:
        np.fill_diagonal(ped_mat, False)
    collision_mat_gt = np.any(collision_t_gt, axis=0) | collision_0_gt
    n_ped_with_col_gt_per_sample = np.any(collision_mat_gt, axis=0)

    return sample_idx, n_ped_with_col_pred_per_sample, n_ped_with_col_gt_per_sample


def compute_CR(pred_arr,
               gt_arr=None,
               aggregation='max',
               return_sample_vals=False,
               return_collision_mat=False,
               collision_rad=None):
    """Compute collision rate and collision-free likelihood.
    Input:
        - pred_arr: (np.array) (n_pedestrian, n_samples, timesteps, 4)
        - gt_arr: (np.array) (n_pedestrian, timesteps, 4)
    Return:
        Collision rates
    """
    # (n_agents, n_samples, timesteps, 4) > (n_samples, n_agents, timesteps 4)

    n_ped, n_sample, _, _ = pred_arr.shape

    col_pred = np.zeros((n_sample))
    col_mats = []
    if n_ped > 1:
        # with nool(processes=multiprocessing.cpu_count() - 1) as pool:
        #     r = pool.starmap(
        #             partial(check_collision_per_sample, gt_arr=gt_arr),
        #             enumerate(pred_arr))
        for sample_idx, pa in enumerate(pred_arr):
            # n_ped_with_col_pred, col_mat = get_collisions_mat_old(pred_arr[:, sample_idx], collision_rad)
            n_ped_with_col_pred, col_mat = check_collision_per_sample_no_gt(pred_arr[:, sample_idx], collision_rad)
            # assert np.all(n_ped_with_col_pred == n_ped_with_col_pred2), f'{n_ped_with_col_pred}\nshould equal\n{n_ped_with_col_pred2}'
            # assert np.all(col_mat2 == col_mat), f'{col_mat} \nshould equal\n{col_mat2}'
            col_mats.append(col_mat)
            col_pred[sample_idx] += (n_ped_with_col_pred.sum())

    if aggregation == 'mean':
        cr_pred = col_pred.mean(axis=0)
    elif aggregation == 'min':
        cr_pred = col_pred.min(axis=0)
    elif aggregation == 'max':
        cr_pred = col_pred.max(axis=0)
    else:
        raise NotImplementedError()

    # Multiply by 100 to make it percentage
    # cr_pred *= 100
    crs = [cr_pred / n_ped]
    if return_sample_vals:
        crs.append(col_pred / n_ped)
    if return_collision_mat:
        crs.append(col_mats)
    return tuple(crs) if len(crs) > 1 else crs[0]


def check_collision_per_sample_no_gt(sample, ped_radius=0.1):
    """sample: (num_peds, ts, 2)"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred_t_bool = np.stack([squareform(cm) for cm in np.concatenate([collision_0_pred[np.newaxis,...], collision_t_pred])])
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)

    return n_ped_with_col_pred_per_sample, collision_mat_pred_t_bool


stats_func = {
            'ADE': compute_ADE,
            'FDE': compute_FDE,
            'CR_pred': partial(compute_CR, aggregation='max'),
            'CR_pred_mean': partial(compute_CR, aggregation='mean'),
}


def main():
    """debug new, fast collision fn... edge case where agent is in same position between two timesteps
    was wrong this whole time"""
    cr = 0.1
    pred_arr = np.zeros((2, 12, 2))
    pred_arr[1] = np.ones((1, 12, 2))
    pred_arr[1, -1] = 0.15 * np.ones(2)
    _, cols_old, col_mats_old = get_collisions_mat_old(None, pred_arr, cr)
    _, cols, col_mats = check_collision_per_sample_no_gt(None, pred_arr, cr)
    print("col_mats_old:", col_mats_old)
    print("col_mats:", col_mats)
    print("cols_old:\n", cols_old)
    print("cols:\n", cols)
    from viz_utils import plot_traj_anim
    save_fn_old = 'viz/old.mp4'
    save_fn = 'viz/new.mp4'
    plot_traj_anim(save_fn=save_fn_old, pred_traj_fake=pred_arr.swapaxes(0,1), collision_mats=col_mats_old, ped_radius=cr)
    plot_traj_anim(save_fn=save_fn, pred_traj_fake=pred_arr.swapaxes(0,1), collision_mats=col_mats, ped_radius=cr)
    assert np.all(cols) == np.all(cols_old)


if __name__ == "__main__":
    main()
