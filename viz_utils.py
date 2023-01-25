import numpy as np

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


def plot_fig(save_fn, title=None, plot_size=None, *list_of_arg_dicts):
    if plot_size is None:
        if len(list_of_arg_dicts) > 4:
            num_plots_height = 2
            num_plots_width = int((len(list_of_arg_dicts) + 1)/2)
        else:
            num_plots_height = 1
            num_plots_width = len(list_of_arg_dicts)
    else:
        num_plots_width, num_plots_height = plot_size

    assert num_plots_width * num_plots_height >= len(list_of_arg_dicts), \
        f'plot_size ({plot_size}) must be able to accomodate {len(list_of_arg_dicts)} graphs'
    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(7.5 * num_plots_width, 5 * num_plots_height))
    if isinstance(axes[0], np.ndarray):
        axes = [a for ax in axes for a in ax]
    # fig, axes = plt.subplots(1, len(list_of_arg_dicts), figsize=(10 * len(list_of_arg_dicts), 10))
    anim_graphs = []

    # determine obs_len and pred_len automatically
    # obs_len = list_of_arg_dicts[0]['obs_traj'].shape[0]
    # if list_of_arg_dicts[0]['pred_traj_gt'] is not None:
    #     pred_len = list_of_arg_dicts[0]['pred_traj_gt'].shape[0]
    # elif list_of_arg_dicts[0]['pred_traj_fake'] is not None:
    #     pred_len = list_of_arg_dicts[0]['pred_traj_fake'].shape[0]
    # else:
    #     pred_len = 0
    obs_len = 8
    pred_len = 12

    bounds = []
    for ax_i, (arg_dict,ax) in enumerate(zip(list_of_arg_dicts, axes)):
        ao = AnimObj()
        anim_graphs.append(ao)
        ao.plot_traj_anim(**arg_dict, ax=ax)
        bounds.append(ao.bounds)

    # reset bounds to be equal across all graphs, now that they are calculated
    bounds = np.stack(bounds)
    x_low, y_low = np.min(bounds[:, [0,2]], axis=0)
    x_high, y_high = np.max(bounds[:, [1,3]], axis=0)
    for ax_i, (arg_dict,ax) in enumerate(zip(list_of_arg_dicts, axes)):
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    anim = animation.FuncAnimation(fig, lambda frame_i: [ag.update(frame_i) for ag in anim_graphs],
                                   frames=obs_len + pred_len, interval=500)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    anim.save(save_fn)
    print(f"saved animation to {save_fn}")
    plt.close(fig)


def plot_traj_anim(**kwargs):
    AnimObj().plot_traj_anim(**kwargs)


class AnimObj:

    def __init__(self):
        self.update = None
        self.bounds = None

    def plot_traj_anim(self, obs_traj=None, save_fn=None, ped_radius=0.1, ped_discomfort_dist=0.2, pred_traj_gt=None,
                       pred_traj_fake=None, ped_num_label_on='gt',
                       bounds=None, int_cat_abbv=None, scene_stats=None, cfg_names=None,
                       collision_mats=None, cmap_name='tab10', extend_last_frame=3, show_ped_stats=False,
                       text_time=None, text_fixed=None, grid_values=None, plot_collisions_all=False, plot_title=None,
                       ax=None, update=None, pred_alpha=None):
        """
        obs_traj: shape (8, num_peds, 2) observation input to model, first 8 timesteps of the scene
        save_fn: file name where to save animation
        ped_diameter: collision threshold -- pedestrian radius * 2
        pred_traj_fake: tensor of shape (8 or 12 pred timesteps, num_peds, 2)
                        or tensor of shape (num_samples, 8 or 12 pred timesteps, num_peds, 2)
                        or list of tensors of shape (8 or 12, num_peds, 2)  (where each item are the samples predicted by a different model)
                        or list of tensors of shape (num_samples, 8 or 12 pred timesteps, num_peds, 2)
        bounds: x_low, x_high, y_low, y_high: plotting bounds
                if not specified the min and max bounds of whichever trajectories are present are used
        pred_traj_gt: shape (8 or 12, num_peds, 2) ground-truth trajectory
        interaction_matrix: shape (num_peds, num_peds - 1) specifies which pairs belong to the given int_type.
                            only used for int_types that are pairwise, i.e. "linear" "static" etc. are not relevant.
                            np.sum(interaction_matrix, axis=-1) produces an "interaction level" for each ped, which
                            is used to color it in the plot. the more green a ped is, the greater number of peds it
                            shares that int_type with. the more blue, the fewer.
        int_type_abbv: used for the title and coloring peds
        scene_stats: statistics for each ped in the scene to plot
        collision mats: if already computed, plots when a collision occurs
        cmap_name: which color map to use for coloring pedestrians
        extend_last_frame: how many timesteps to extend the last frame so the viewer can better observe the full trajectory
        scatter_dots: a dict mapping labels to sets of np.ndarray scatter points, or a list of np.ndarray scatter points,
                     or an np.ndarray set of scatter points, to plot
        show_ped_stats: (bool) whether to display statistics for each pedestrian on the plot
        text_time (list): list of strings of len = num_timesteps, of text to plot that changes each timestep
        grid_values (np.array): colored grid to plot, for debugging purposes
        plot_collisions_all: if True, and collision_mats is specified, plots obs step and pred step collisions
                             o/w: plots only pred step fake collisions
        """
        assert not all([obs_traj is None, pred_traj_fake is None, pred_traj_gt is None]), "at least one of obs_traj, pred_traj_fake, or pred_traj_gt must be supplied"

        # instantiate ax if not exist
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = None
        plot_title = f"{plot_title}\n" if plot_title is not None else ""
        ax.set_title(plot_title, fontsize=16)
        # ax.set_title(f"{plot_title}{save_fn}\ninteraction_type: {int_cat_abbv}")
        ax.set_aspect("equal")

        # make pred_traj_fake standard shape
        if pred_traj_fake is not None:
            if isinstance(pred_traj_fake, np.ndarray):
                if len(pred_traj_fake.shape) == 3:
                    pred_traj_fake = pred_traj_fake[np.newaxis]  # make number of samples = 1
                else:
                    assert len(pred_traj_fake.shape) == 4
                pred_traj_fake = [pred_traj_fake]  # make number of models = 1
            else:
                assert isinstance(pred_traj_fake, list)
                if len(pred_traj_fake[0].shape) == 3:
                    pred_traj_fake = [ptf[np.newaxis] for ptf in pred_traj_fake]  # make number of samples = 1
                else:
                    assert len(pred_traj_fake[0].shape) == 4
            assert isinstance(pred_traj_fake, list) and isinstance(pred_traj_fake[0], np.ndarray)
            assert len(pred_traj_fake[0].shape) == 4

        # obs len
        if obs_traj is not None:
            obs_len = obs_traj.shape[0]
        else:
            obs_len = 0
        # pred len
        if pred_traj_gt is not None:
            pred_len = pred_traj_gt.shape[0]
        elif pred_traj_fake is not None:
            pred_len = pred_traj_fake[0].shape[1]
        else:
            pred_len = 0
        # num_peds
        if obs_traj is not None:
            num_peds = obs_traj.shape[1]
        elif pred_traj_gt is not None:
            num_peds = pred_traj_gt.shape[1]
        elif pred_traj_fake is not None:
            num_peds = pred_traj_fake[0].shape[2]
        else:
            raise RuntimeError

        # calculate bounds automatically
        if bounds is None:
            all_traj = np.zeros((0, 2))
            if obs_traj is not None:
                all_traj = obs_traj.reshape(-1, 2)
            if pred_traj_gt is not None:
                all_traj = np.concatenate([all_traj, pred_traj_gt.reshape(-1, 2)])
            if pred_traj_fake is not None:
                all_traj = np.concatenate([all_traj, *[p.reshape(-1, 2) for ptf in pred_traj_fake for p in ptf]])
            x_low, x_high = np.min(all_traj[:, 0]) - ped_radius, np.max(all_traj[:, 0]) + ped_radius
            y_low, y_high = np.min(all_traj[:, 1]) - ped_radius, np.max(all_traj[:, 1]) + ped_radius
        else:  # set bounds as specified
            x_low, x_high, y_low, y_high = bounds
        self.bounds = x_low, x_high, y_low, y_high
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

        # color and style properties
        delta = .32  # ped stats text offset
        text_offset_x = 0.2
        text_offset_y = 0.2
        obs_alpha = 1  # how much alpha to plot obs traj
        if pred_alpha is None:
            pred_alpha = 0.7  # how much alpha to plot gt traj, if they exist
        # each sample a different marker
        markers_0 = [None] * 10#['o', '*', '^', 's', '1', 'P', 'x', '$\#$', ',', '$\clubsuit$'] #'v', '<', ',', ]
        markers_1 = [None] * 10#['P', 'x', '$\#$', ',', '$\clubsuit$'] #'v', '<', ',', ]
        # each ped a different color
        cmap_real = plt.get_cmap(cmap_name, max(10, num_peds))
        cmap_fake = plt.get_cmap(cmap_name, max(10, num_peds))
        color_fake = [['#0D47A1', '#2196F3'],  # blue
                      ['#E65100', '#FF9800'],  # orange
                      ['#194D33', '#4CAF50'],  # green
                      ['#B71C1C', '#F44336'],  # red
                      ['#4A148C', '#9C27B0'],  # purple
                      ['#312502', '#795548'],  # brown
                      ['#b31658', '#E91E63'],  # pink
                      ['#333333', '#999999'],  # gray
                      ['#F0F4C3', '#AFB42B'],  # olive
                      ]
        # each model a different alpha and linestyle
        alpha_min = 0.3
        linestyles = ['dotted', '--']

        # add scene-related stats as descriptive text
        if show_ped_stats:
            if scene_stats is not None:
                values = map(lambda x: f"{x:0.2f}", scene_stats.values())
                scene_stats_text = f'{" / ".join(map(str, scene_stats.keys()))}\n{" / ".join(values)}'
                ax.add_artist(plt.text(x_low + 0.1, y_high + .2, scene_stats_text, fontsize=8))
                # ax.add_artist(plt.text(x_low + 0.1, y_high - .3, 'obs // pred (avg_speed / std_speed / smoothness)', fontsize=8))

        # ## text that changes each frame
        if text_time is not None:
            text_over_time = ax.text(14, 6, "", fontsize=10, color='k', weight='bold')
            ax.add_artist(text_over_time)

        ## text that stays fixed each frame
        offset_lower = 0.1
        text_fixed_fs = 16
        if isinstance(text_fixed, str):
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text_fixed, fontsize=text_fixed_fs))
        elif isinstance(text_fixed, list):
            text = "\n".join(text_fixed)
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=text_fixed_fs))
        elif isinstance(text_fixed, dict):
            text = "\n".join([f'{k}: {v:0.3f}' for k, v in text_fixed.items()])
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=text_fixed_fs))
        else:
            if text_fixed is not None:
                raise NotImplementedError("text_fixed is unrecognized format")

        # ped graph elements
        circles_gt, circles_fake, last_obs_circles, lines_pred_gt, \
        lines_obs_gt, lines_pred_fake = [], [], [], [], [], []

        # plot circles to represent peds
        legend_lines = []
        legend_labels = []

        for ped_i in range(num_peds):
            color_real = cmap_real(ped_i)
            # color_fake = cmap_fake(ped_i)
            # color_real = cmap_real(0)

            ## plot ground-truth obs and pred
            if obs_traj is not None:
                circles_gt.append(ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=color_real, zorder=0)))
                line_obs_gt = mlines.Line2D(*obs_traj[0:1].T, color=color_real, marker='.', linestyle='-', linewidth=1,
                                            alpha=obs_alpha, zorder=0)
                lines_obs_gt.append(ax.add_artist(line_obs_gt))

            if pred_traj_gt is not None:
                if obs_traj is None:
                    circles_gt.append(ax.add_artist(plt.Circle(pred_traj_gt[0, ped_i], ped_radius, fill=True, color=color_real, zorder=0)))
                line_pred_gt = mlines.Line2D(*pred_traj_gt[0:1].T, color=color_real, marker='.', linestyle='-', linewidth=1,
                                             alpha=obs_alpha, zorder=0, visible=False)
                lines_pred_gt.append(ax.add_artist(line_pred_gt))

            if pred_traj_fake is not None:  # plot fake pred trajs
                lpf, cf = [], []
                for model_i, ptf in enumerate(pred_traj_fake):
                    lpf_inner, cf_inner = [], []
                    for sample_i, p in enumerate(ptf):
                        circle_fake = plt.Circle(ptf[0, ped_i], ped_radius, fill=True, color=color_fake[ped_i][model_i],
                                                 alpha=pred_alpha, visible=False, zorder=1)
                        cf_inner.append(ax.add_artist(circle_fake))
                        if cfg_names is not None:
                            label = f"{cfg_names[model_i]} ped {ped_i}" if sample_i == 0 else None
                        marker = locals()[f'markers_{model_i}'][sample_i]
                        color = color_fake[ped_i][model_i]
                        line_pred_fake = mlines.Line2D(*ptf[0:1].T, color=color,
                                                       marker=marker,
                                                       linestyle=linestyles[model_i],
                                                       alpha=pred_alpha, zorder=2,
                                                       visible=False)
                        if cfg_names is not None and label is not None:
                            legend_labels.append(label)
                            legend_lines.append(patches.Patch(color=color, linestyle=linestyles[model_i], label=label))

                        lpf_inner.append(ax.add_artist(line_pred_fake))
                    cf.append(cf_inner)
                    lpf.append(lpf_inner)
                lines_pred_fake.append(lpf)
                circles_fake.append(cf)

        ax.legend(handles=legend_lines, loc='upper right')
        # ax.legend(legend_lines, legend_labels, loc='upper right')
        # ax.legend(legend_lines, legend_labels, loc='upper right')
        # ax.legend(handles=[line1, line2])

        # add interaction category annotations, if specified
        ped_texts = []
        if ped_num_label_on == 'gt':
            circles_to_plot_ped_num = circles_gt
        elif ped_num_label_on == 'pred' or obs_traj is None and pred_traj_gt is None:
            circles_to_plot_ped_num = circle_fake
        else:
            raise RuntimeError
        for ped_i, circle in enumerate(circles_to_plot_ped_num):
            int_text = ax.text(circle.center[0] + text_offset_x, circle.center[1] - text_offset_y,
                               str(ped_i), color='black', fontsize=8)
            ped_texts.append(ax.add_artist(int_text))

        # plot collision circles for predictions only
        if collision_mats is not None:
            collide_circle_rad = (ped_radius + ped_discomfort_dist)
            # assert collision_mats.shape == (pred_len, num_peds, num_peds)
            collision_circles = [ax.add_artist(plt.Circle((0, 0), collide_circle_rad, fill=False, zorder=5, visible=False))
                                 for _ in range(num_peds)]
            collision_texts = [ax.add_artist(ax.text(0, 0, "", visible=False, fontsize=8)) for _ in range(num_peds)]
            collision_delay = 3
            yellow = (.8, .8, 0, .2)
            collided_delays = np.zeros(num_peds)

        # heatmap
        if grid_values is not None:
            x, y = np.meshgrid(np.linspace(*bounds[:2], grid_values.shape[1] + 1),
                               np.linspace(*bounds[2:4], grid_values.shape[2] + 1))
            # z = grid_values[0].reshape(x.shape[0] - 1, x.shape[1] - 1)
            z = grid_values[0]

            z_min, z_max = np.min(np.array(z)), np.max(np.array(z))
            state_mesh = ax.pcolormesh(x, y, z, alpha=.8, vmin=0, vmax=1, zorder=3)

        ## animation update function
        def update(frame_i):
            nonlocal x, y
            # for replicating last scene
            if frame_i >= obs_len + pred_len:
                return

            # energy text
            if frame_i < obs_len + pred_len - 1 and text_time is not None:
                energy_time, energy_time_each_ped, dmp, spd, drc, col, atr, grp = text_time[frame_i]
                energy_text = '\n'.join(f"{text_time[frame_i]:0.3f}")
                text_over_time.set_text(f"{energy_text}")

            # heatmap
            if grid_values is not None and frame_i < obs_len + pred_len - 1:
                nonlocal state_mesh, x, y
                z = grid_values[frame_i]
                normed_z = ((z - z_min) / (z_max - z_min)).reshape(x.shape[0] - 1, x.shape[1] - 1)
                state_mesh.remove()
                state_mesh = ax.pcolormesh(x, y, normed_z, alpha=.1, vmin=0, vmax=1, zorder=1)

            # move the real and pred (fake) agent
            if frame_i < obs_len:
                for ped_i, (circle_gt, line_obs_gt) in enumerate(zip(circles_gt, lines_obs_gt)):
                    circle_gt.center = obs_traj[frame_i, ped_i]
                    line_obs_gt.set_data(*obs_traj[0:frame_i + 1, ped_i].T)
                # move the pedestrian texts (ped number and relation)
                for ped_text, circle in zip(ped_texts, circles_to_plot_ped_num):
                    ped_text.set_position((circle.center[0] + text_offset_x, circle.center[1] - text_offset_y))

            elif frame_i == obs_len:
                [circle_fake.set_visible(True) for cf in circles_fake for cf_inner in cf for circle_fake in cf_inner]
                for circle_gt in circles_gt:
                    circle_gt.set_radius(ped_radius * 0.5)
                    circle_gt.set_alpha(0.3)
                for line_obs_gt in lines_obs_gt:
                    line_obs_gt.set_alpha(0.2)
                if pred_traj_gt is not None:
                    for line_pred_gt in lines_pred_gt:
                        line_pred_gt.set_visible(True)
                if pred_traj_fake is not None:
                    for lpf in lines_pred_fake:
                        for lpf_inner in lpf:
                            for line_pred_fake in lpf_inner:
                                line_pred_fake.set_visible(True)

                for last_obs_circ in last_obs_circles:
                    last_obs_circ.set_radius(ped_radius * 0.75)
                    last_obs_circ.set_alpha(0.3)
                # obs_pred_text.set_text('prediction')

            if obs_len <= frame_i < obs_len + pred_len:
                # if frame_i == 12 or frame_i == 11:
                #     import ipdb; ipdb.set_trace()
                if pred_traj_gt is not None:
                    # traj_gt = np.concatenate([obs_traj, pred_traj_gt])
                    assert len(circles_gt) == len(lines_pred_gt) == len(ped_texts), f'{len(circles_gt)}, {len(lines_pred_gt)}, {len(ped_texts)} should all be equal'
                    for ped_i, (circle_gt, line_pred_gt, ped_text) in enumerate(zip(
                            circles_gt, lines_pred_gt, ped_texts)):
                        circle_gt.center = pred_traj_gt[frame_i - obs_len, ped_i]
                        if obs_traj is not None:
                            last_obs_pred_gt = np.concatenate([obs_traj[-1:, ped_i], pred_traj_gt[0:frame_i + 1 - obs_len, ped_i]])
                        else:
                            last_obs_pred_gt = pred_traj_gt[0:frame_i + 1 - obs_len, ped_i]
                        line_pred_gt.set_data(*last_obs_pred_gt.T)
                        # move the pedestrian texts (ped number and relation)
                        if len(ped_texts) > 0:
                            ped_text.set_position(
                                    (circle_gt.center[0] + text_offset_x, circle_gt.center[1] - text_offset_y))

                if pred_traj_fake is not None:
                    assert len(lines_pred_fake) == len(circles_fake)
                    for ped_i, (cf, lpf) in enumerate(zip(circles_fake, lines_pred_fake)):
                        assert len(cf) == len(lpf)
                        for model_i, (cf_inner, lpf_inner) in enumerate(zip(cf, lpf)):
                            assert len(cf_inner) == len(lpf_inner)
                            for sample_i, (circle_fake, line_pred_fake) in enumerate(zip(cf_inner, lpf_inner)):
                                circle_fake.center = pred_traj_fake[model_i][sample_i, frame_i - obs_len, ped_i]
                                if obs_traj is not None:
                                    last_obs_pred_fake = np.concatenate([obs_traj[-1:, ped_i], pred_traj_fake[model_i][sample_i, 0:frame_i + 1 - obs_len, ped_i]])
                                else:
                                    last_obs_pred_fake = pred_traj_fake[model_i][sample_i, 0:frame_i + 1 - obs_len, ped_i]
                                line_pred_fake.set_data(*last_obs_pred_fake.T)

            # update collision circles (only if we are during pred timesteps)
            if (plot_collisions_all or obs_len <= frame_i <= obs_len + pred_len) and collision_mats is not None:
                assert len(collision_mats.shape) == 3 and collision_mats.shape[1] == collision_mats.shape[2], 'collision mats is not square'
                if plot_collisions_all:
                    assert len(collision_mats) == obs_len + pred_len, f'plot_collisons_all is {plot_collisions_all}, so collision_mat size should be {obs_len + pred_len} but is {len(collision_mats)}'
                else:
                    assert len(collision_mats) == pred_len, f'plot_collisons_all is {plot_collisions_all}, so collision_mat size should be {pred_len} but is {len(collision_mats)}'

                if pred_traj_fake is not None and obs_traj is not None:
                    assert len(pred_traj_fake) == 1, "if plotting collision circles, should only plot one model"
                    assert pred_traj_fake[0].shape[0] == 1, "if plotting collision circles, should only plot one sample"
                    obs_gt_fake = np.concatenate([obs_traj, pred_traj_fake[0][0]])
                elif pred_traj_gt is not None and obs_traj is not None:
                    obs_gt_fake = np.concatenate([obs_traj, pred_traj_gt])
                elif pred_traj_fake is not None:
                    obs_gt_fake = pred_traj_fake[0][0]
                elif pred_traj_gt is not None:
                    obs_gt_fake = pred_traj_gt
                else:
                    raise RuntimeError

                for ped_i in range(num_peds):
                    # new frame; decrease the text disappearance delay by 1
                    if collided_delays[ped_i] > 0:
                        collided_delays[ped_i] -= 1
                    for ped_j in range(ped_i):
                        if plot_collisions_all:
                            collision_frame_idx = frame_i
                        else:
                            collision_frame_idx = frame_i - obs_len
                        if collided_delays[ped_i] > 0:  # still in delay, circle doesn't disappear
                            break
                        elif collision_mats[collision_frame_idx, ped_i, ped_j]:
                            ## put the center of the circle in the point between the two ped centers
                            x = (obs_gt_fake[frame_i][ped_i][0] + obs_gt_fake[frame_i][ped_j][0]) / 2
                            y = (obs_gt_fake[frame_i][ped_i][1] + obs_gt_fake[frame_i][ped_j][1]) / 2
                            collision_circles[ped_i].set_center((x, y))
                            collision_circles[ped_i].set_edgecolor(cmap_fake(ped_i))
                            collision_circles[ped_i].set_visible(True)

                            ## add persistent yellow collision circle
                            ax.add_artist(plt.Circle((x, y), collide_circle_rad, fc=yellow, zorder=1, ec='none'))
                            collided_delays[ped_i] = collision_delay
                            break
                        collision_circles[ped_i].set_visible(False)
                        collision_texts[ped_i].set_visible(False)

        self.update = update

        if fig is not None:
            anim = animation.FuncAnimation(fig, update, frames=obs_len + pred_len + extend_last_frame, interval=500)
            anim.save(save_fn)
            print(f"saved animation to {save_fn}")
            plt.close(fig)

