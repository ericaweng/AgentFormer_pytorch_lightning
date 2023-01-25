import torch
import pytorch_lightning as pl
import numpy as np
from functools import partial
from eval import eval_one_seq
from metrics import stats_func
import multiprocessing
from model.agentformer import AgentFormer
from utils.torch import get_scheduler

from viz_utils import plot_fig


class AgentFormerTrainer(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        self.model = AgentFormer(cfg)
        self.cfg = cfg
        self.args = args
        self.num_workers = min(args.num_workers, int(multiprocessing.cpu_count() / args.devices))
        self.batch_size = args.batch_size
        self.collision_rad = cfg.get('collision_rad', 0.1)
        self.hparams.update(vars(cfg))
        self.hparams.update(vars(args))

    def on_test_start(self):
        self.model.set_device(self.device)

    def on_fit_start(self):
        self.model.set_device(self.device)

    def _step(self, batch, mode):
        self.model.set_data(batch)
        data = self.model()
        total_loss, loss_dict, loss_unweighted_dict = self.model.compute_loss()

        # losses
        self.log(f'{mode}/loss', total_loss, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)
        for loss_name, loss in loss_dict.items():
            self.log(f'{mode}/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True, logger=True, batch_size=self.batch_size)

        return data, {'loss': total_loss, **loss_dict}

    def training_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'train')
        return loss_dict

    def validation_step(self, batch, batch_idx):
        data, loss_dict = self._step(batch, 'test')
        gt_motion = self.cfg.traj_scale * data['fut_motion'].transpose(1, 0).cpu()
        pred_motion = self.cfg.traj_scale * data[f'infer_dec_motion'].detach().cpu()
        return {**loss_dict, 'gt_motion': gt_motion, 'pred_motion': pred_motion, 'frame': batch['frame'], 'seq': batch['seq']}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _epoch_end(self, outputs, mode):
        args_list = [(output['gt_motion'].numpy(), output['pred_motion'].numpy()) for output in outputs]
        with multiprocessing.Pool(self.num_workers) as pool:
            all_meters_values = pool.starmap(partial(eval_one_seq,
                                                     collision_rad=self.collision_rad,
                                                     return_agent_traj_nums=False,
                                                     return_sample_vals=self.args.save_viz), args_list)
        if self.args.save_viz:
            all_meters_values, all_sample_vals = zip(*all_meters_values)
            args_list = []
            all_sample_vals = np.array(all_sample_vals).swapaxes(1, 2)
            for frame_i, (output, sample_vals) in enumerate(zip(outputs, all_sample_vals)):
                frame = output['frame']
                seq = output['seq']
                pred_gt_traj = output['gt_motion'].numpy().swapaxes(0, 1)
                pred_fake_traj = output['pred_motion'].numpy().transpose(1,2,0,3)
                anim_save_fn = f'viz/{seq}_frame-{frame}.mp4'
                args_dicts = [anim_save_fn, f"Seq: {seq} frame: {frame}", (5, 4)]
                for sample_i, sample_vals in enumerate(sample_vals):
                    args_dict = {'plot_title': f"Sample {sample_i}",
                                 'pred_traj_gt': pred_gt_traj,
                                 'pred_traj_fake': pred_fake_traj[sample_i],
                                 'text_fixed': "\n".join([f"{k} {v:0.4f}" for k, v in zip(stats_func.keys(), sample_vals)])
                                 }
                    args_dicts.append(args_dict)
                args_list.append(args_dicts)

            with multiprocessing.Pool(self.num_workers) as pool:
                pool.starmap(plot_fig, args_list)

        total_num_agents = np.array([output['gt_motion'].shape[0] for output in outputs])
        self.log(f'{mode}/total_num_agents', float(np.sum(total_num_agents)), sync_dist=True, logger=True)
        for key, values in zip(stats_func.keys(), zip(*all_meters_values)):
            value = np.sum(values * total_num_agents) / np.sum(total_num_agents)
            self.log(f'{mode}/{key}', value, sync_dist=True, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):
        self.model.step_annealer()

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler_type = self.cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear':
            scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=self.cfg.lr_fix_epochs, nepoch=self.cfg.num_epochs)
        elif scheduler_type == 'step':
            scheduler = get_scheduler(optimizer, policy='step', decay_step=self.cfg.decay_step, decay_gamma=self.cfg.decay_gamma)
        else:
            raise ValueError('unknown scheduler type!')

        return [optimizer], [scheduler]