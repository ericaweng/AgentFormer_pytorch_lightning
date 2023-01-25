import torch, os, numpy as np, copy
import cv2
import glob
from .map import GeometricMap


class SDDPreprocess(object):

    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)
        self.traj_scale = parser.traj_scale
        self.past_traj_scale = parser.traj_scale
        self.load_map = parser.get('load_map', False)
        self.map_version = parser.get('map_version', '0.1')
        self.seq_name = seq_name
        self.split = split
        self.phase = phase
        self.log = log

        label_path = f'{data_root}/{split}/{seq_name}.txt'
        delimiter = ' '

        self.gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
        assert np.all(self.gt[:, 0].astype(np.int) % 12) == 0
        self.gt = self.gt.astype('float32')
        self.gt[:, 0] = np.round(self.gt[:, 0] / 12.0)

        frames = self.gt[:, 0].astype(np.int)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start
        self.num_fr = fr_end + 1 - fr_start

        if self.load_map:
            self.load_scene_map()
        else:
            self.geom_scene_map = None

        self.xind, self.zind = 2, 3

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self):
        return self.num_fr

    def PreData(self, frame):
        DataList = []
        for i in range(self.past_frames):
            if frame - i < self.init_frame:
                data = []
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def FutureData(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        cur_id = self.GetID(pre_data[0])
        valid_id = []
        for idx in cur_id:
            exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id

    def get_pred_mask(self, valid_id, pre_data, fut_data):
        pred_mask = []
        for idx in valid_id:
            exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:self.past_frames]]
            exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:self.future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                pred_mask.append(1)
            else:
                pred_mask.append(-1)
        return np.array(pred_mask)

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data[cur_data[:, 1] == idx].squeeze()[16]
        return heading

    def load_scene_map(self):
        map_file = f'{self.data_root}/annotations/{self.seq_name[:-2]}/video{self.seq_name[-1]}/reference.jpg'
        # map_vis_file = f'{self.data_root}/map_{self.map_version}/vis_{self.seq_name}.png'
        # map_meta_file = f'{self.data_root}/map_{self.map_version}/meta_{self.seq_name}.txt'
        self.scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        print("scene_map.shape:", self.scene_map.shape)
        # self.scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
        # self.meta = np.loadtxt(map_meta_file)
        # self.map_origin = self.meta[:2]
        # self.map_scale = scale = self.meta[2]
        # homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        homography = np.eye(3)
        self.geom_scene_map = GeometricMap(self.scene_map, homography, np.array([0, 0]))
        # self.scene_vis_map = GeometricMap(self.scene_vis_map, homography, np.array([0, 0]))

    def PreMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            for j in range(self.past_frames):
                past_data = DataTuple[j]              # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.past_traj_scale
                    box_3d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames-1 - j] = 1.0
                elif j > 0:
                    box_3d[self.past_frames-1 - j, :] = box_3d[self.past_frames - j, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def FutureMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.future_frames)
            pos_3d = torch.zeros([self.future_frames, 2])
            for j in range(self.future_frames):
                fut_data = DataTuple[j]              # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.traj_scale
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        return motion, mask

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (frame, self.TotalFrame())


        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        pred_mask = self.get_pred_mask(valid_id, pre_data, fut_data)
        heading = None

        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame
        }

        return data
