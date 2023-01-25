import torch

from itertools import zip_longest
from torch._six import string_classes
from torch.utils.data import Dataset

from data.nuscenes_pred_split import get_nuscenes_pred_split
import numpy as np

from .preprocessor import preprocess
from .preprocessor_sdd import SDDPreprocess
from .stanford_drone_split import get_stanford_drone_split
from .ethucy_split import get_ethucy_split, get_ethucy_split_dagger


class AgentFormerDataset(Dataset):
    """ torch Dataset """

    def __init__(self, parser, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.dagger = parser.get('dagger', False)
        self.dagger_data = parser.get('dagger_data', None)
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif self.dagger:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = get_ethucy_split_dagger(parser.dataset, self.dagger_data)
            print("len(seq_test):", len(seq_test))
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset == 'trajnet_sdd':
            data_root = parser.data_root_trajnet_sdd
            seq_train, seq_val, seq_test = get_stanford_drone_split()
        else:
            raise ValueError('Unknown dataset!')

        process_func = SDDPreprocess if 'sdd' in parser.dataset else preprocess
        self.data_root = data_root

        print("\n-------------------------- loading %s data --------------------------" % split)
        if self.split == 'train' and not parser.get('sanity', False):
            self.sequence_to_load = seq_train
        elif self.split == 'val':
            self.sequence_to_load = seq_val
        elif self.split == 'test' or parser.get('sanity', False):
            self.sequence_to_load = seq_test
        else:
            assert False, 'error'

        num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            print("loading sequence {} ...".format(seq_name))
            preprocessor = process_func(data_root, seq_name, parser, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (
                        parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip

            num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)

        print(f'total num samples: {num_total_samples}')

        datas = []
        for idx in list(range(num_total_samples)):
            seq_index, frame = self.get_seq_and_frame(idx)
            seq = self.sequence[seq_index]
            data = seq(frame)
            if data is None:
                continue
            datas.append(data)
        self.sample_list = datas

        print(f'num valid samples: {len(self.sample_list)}')
        print("------------------------------ done --------------------------------\n")

    def __len__(self):
        return len(self.sample_list)#self.num_total_samples

    def get_seq_and_frame(self, index):
        index_tmp = index
        # find the seq_name (environment) this index corresponds to, and then the frame in that environment
        for seq_index in range(len(self.num_sample_list)):  # 0-indexed  # for each seq_name
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[
                    seq_index].init_frame  # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def __getitem__(self, idx):
        """Returns a single example from dataset
            Args:
                idx: Index of scenario
            Returns:
                output: Necessary values for scenario
        """
        return self.sample_list[idx]

    @staticmethod
    def collate(batch):
        """batch: list of data objects
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
        """
        return batch[0]
        def pad_and_stack(batch):
            import ipdb; ipdb.set_trace()
            max_shape_in_each_dim = torch.max(torch.stack([data.shape for data in batch]), axis=-1)
            # torch.zeros((len(batch), *max_shape_in_each_dim)).to(batch[0].device)
            batch = [torch.nn.functional.pad(data, tuple([dim - data.shape[dim_i] for dim_i, dim in enumerate(max_shape_in_each_dim)]), value=0.) for data in batch]
            batch = torch.stack(batch)
            return batch

        data_all = {}
        for key in batch[0]:
            if key in 'pre_motion_3D,fut_motion_3D,fut_motion_mask,pre_motion_mask,pre_data,fut_data':
                data_all[key] = pad_and_stack([data[key] for data in batch])
            elif True:
                pass
            else:
                raise RuntimeError(f"unexpected key: {key}")


        return data_all
        def pad_batch(batch_dict, max_agent_actors, max_social_actors, max_graph_segments, max_lane_segment_lengths,
                      max_num_edges):
            '''
                Pad batch such that all examples have same number of social actors. Allows for batch training of graph models.
            '''
            for key, value in batch_dict.items():
                if isinstance(value, dict):
                    batch_dict[key] = pad_batch(value, max_agent_actors, max_social_actors, max_graph_segments,
                                                max_lane_segment_lengths, max_num_edges)
                elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    if 'pred_history' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            if len(elem.size()) > 1:
                                elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            else:
                                elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_history' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            if len(elem.size()) > 1:
                                elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            else:
                                elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'adjacency' in key:
                        for index, elem in enumerate(value):
                            pred_agents = batch_dict['masks']['num_pred_mask'][index].sum().long().item()
                            social_agents = batch_dict['masks']['num_social_mask'][index].sum().long().item()
                            if social_agents == 0:
                                social_agents = 1
                            num_pred_agent_pad = max_agent_actors - pred_agents
                            num_social_agent_pad = max_social_actors - social_agents
                            elem = torch.cat([torch.nn.functional.pad(elem[:, :pred_agents, :],
                                                                      (0, 0, 0, num_pred_agent_pad), value=0.),
                                              torch.nn.functional.pad(
                                                      elem[:, pred_agents:, :], (0, 0, 0, num_social_agent_pad),
                                                      value=0.)], 1)
                            elem = torch.cat([torch.nn.functional.pad(elem[:, :, :pred_agents], (0, num_pred_agent_pad),
                                                                      value=0.), torch.nn.functional.pad(
                                    elem[:, :, pred_agents:], (0, num_social_agent_pad), value=0.)], 2)
                            value[index] = elem
                    if 'mask' in key:
                        for index, elem in enumerate(value):
                            num_agent = (max_social_actors - elem.size(0)
                                         ) if key == 'num_social_mask' else (max_agent_actors - elem.size(0))
                            if num_agent == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, num_agent), value=0.)
                            value[index] = elem
                    if 'pred_labels' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_labels' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'pred_labels_lengths' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_labels_lengths' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'pred_translation' == key or 'pred_rotation' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if len(elem.size()) == 2:
                                elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            else:
                                elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_translation' == key or 'social_rotation' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if len(elem.size()) == 2:
                                elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            else:
                                elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'pred_grid' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_grid' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'lane_graph' in key:
                        for index, elem in enumerate(value):
                            if 'inverted' in key:
                                num_agent_pad = max_num_edges - elem.size(0)
                            else:
                                num_agent_pad = max_graph_segments - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, num_agent_pad, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'inverted_indices' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_num_edges - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'lane_segments' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_graph_segments - elem.size(0)
                            num_segment_pad = max_graph_segment_lengths - elem.size(1)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_segment_pad, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'lane_segments_lengths' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_graph_segments - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'closest_lane_ids' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = (max_agent_actors - elem.size(0)) if 'pred' in key else (
                                        max_social_actors - elem.size(0))
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'edges' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_graph_segments - elem.size(1)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'lane_xy_delta' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_graph_segments - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'lane_xy_delta_start' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_graph_segments - elem.size(0)
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'pred_xy_delta' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_xy_delta' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_xy_delta_start' == key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'social_tstamps' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_social_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    if 'pred_tstamps' in key:
                        for index, elem in enumerate(value):
                            num_agent_pad = max_agent_actors - elem.size(0)
                            if num_agent_pad == 0:
                                continue
                            elem = torch.nn.functional.pad(elem, (0, 0, 0, num_agent_pad), value=0.)
                            value[index] = elem
                    try:
                        batch_dict[key] = torch.stack(value)
                    except:
                        import ipdb;
                        ipdb.set_trace()
                        a = 1
            return batch_dict

        def collate_batch(batch):
            """Puts each data field into a tensor with outer dimension batch size"""
            elem_type = type(batch[0])
            if isinstance(batch[0], torch.Tensor):
                out = None
                if False:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = batch[0].storage()._new_shared(numel)
                    out = batch[0].new(storage)
                try:
                    return torch.stack(batch, 0, out=out)
                except:
                    return batch
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                elem = batch[0]
                if elem_type.__name__ == 'ndarray':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(error_msg_fmt.format(elem.dtype))

                    return collate_batch([torch.from_numpy(b) for b in batch])
                if elem.shape == ():  # scalars
                    py_type = float if elem.dtype.name.startswith('float') else int
                    return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
            elif isinstance(batch[0], float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(batch[0], int_classes):
                return torch.tensor(batch)
            elif isinstance(batch[0], string_classes):
                return batch
            elif isinstance(batch[0], container_abcs.Mapping):
                return {key: collate_batch([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
                return type(batch[0])(*(collate_batch(samples) for samples in zip(*batch)))
            elif isinstance(batch[0], container_abcs.Sequence):
                transposed = zip_longest(*batch)
                return [collate_batch(samples) for samples in transposed]
            else:
                return batch
            raise TypeError((error_msg_fmt.format(type(batch[0]))))

        batch = collate_batch(batch)
        max_agent_actors = np.max([x.shape[0] for x in batch[0]['pred_history']])
        max_social_actors = np.max([x.shape[0] for x in batch[0]['social_history']])
        max_graph_segments = np.max([x.shape[0] for x in batch[0]['lane_graph']])
        max_graph_segment_lengths = np.max([x.shape[1] for x in batch[0]['lane_segments']])
        try:
            max_num_edges = np.max([x.shape[0] for x in batch[0]['inverted_lane_graph']])
        except:
            max_num_edges = 0
        batch[0] = pad_batch(batch[0], max_agent_actors, max_social_actors, max_graph_segments,
                             max_graph_segment_lengths, max_num_edges)
        batch[1] = pad_batch(batch[1], max_agent_actors, max_social_actors, max_graph_segments,
                             max_graph_segment_lengths, max_num_edges)
        return batch