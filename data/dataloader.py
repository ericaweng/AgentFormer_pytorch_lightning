from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy

from .preprocessor import preprocess
from .preprocessor_sdd import SDDPreprocess
from .stanford_drone_split import get_stanford_drone_split
from .ethucy_split import get_ethucy_split, get_ethucy_split_dagger
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.dagger = parser.get('dagger', False)
        self.dagger_split = parser.get('dagger_split', None)
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

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train' and not parser.get('sanity', False):  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test' or parser.get('sanity', False): self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_total_samples_dagger = 0
        self.num_sample_list = []
        self.num_sample_list_dagger = []
        self.sequence = []
        self.sequence_dagger = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip

            if self.dagger and self.dagger_split != 'default' and 'pred' in seq_name:
                self.num_total_samples_dagger += num_seq_samples
                self.num_sample_list_dagger.append(num_seq_samples)
                self.sequence_dagger.append(preprocessor)
            else:
                self.num_total_samples += num_seq_samples
                self.num_sample_list.append(num_seq_samples)
                self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        self.sample_list_dagger = list(range(self.num_total_samples_dagger))
        self.index = 0
        self.index_dagger = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        random.shuffle(self.sample_list_dagger)

    def get_seq_and_frame_dagger(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list_dagger)):    # 0-indexed
            if index_tmp < self.num_sample_list_dagger[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence_dagger[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list_dagger[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        if self.dagger and self.dagger_split != 'default' and np.random.rand() < self.dagger_split:
            sample_index = self.num_sample_list_dagger[self.index]
            seq_index, frame = self.get_seq_and_frame_dagger(sample_index)
            seq = self.sequence_dagger[seq_index]
            self.index_dagger += 1
            data = seq(frame)
        else:
            sample_index = self.sample_list[self.index]
            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            self.index += 1

            data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()
