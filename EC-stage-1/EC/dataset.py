import os
import glob
import numpy as np
from scipy import misc

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pickle
import copy


figure_configuration_names = ['center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single']

def to_onehot(a, num_classes=10):
    return np.identity(num_classes)[a]

rules_convert_dict = {
    "Constant": {
        "0": 1
    },
    "Progression": {
        "-2": 2,
        "-1": 3,
        "1": 4,
        "2": 5
    },
    "Arithmetic": {
        "-1": 6,
        "1": 7
    },
    "Comparison": {
        "-1": 8,
        "1": 9
    },
    "Varprogression": {
        "1": 10,
        "2": 11,
        "-1": 12,
        "-2": 13,
    },
}

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class RAVENDataset(Dataset):
    def __init__(self, mode, args):
        self.data_dir = args.data_dir
        self.mode = mode
        self.data_format_str = args.data_format_str

        # self.figure_configurations = args.figure_configurations
        with open(os.path.join(self.data_dir, self.data_format_str % self.mode), 'rb') as f:
            data = pickle.load(f) # list of dict

        self.data = []
        for item in data:
            temp_d = dict()
            temp_d.update({'label': np.array(item['label'])})
            if args.visual:
                temp_d.update({'image': item['image'].astype(np.float32)})
                temp_d.update({'target_image': copy.deepcopy(item['image'][8 + item['label']]).astype(np.float32)})
            elif args.symbol:
                s = item['symbol']
                n, k, d = s.shape
                if k == 1:
                    s = np.reshape(s, (n, d)) 
                    s = to_onehot(s, num_classes=args.symbol_onehot_dim)
                    temp_d.update({'symbol': s.astype(np.float32)})
                    temp_d.update({'target_symbol': copy.deepcopy(s[8 + item['label']]).astype(np.float32)})
                else:
                    new_s = []
                    for i in range(n):
                        temp_s = []
                        for j in range(k):
                            if s[i, j, 0] == -1:
                                temp_s.append(np.zeros((args.symbol_attr_dim, args.symbol_onehot_dim)))
                            else:
                                temp_s.append(to_onehot(s[i, j, :], num_classes=args.symbol_onehot_dim))
                        new_s.append(temp_s)
                    temp_d.update({'symbol': np.array(new_s).astype(np.float32)})
                    temp_d.update({'target_symbol': copy.deepcopy(new_s[8 + item['label']]).astype(np.float32)})
            if args.rule:
                rules = []
                
                for r in item['rules']:
                    if len(r) != 1:
                        r = [r]
                    arr = np.array([rules_convert_dict[i[1].capitalize()][str(i[2])] for i in r])
                    onehot_arr = to_onehot(arr, num_classes=args.rules_dim)
                    rules.append(onehot_arr)
                rules = np.concatenate(rules, axis=0)
                temp_d.update({'rules': rules.astype(np.float32)})
            self.data.append(temp_d)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return copy.copy(self.data[idx])
        
