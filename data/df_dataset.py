from data.base_dataset import BaseDataset
import numpy as np
import torch
from os.path import join, dirname


class DFDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        dataroot = join(dirname(__file__), 'objects')
        if opt.class_3d == 'chair':
            filelist = join(dataroot, 'df_chair.txt')
        elif opt.class_3d == 'car':
            filelist = join(dataroot, 'df_car.txt')
        else:
            raise NotImplementedError('%s not supported' % opt.class_3d)
        items = open(filelist).readlines()
        self.items = [join(dirname(__file__), x.strip('\n')) for x in items]
        self.sigma = opt.df_sigma
        self.size = len(self.items)
        self.df_flipped = opt.df_flipped
        self.return_raw = opt.dataset_mode == 'concat_real_df'
        if hasattr(opt, 'real_shape'):
            self.is_test_dummy = not opt.real_shape
        else:
            self.is_test_dummy = False

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--df_sigma', type=float, default=8.0, help='data = exp(-df_sigma * df)')
        parser.add_argument('--df_flipped', action='store_true', help='if specified, flip the generated voxel to match rotation definition. Only used for early 3D GAN version')
        return parser

    def __getitem__(self, index):
        if not self.is_test_dummy:
            return self.get_item(index)
        else:
            return {'voxel': 0, 'path': 'dummy'}

    def get_item(self, index):
        index = index % len(self)
        data = np.load(self.items[index])['df']
        # 1 near surface instead of 0
        data = np.exp(-self.sigma * data)
        data = torch.from_numpy(data).float().unsqueeze(0)
        if not self.df_flipped:
            data = data.transpose(1, 2)
            data = torch.flip(data, [1])
            data = data.transpose(2, 3)
            data = torch.flip(data, [2])
            data = data.contiguous()
        if self.return_raw:
            return data
        else:
            return {'voxel': data, 'path': self.items[index]}

    def __len__(self):
        return self.size
