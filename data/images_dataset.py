import os.path
from data.base_dataset import BaseDataset, get_transform, get_normaliztion
import numpy as np
import random
from PIL import Image
import torch
from torch.nn.functional import pad as pad_tensor
from os.path import join, dirname


class ImagesDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.root = join(dirname(__file__), 'images')
        if opt.class_3d == 'car':
            pose_pool = np.load(join(self.root, 'pose_car.npz'))
            azs = pose_pool['azs']
            eles = pose_pool['eles']
            self.pose_pool = np.zeros([len(azs), 2])
            self.pose_pool[:, 0] = np.array(eles)
            self.pose_pool[:, 1] = np.array(azs)
            np.random.shuffle(self.pose_pool)
            crawl_list = os.path.join(self.root, 'imgs_car.txt')
        elif opt.class_3d == 'chair':
            pose_pool = np.load(join(self.root, 'pose_chair.npz'))
            azs = pose_pool['azs']
            eles = pose_pool['eles']
            self.pose_pool = np.zeros([len(azs), 2])
            self.pose_pool[:, 0] = np.array(eles)
            self.pose_pool[:, 1] = np.array(azs)
            np.random.shuffle(self.pose_pool)
            crawl_list = os.path.join(self.root, 'imgs_chair.txt')
        else:
            raise NotImplementedError

        with open(crawl_list) as f:
            imgs_paths = f.read().splitlines()
        self.paths = [join(dirname(__file__), x) for x in imgs_paths]
        self.size = len(self.paths)
        self.transform_mask = get_transform(opt, has_mask=True, no_flip=True, no_normalize=True)
        self.transform_rgb = get_transform(opt, has_mask=False, no_flip=True, no_normalize=True)
        self.no_flip = opt.no_flip
        self.random_shift = opt.random_shift
        if hasattr(opt, 'real_texture'):
            self.is_test_dummy = not opt.real_texture
        else:
            self.is_test_dummy = False

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--random_shift', action='store_true', help='add random shift to real images and rendered ones')
        parser.add_argument('--color_jitter', action='store_true', help='jitter the hue of loaded images')
        # type of  pose pool to sample from:
        parser.add_argument('--pose_type', type=str, default='hack', choices=['hack'], help='select which pool of poses to sample from')
        parser.add_argument('--pose_align', action='store_true', help='choose to shuffle pose or not. not shuffling == paired pose')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        return parser

    def shift(self, rgb, mask):
        shift_h = random.randint(-2, 2)
        shift_v = random.randint(-2, 2)
        rgb_shift = pad_tensor(rgb, (shift_v, -shift_v, shift_h, -shift_h), mode='constant', value=1)
        mask_shift = pad_tensor(mask, (shift_v, -shift_v, shift_h, -shift_h), mode='constant', value=0)
        return rgb_shift.data, mask_shift.data

    def set_aligned(self, aligned):
        self.aligned = aligned

    @staticmethod
    def azele2matrix(az=0, ele=0):
        R0 = torch.zeros([3, 3])
        R = torch.zeros([3, 4])
        R0[0, 1] = 1
        R0[1, 0] = -1
        R0[2, 2] = 1
        az = az * np.pi / 180
        ele = ele * np.pi / 180
        cos = np.cos
        sin = np.sin
        R_ele = torch.FloatTensor(
            [[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]])
        R_az = torch.FloatTensor(
            [[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R_rot = torch.mm(R_az, R_ele)
        R_all = torch.mm(R_rot, R0)
        R[:3, :3] = R_all
        return R

    def get_posepool(self):
        return self.pose_pool

    def __getitem__(self, index):
        if not self.is_test_dummy:
            return self.get_item(index)
        else:
            return {'image_paths': 'dummy', 'image': 0, 'rotation_matrix': 0, 'real_im_mask': 0, 'viewpoint': 0}

    def __len__(self):
        return len(self.paths)

    def get_item(self, index):
        index = index % len(self)  # .__len__()
        pose_id = index
        flip = not self.no_flip and random.random() < 0.5
        azs = self.pose_pool[pose_id, 1]
        eles = self.pose_pool[pose_id, 0]
        if flip:
            azs = -azs
        R = self.azele2matrix(azs, self.pose_pool[pose_id, 0])
        viewpoint = np.array([azs, eles]) * np.pi / 180
        pathA = self.paths[index % self.size]
        im = Image.open(pathA)
        im_out = self.transform_mask(im)
        mask = im_out[3, :, :]
        mask = mask.unsqueeze(0)
        rgb = im.convert('RGB')
        rgb = self.transform_rgb(rgb)
        if self.random_shift:
            rgb, mask = self.shift(rgb, mask)
        rgb = get_normaliztion()(rgb)
        if flip:
            idx = [i for i in range(rgb.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            rgb = rgb.index_select(2, idx)
            mask = mask.index_select(2, idx)

        return {'image_paths': pathA, 'image': rgb, 'rotation_matrix': R, 'real_im_mask': mask, 'viewpoint': viewpoint}
