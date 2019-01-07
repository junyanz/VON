import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import torch
from util.util_voxel import downsample
from util.util_print import str_verbose, str_error
import os
from scipy.io import loadmat
import numpy as np
import glob
from os.path import dirname, basename

paths = {
    'canon_scaled': {
        'merged': None,  # '/data/vision/billf/jwu-phys/shape_oneshot/ckzhang/datacache/ShapeNetCore.v2/{classid}_{res:d}_{split:02d}.mat',
        'filelist': '/data/vision/billf/jwu-phys/shape_oneshot/ckzhang/datacache/ShapeNetCore.v2.scaled/{classid}_{res:d}.csv',
        'filepath': '/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/{classid}/*/models/model_normalized_{res:d}.mat',
        'matname': 'voxel'
    },
    'canon': {
        'merged': None,
        'filelist': '/data/vision/billf/jwu-phys/shape_oneshot/ckzhang/datacache/ShapeNetCore.v2/{classid}_{res:d}.csv',
        'filepath': '/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/{classid}/*/models/model_normalized_samescale_zup_{res:d}.mat',
        'matname': 'voxel'
    },
    'rotated_scaled': {
        'merged': None,  # '/data/vision/billf/jwu-phys/shape_oneshot/xiuming/output/shapenet-core-v2_normal-depth-voxel-packs/packs_{pack_n}/?????????.npz',
        'filelist': '/data/vision/billf/jwu-phys/shape_oneshot/ckzhang/datacache/ShapeNetCore.v2.scaled.rotate/{classid}_{res:d}.csv',
        'filepath': '/data/vision/billf/jwu-phys/shape_oneshot/xiuming/output/shapenet-core-v2_single-pass/{classid}/*/{classid}_*_view???_voxel_{res:d}.npz',
        'matname': 'voxel',
        # 'merged_matname': 'voxels',
        # 'merged_ids': 'class_ids'
    },
    'rotated': {
        'merged': None,
        'filelist': '/data/vision/billf/jwu-phys/shape_oneshot/ckzhang/datacache/ShapeNetCore.v2.rotate/{classid}_{res:d}.csv',
        'filepath': '/data/vision/billf/jwu-phys/shape_oneshot/xiuming/output/shapenet-core-v2_single-pass/{classid}/*/{classid}_*_view???_gt_rotvox_samescale_{res:d}.npz',
        'matname': 'voxel',
    }
}


common_classes = {
    'all': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03001627+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04379243+04401088+04460130+04468005+04530566+04554684',
    'ikea': '02818832+02871439+03001627+04256520+04379243',
    '7class': '02691156+02958343+03001627+04090263+04256520+04379243+04530566',
    'test': '04074963',
    'chair': '03001627',
    'car': '02958343',
    'plane': '02691156',
    'mug': '03797390',
    'lamp': '03636649',
    'tl': '02818832+03001627+03337140+04256520+04379243',
    'r2n2': '02691156+02828884+03337140+02958343+03001627+03211117+03636649+03691459+03948459+04090263+04256520+04379243+02992529+04401088+04530566',
    'drc': '02691156+02958343+03001627',
}
common_classes['ptset'] = common_classes['r2n2']
label_to_class = common_classes['all'].split('+')
class_to_label = {label_to_class[idx]: idx for idx in range(len(label_to_class))}


chair_subclass_file = '/data/vision/billf/jwu-phys/shape_oneshot/xiuming/output/shapenet-core-v2_subclass/filelist_03001627.txt'
chair_subclass_aliases = {
    'lt30': '+'.join([str(x) for x in range(20, 46)]),
    'ge30': '+'.join([str(x) for x in range(1, 20)])}


def _parse_class(class_str):
    class_remove = None
    if class_str in common_classes:
        class_str = common_classes[class_str]
    elif class_str[:8] == 'all_but_':
        class_remove = class_str[8:]
        class_str = common_classes['all']

    classlist = class_str.split('+')
    if class_remove is not None:
        for c in class_remove.split('+'):
            if c not in classlist:
                print(str_error, 'removing a class not used: ' + c)
            classlist.remove(c)

    return sorted(classlist)


class VoxelDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init(self, opt)
        opt.downsample = 1
        opt.pack_n = 10000
        opt.excl_subclass = None

        classes = opt.class_3d
        assert classes is not None, 'dataset argument [classes] has to be set'

        self.res = opt.voxel_res

        dataset_paths = paths['canon']

        # parse classes
        classlist = _parse_class(classes)
        self._class_str = '+'.join(classlist)

        assert (opt.excl_subclass is None) or (classlist == ['03001627']), 'subclass exclusion only supported for chair'

        class_size = dict()
        class_to_ind = dict()
        filelist = list()
        labellist = list()

        self._matname = dataset_paths['matname']
        dataset_merged_csv = dataset_paths['filelist']
        dataset_path_format = dataset_paths['filepath']
        for i, c in enumerate(classlist):
            merged_csvfile = dataset_merged_csv.format(classid=c, res=self.res)
            if os.path.isfile(merged_csvfile):
                with open(merged_csvfile, 'r') as fin:
                    class_filelist = list(map(str.strip, fin.readlines()))
            else:
                class_filelist = sorted(glob.glob(dataset_path_format.format(classid=c, res=self.res), recursive=False))
            if len(class_filelist) == 0:
                raise ValueError('No .mat files found for class: ' + c)
            class_size[c] = len(class_filelist)
            class_to_ind[c] = i
            filelist += sorted(class_filelist)
            labellist += [class_to_label[c]] * len(class_filelist)

            # filter out specific subclasses
            subclasses_excl = opt.excl_subclass
            if subclasses_excl is not None:
                assert (classlist == ['03001627']), \
                    "Excluding subclasses only supported for chairs"
                if subclasses_excl in chair_subclass_aliases:
                    subclasses_excl = chair_subclass_aliases[subclasses_excl]
                subclasses_excl = (set(subclasses_excl.split('+')))
                with open(chair_subclass_file, 'r') as f:
                    lines = f.readlines()
                list_obj_ids = [l.split(' ')[0] for l in lines]
                list_subclasses = [set(l.split(' ')[1].replace('\n', '').split(',')) for l in lines]
                subclass_dict = dict(zip(list_obj_ids, list_subclasses))
                idlist = [basename(dirname(dirname(filename))) for filename in filelist]
                intersection_list = [(len(subclasses_excl.intersection(subclass_dict[id_])) if id_ in subclass_dict else -1) for id_ in idlist]
                print(str_verbose, 'subclass exclusion: ')
                print(str_verbose, '\tvoxels kept: %d' % (np.array(intersection_list) == 0).sum())
                print(str_verbose, '\tvoxels excluded: %d' % (np.array(intersection_list) > 0).sum())
                print(str_verbose, '\tvoxels missing subclass: %d' % (np.array(intersection_list) < 0).sum())

                # update filelist and labellist
                filelist = [filelist[i] for i in range(len(filelist)) if intersection_list[i] == 0]
                labellist = [labellist[i] for i in range(len(labellist)) if intersection_list[i] == 0]

        self._classes = classlist
        self._class_to_ind = class_to_ind
        self._filelist = filelist
        self._labellist = labellist

        self._transform = self.get_transform(opt, opt.downsample)
        self.size = len(self._filelist)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def _set_use_dict(self):
        self.is_dict = True

    def get_transform(self, opt, downsample_ratio):
        transform_list = list()
        if opt.downsample > 1:
            transform_list.append(lambda v: downsample(v, times=downsample_ratio, use_max=True))
        transform_list.append(lambda v: v[np.newaxis, :, :, :])
        transform_list.append(lambda v: torch.from_numpy(v).float())
        transform = transforms.Compose(transform_list)
        return transform

    def __getitem__(self, index):
        index = index % len(self)
        filename = self._filelist[index]
        if filename.endswith('.mat'):
            voxel = loadmat(filename)[self._matname]
        elif filename.endswith('.npz'):
            voxel = np.load(filename)[self._matname]
        voxel[voxel > 1] = 1     # fix a numerial bug when generating rotated voxels
        if self._transform:
            voxel = self._transform(voxel)
        return {'voxel': voxel}

    def get_classes(self):
        return self._class_str

    def __len__(self):
        return self.size
