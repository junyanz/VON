from .images_dataset import ImagesDataset
from .voxel_dataset import VoxelDataset
from .concat_dataset import ConcatDataset


class ImageAndVoxelDataset(ConcatDataset):
    def __init__(self):
        self.datasets = [ImagesDataset(), VoxelDataset()]

    @staticmethod
    def modify_commandline_options(parser, is_train):
        ImagesDataset.modify_commandline_options(parser, is_train)
        VoxelDataset.modify_commandline_options(parser, is_train)
        return parser

    def name(self):
        return 'ImageAndVoxelDataset'
