from .images_dataset import ImagesDataset
from .df_dataset import DFDataset
from .concat_dataset import ConcatDataset


class ImageAndDFDataset(ConcatDataset):
    def __init__(self, opt):
        ConcatDataset.__init__(self, opt)
        self.datasets = [ImagesDataset(opt), DFDataset(opt)]

    @staticmethod
    def modify_commandline_options(parser, is_train):
        ImagesDataset.modify_commandline_options(parser, is_train)
        DFDataset.modify_commandline_options(parser, is_train)
        return parser
