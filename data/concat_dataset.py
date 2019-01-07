from .base_dataset import BaseDataset


class ConcatDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
