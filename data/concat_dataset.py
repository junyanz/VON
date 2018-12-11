from .base_dataset import BaseDataset


class ConcatDataset(BaseDataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'ConcatDataset'

    def initialize(self, opt):
        for dataset in self.datasets:
            dataset.initialize(opt)
