
from torch.utils.data.sampler import Sampler
import numpy as np
import math


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in np.random.permutation(np.arange(len(self.indices))))

    def __len__(self):
        return len(self.indices)


class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class AdaptiveLengthSampler(Sampler):
    """ Samples elements from part of a predefined index list
    Dataset size can change at each epoch, but not during epoch
    Dataset's max size should be smaller than the size of the predefined index list.
    """

    def __init__(self, data_source, index_list, train_ratio, mode, shuffle=True):
        self.dataset = data_source
        self.index_list = np.array(index_list)
        self.train_ratio = train_ratio
        self.mode = mode
        self.shuffle = shuffle
        assert mode in ('train', 'eval')

    def __iter__(self):
        assert len(self.dataset) <= len(self.index_list)
        train_num = int(math.floor(self.train_ratio * len(self.dataset)))
        all_index = self.index_list
        if len(all_index) > len(self.dataset):
            all_index = list(filter(lambda x: x < len(self.dataset), all_index))
        index_list = all_index[:train_num] if self.mode == 'train' else all_index[train_num:]
        if self.shuffle:
            return (index_list[i] for i in np.random.permutation(np.arange(len(index_list))))
        else:
            return (index_list[i] for i in range(len(index_list)))

    def __len__(self):
        train_num = int(math.floor(self.train_ratio * len(self.dataset)))
        return train_num if self.mode == 'train' else len(self.dataset) - train_num
