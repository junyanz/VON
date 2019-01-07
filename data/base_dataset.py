"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import numpy as np


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform(opt, has_mask=False, no_flip=None, no_normalize=False):
    transform_list = []
    if hasattr(opt, 'color_jitter') and opt.color_jitter and not has_mask:
        transform_list.append(transforms.ColorJitter(hue=0.1))
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.crop_size)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size)))
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif opt.resize_or_crop == 'crop_real_im':
        transform_list.append(transforms.Lambda(
            lambda img: __pad_real_im(img, img_size=opt.load_size, padding_value=0 if has_mask else 255)))
        transform_list.append(transforms.Resize((opt.load_size, opt.load_size), Image.BICUBIC))

    if opt.isTrain and not no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if has_mask or no_normalize:
        transform_list += [transforms.ToTensor()]
    else:
        transform_list += [transforms.ToTensor(), get_normaliztion()]
    return transforms.Compose(transform_list)


def get_normaliztion():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __pad_real_im(img_pil, padding_pix_pct=0.03, img_size=256, padding_value=255):
    img = np.asarray(img_pil)
    padding_pix = int(img_size * padding_pix_pct)
    """
    make the bounding box larger by a few pixels (equiv. to
    padding_pix pixels after resize), then add edge padding
    to make it a square, then resize to desired resolution
    """
    y1, x1, y2, x2 = [0, 0, img.shape[1], img.shape[0]]
    # print(img.shape)
    # print(y1, x1, y2, x2)
    w, h = img.shape[1], img.shape[0]
    x_mid = (x1 + x2) / 2.
    y_mid = (y1 + y2) / 2.
    ll = max(x2 - x1, y2 - y1) * img_size / (img_size - 2. * padding_pix)
    x1 = int(np.round(x_mid - ll / 2.))
    x2 = int(np.round(x_mid + ll / 2.))
    y1 = int(np.round(y_mid - ll / 2.))
    y2 = int(np.round(y_mid + ll / 2.))
    # print(y1, x1, y2, x2)
    b_x = 0
    if x1 < 0:
        b_x = -x1
        x1 = 0
    b_y = 0
    if y1 < 0:
        b_y = -y1
        y1 = 0
    a_x = 0
    if x2 >= h:
        a_x = x2 - (h - 1)
        x2 = h - 1
    a_y = 0
    if y2 >= w:
        a_y = y2 - (w - 1)
        y2 = w - 1
    # print(y1, x1, y2, x2)
    if len(img.shape) == 2:
        crop = np.pad(img[x1: x2 + 1, y1: y2 + 1],
                      ((b_x, a_x), (b_y, a_y)), mode='constant', constant_values=padding_value)
    else:
        crop = np.pad(img[x1: x2 + 1, y1: y2 + 1],
                      ((b_x, a_x), (b_y, a_y), (0, 0)), mode='constant', constant_values=padding_value)
    return Image.fromarray(np.uint8(crop))


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
