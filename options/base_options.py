import argparse
import os
from util import util
import pickle
import models
import data
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, default=None, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=12, help='batch size')
        parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
        parser.add_argument('--fine_size', type=int, default=128, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--nz_texture', type=int, default=8, help='the dimension of texture code')
        parser.add_argument('--nz_shape', type=int, default=200, help='the dimension of shape code')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='base', help='choose which model to use: base | shape_gan | stage2_real | stage2 | full | test')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='val', help='train | val | test, etc')
        parser.add_argument('--num_threads', default=6, type=int, help='# sthreads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='../../results_texture/', help='models are saved here')
        parser.add_argument('--display_winsize', type=int, default=128, help='display window size')
        # dataset
        parser.add_argument('--dataset_mode', type=str, default='base', help='chooses how datasets are loaded: base | image_and_df | image_and_voxel | df | voxel')
        parser.add_argument('--resize_or_crop', type=str, default='crop_real_im', help='crop_real_im | resize_and_crop | crop | scale_width | scale_width_and_crop')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # models
        parser.add_argument('--num_Ds', type=int, default=2, help='the number of discriminators')
        parser.add_argument('--netD', type=str, default='multi', help='selects model to use for netD: single | multi')
        parser.add_argument('--netG', type=str, default='resnet_cat', help='selects model to use for netG: unet | resnet_cat')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--netE', type=str, default='adaIN', help='selects model to use for netE: resnet | conv | adaIN')
        parser.add_argument('--where_add', type=str, default='all', help='where to add z in the network G: input | all')
        parser.add_argument('--netG_3D', type=str, default='G0', help='selects model to use for netG_3D: G0')
        parser.add_argument('--netD_3D', type=str, default='D0', help='selects model to use for netD_3D: D0')
        parser.add_argument('--norm', type=str, default='inst', help='instance normalization or batch normalization: batch | inst | none')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu (we hard-coded lrelu for the discriminator)')
        parser.add_argument('--G_norm_3D', type=str, default='batch3d', help='normalization layer for G: inst3d | batch3d | none')
        parser.add_argument('--D_norm_3D', type=str, default='none', help='normalization layer for D: inst3d | batch3d | none')
        # number of channels in our networks
        parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in the first conv layer')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--ngf_3d', type=int, default=64, help='# of 3D gen filters in the last conv layer')
        parser.add_argument('--ndf_3d', type=int, default=64, help='# of 3D discrim filters in the first conv layer')
        # extra parameters
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='dcgan | lsgan | wgangp | hinge')  # for 2D texture network; not for 3D; use gan_mode_3D for 3D shape
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization: normal | xavier | kaiming | orth')
        parser.add_argument('--init_param', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # 3D paramters:
        parser.add_argument('--voxel_res', type=int, default=128, help='the resolution of voxelized data')
        parser.add_argument('--class_3d', type=str, default='car', choices=['car', 'chair'], help='3d model class')
        parser.add_argument('--model3D_dir', type=str, default=None, help='directory to store pretrained 3D model')
        parser.add_argument('--model2D_dir', type=str, default=None, help='directory to store pretrained 2D model')
        parser.add_argument('--use_df', action='store_true', help='use distance function (DF) representation')
        parser.add_argument('--df_th', type=float, default=0.90, help='threshold for rendering distance function (DF)')
        # misc
        parser.add_argument('--no_largest', action='store_true', help='disable using the largest connected component during rendering')
        parser.add_argument('--crop_align', action='store_true', help='if the croping is aligned between real and fake')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: e.g., {netD}_{netG}_voxel{voxel_res}')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--print_grad', action='store_true', help='if print grad for 2D and 3D gan loss')
        parser.add_argument('--seed', type=int, default=0, help='seed')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if self.isTrain:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
            pkl_file = os.path.join(expr_dir, 'train_opt.pkl')
            pickle.dump(opt, open(pkl_file, 'wb'))
        else:
            util.mkdirs(opt.results_dir)

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        if opt.suffix:
            opt.name = (opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                opt.gpu_ids.append(int_id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
