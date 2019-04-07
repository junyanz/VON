import os
import torch
from . import networks, networks_3d
from collections import OrderedDict
from abc import ABC, abstractmethod
from .basics import get_scheduler
import numpy as np


class BaseModel(ABC):
    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.use_cuda = len(self.gpu_ids) > 0
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.manual_seed(opt.seed)
        self.is_loaded = False
        self.model_names = []
        self.loss_names = []
        self.visual_names = []
        self.cuda_names = []
        self.optimizers = []
        self.is_skip = False
        if opt.resize_or_crop != 'scale_width':
            print('enable cudnn benchmark')
            torch.backends.cudnn.benchmark = True

    def deduplicate_names(self):
        self.model_names = list(set(self.model_names))
        self.loss_names = list(set(self.loss_names))
        # self.visual_names = list(set(self.visual_names))
        self.cuda_names = list(set(self.cuda_names))

    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if (not self.isTrain or opt.continue_train) and not self.is_loaded:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

    def define_G_3D(self):
        opt = self.opt
        netG = networks_3d.define_G_3D(nz=opt.nz_shape, res=opt.voxel_res, model=opt.netG_3D,
                                       ngf=opt.ngf_3d, norm=opt.G_norm_3D,
                                       init_type=opt.init_type, init_param=opt.init_param, gpu_ids=opt.gpu_ids)
        if opt.model3D_dir:
            self.load_network(netG, opt.model3D_dir + '_G_3D.pth')
        return netG

    def define_D_3D(self):
        opt = self.opt
        netD = networks_3d.define_D_3D(res=opt.voxel_res, model=opt.netD_3D,
                                       ndf=opt.ndf_3d, norm=opt.D_norm_3D,
                                       init_type=opt.init_type, init_param=opt.init_param, gpu_ids=opt.gpu_ids)
        if opt.model3D_dir:
            self.load_network(netD, opt.model3D_dir + '_D_3D.pth')
        return netD

    def define_G(self, input_nc, output_nc, nz, ext=''):
        opt = self.opt
        netG = networks.define_G(input_nc, output_nc, nz, opt.ngf,
                                 model=opt.netG, crop_size=opt.crop_size,
                                 norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_param=opt.init_param,
                                 gpu_ids=self.gpu_ids, where_add=self.opt.where_add)

        if opt.model2D_dir:
            self.load_network(netG, opt.model2D_dir + '_net_G_%s.pth' % ext)
        return netG

    def define_D(self, input_nc, ext=''):
        opt = self.opt
        netD = networks.define_D(input_nc, opt.ndf,
                                 model=opt.netD, crop_size=opt.crop_size,
                                 norm=opt.norm, nl=opt.nl, init_type=opt.init_type,
                                 init_param=opt.init_param, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        # if opt.model2D_dir: # skip loading Ds
        # self.load_network(netD, opt.model2D_dir + '_net_D_%s.pth' % ext)
        return netD

    def define_E(self, input_nc, vae):
        opt = self.opt
        netE = networks.define_E(input_nc, opt.nz_texture, opt.nef,
                                 model=opt.netE, crop_size=opt.crop_size,
                                 norm=opt.norm, nl=opt.nl,
                                 init_type=opt.init_type, gpu_ids=self.gpu_ids,
                                 vae=vae)
        if opt.model2D_dir:
            self.load_network(netE, opt.model2D_dir + '_net_E.pth')
        return netE

    def define_E_all_z(self, input_nc, output_list, vae_list):
        opt = self.opt
        netE = networks.define_E_all_z(input_nc, output_list, opt.nef,
                                       model=opt.netE_all_z, crop_size=opt.crop_size,
                                       norm=opt.norm, nl=opt.nl,
                                       init_type=opt.init_type, gpu_ids=self.gpu_ids, vae_list=vae_list)
        if opt.model2D_dir:
            self.load_network(netE, opt.model2D_dir + '_net_E_all_z.pth', notfound_ok=True)
        return netE

    @abstractmethod
    def update_D(self):
        pass

    @abstractmethod
    def update_G(self):
        pass

    def encode(self, input_image, vae=False):
        if vae:
            mu, logvar = self.netE(input_image)
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            return eps.mul(std).add_(mu), mu, logvar
        else:
            z = self.netE(input_image)
            return z, None, None

    def load_network(self, net, path, notfound_ok=False):
        if os.path.exists(path):
            print('loading model from %s' % path)
            net.module.load_state_dict(torch.load(path))
        else:
            if notfound_ok:
                print('Warning: network file %s not found. starting from scratch' % path)
            else:
                raise ValueError('Network file %s not found.' % path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(None)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def apply_mask(self, input_image, mask, bg_color):
        output = mask * input_image + (1 - mask) * bg_color
        return output

    def setup_DR(self, opt):
        from render_module.render_sketch import VoxelRenderLayer, CroppingLayer, GetRotationMatrix, FineSizeCroppingLayer
        self.angles_2_rotmat = GetRotationMatrix()
        vsize = opt.load_size
        voxel_shape = torch.Size([opt.batch_size, 1, vsize, vsize, vsize])
        self.renderLayer = VoxelRenderLayer(voxel_shape, res=vsize, nsamples_factor=2.5, camera_distance=2)
        self.croppinglayer = CroppingLayer(output_size=vsize, no_largest=opt.no_largest)
        self.fineCropingLayer = FineSizeCroppingLayer(opt.crop_size)
        self.angles_2_rotmat.to(self.device)
        self.renderLayer.to(self.device)
        self.croppinglayer.to(self.device)
        self.fineCropingLayer.to(self.device)

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def crop_image(self, mask_A_full, real_A_full, mask_B_full, real_B_full):
        if not self.isTrain:
            return mask_A_full, real_A_full, mask_B_full, real_B_full
        random_A = np.random.random()
        random_B = random_A if self.opt.crop_align else np.random.random()
        mask_A = self.fineCropingLayer(mask_A_full, random_A)
        real_A = self.fineCropingLayer(real_A_full, random_A)
        mask_B = self.fineCropingLayer(mask_B_full, random_B)
        real_B = self.fineCropingLayer(real_B_full, random_B)
        return mask_A, real_A, mask_B, real_B

    # testing models
    @abstractmethod
    def set_input(self, input):
        pass

    def get_image_paths(self):
        return self.image_paths

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses. train.py will print out these losses as debugging information
    def get_current_losses(self):
        losses_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                var = getattr(self, 'loss_' + name)
                if hasattr(var, 'requires_grad'):
                    if var.requires_grad:
                        var = var.item()
                losses_ret[name] = var
        return losses_ret

    def check_nan_inf(self):
        losses = self.get_current_losses()
        for k, v in losses.items():
            if np.isnan(v):
                print('%s is nan!' % k)
            elif np.isinf(v):
                print('%s is inf!' % k)
            else:
                continue

    def clear_running_mean(self):
        for name in self.loss_names:
            self._safe_set('loss_' + name, 0.0)
            self._safe_set('count_' + name, 0)

    def _safe_set(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # save models nto the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def get_depth(self, voxel, rot_mat, use_df=False, flipped=False):
        if use_df:
            voxel = torch.where(voxel < self.opt.df_th, torch.tensor(0.0).to(self.device), voxel)
        if flipped:
            voxel = voxel.transpose(3, 4)
            voxel = torch.flip(voxel, [3])
        silhouette_orig, depth_orig = self.renderLayer(voxel, rot_mat)
        self.sil_orig = silhouette_orig
        self.depth_orig = depth_orig
        silhouette, depth = self.croppinglayer(silhouette_orig, depth_orig)
        min_depth = depth.data.min()
        max_depth = depth.data.max()
        # normalize the depth to [-1, 1]
        depth2 = 1 - (depth - min_depth) / (max_depth - min_depth) * 2
        return silhouette, depth2

    def move_to_cuda(self, gpu_idx=0):
        for name in self.cuda_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.to(self.device))

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[%s] Total #parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def skip(self):
        return self.is_skip

    @staticmethod
    def accumulate_loss(*args):
        'safely ignore None and non-tensor values'
        v = 0
        for arg in args:
            if arg is not None and hasattr(arg, 'requires_grad'):
                v += arg
        return v

    def safe_render(self, netG, batch_size, nz, flipped=None):
        success = False
        MAX_RETRY = 10
        cnt = 0
        while not success and cnt < MAX_RETRY:
            try:
                z_shape = self.get_z_random(batch_size, nz).view(batch_size, nz, 1, 1, 1).to(self.device)
                voxel = netG(z_shape)
                if self.opt.print_grad:
                    voxel.register_hook(self.grad_hook_gen('2d_grad'))
                mask_A_full, real_A_full = self.get_depth(voxel, self.rot_mat, use_df=self.use_df, flipped=self.use_df if flipped is None else flipped)
                if torch.isnan(mask_A_full).any() or torch.isnan(real_A_full).any():
                    raise ValueError('nan found in rendered mask and depth')
                    success = False
                else:
                    success = True
            except Exception as e:
                if cnt >= MAX_RETRY:
                    print(e)
                    raise RuntimeError('Maximum number of retries reached.')
                cnt += 1
                print(e)
                print('Retry sampling %02d/%02d' % (cnt, MAX_RETRY))
                del voxel
        if success:
            return mask_A_full, real_A_full, z_shape
        else:
            return None, None, None
