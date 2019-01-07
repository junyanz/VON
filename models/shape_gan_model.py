import torch
from .base_model import BaseModel
from .networks_3d import _calc_grad_penalty
from .networks import GANLoss
import numpy as np
import os


class ShapeGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN_3D', type=float, default=1.0, help='GANLoss weight for end to end finetuning; set as 1.0 for shap generation; set as 0.05 for full model')
        parser.add_argument('--lambda_gp_3D', type=float, default=10, help='WGANGP gradient penality coefficient')
        parser.add_argument('--gan_mode_3D', type=str, default='wgangp', help='dcgan | lsgan | wgangp | hinge')
        parser.add_argument('--gp_norm_3D', type=float, default=1.0, help='WGANGP gradient penality norm')
        parser.add_argument('--gp_type_3D', type=str, default='mixed', help='WGANGP graident penalty type')
        parser.add_argument('--vis_batch_num', type=int, default=2, help='number of batch to visulize on epoch end')
        parser.add_argument('--lr_3d', type=float, default=0.0001, help='initial learning rate for adam')
        return parser

    def __init__(self, opt, base_init=True):
        if base_init:
            BaseModel.__init__(self, opt)
        self.loss_names += ['G_3D', 'D_real_3D', 'D_fake_3D', 'D_3D']
        self.nz_shape = opt.nz_shape
        if self.isTrain:
            self.model_names += ['D_3D']
            if opt.lambda_gp_3D > 0.0:
                self.loss_names += ['GP_3D']

        if 'G_3D' not in self.model_names:
            self.model_names += ['G_3D']
            self.netG_3D = self.define_G_3D()

        if self.isTrain:
            self.netD_3D = self.define_D_3D()

        if self.isTrain:
            self.critGAN_3D = GANLoss(gan_mode=opt.gan_mode_3D).to(self.device)
            self.optimizer_G_3D = torch.optim.Adam(self.netG_3D.parameters(), lr=opt.lr_3d, betas=(opt.beta1, 0.9))
            self.optimizer_D_3D = torch.optim.Adam(self.netD_3D.parameters(), lr=opt.lr_3d, betas=(opt.beta1, 0.9))
            self.optimizers += [self.optimizer_G_3D, self.optimizer_G_3D]
            for name in self.loss_names:
                setattr(self, 'loss_' + name, 0.0)
                setattr(self, 'count_' + name, 0)
        self.cuda_names += ['z_shape', 'voxel_real']
        self.deduplicate_names()

    def set_input(self, input):
        self.voxel_real = input['voxel'].to(self.device)
        if self.voxel_real.dim() == 4:
            self.voxel_real = self.voxel_real.unsqueeze(1)
        self.bs = self.voxel_real.shape[0]
        self.z_shape = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
        self.move_to_cuda()

    def _record_loss(self, name, value):
        loss = getattr(self, 'loss_' + name)
        count = getattr(self, 'count_' + name)
        v = loss * count / (count + 1)

        if type(value) != float:
            value = value.item()
        v += value / (count + 1)
        setattr(self, 'count_' + name, count + 1)
        setattr(self, 'loss_' + name, v)

    def backward_D_3D(self):
        self.z_shape.normal_(0, 1)
        with torch.no_grad():
            fake = self.netG_3D(self.z_shape)
        pred_fake = self.netD_3D(fake)
        pred_real = self.netD_3D(self.voxel_real)
        errD_fake = self.critGAN_3D(pred_fake, False)
        errD_real = self.critGAN_3D(pred_real, True)
        loss_GP, gradients = _calc_grad_penalty(self.netD_3D, self.voxel_real, fake.detach(), self.device, type=self.opt.gp_type_3D, constant=self.opt.gp_norm_3D,
                                                lambda_gp=self.opt.lambda_gp_3D * self.opt.lambda_GAN_3D)

        if type(loss_GP) != float:
            loss_GP.backward(retain_graph=True)
            self._record_loss('GP_3D', loss_GP)
        errD = (errD_fake + errD_real) * self.opt.lambda_GAN_3D
        errD.backward()
        self._record_loss('D_real_3D', errD_real)
        self._record_loss('D_fake_3D', errD_fake)
        self._record_loss('D_3D', errD_real + errD_fake)

    def backward_G_3D(self):
        self.z_shape.normal_(0, 1)
        fake = self.netG_3D(self.z_shape, return_stat=False)
        if self.opt.print_grad:  # HACK
            fake.register_hook(self.grad_hook_gen('3d_grad'))
        pred_fake = self.netD_3D(fake)
        errG = self.critGAN_3D(pred_fake, True) * self.opt.lambda_GAN_3D
        self._record_loss('G_3D', errG)
        errG.backward()

    def update_D(self):
        self.optimizer_D_3D.zero_grad()
        self.backward_D_3D()
        self.optimizer_D_3D.step()

    def update_G(self):
        self.set_requires_grad([self.netD_3D], False)
        self.optimizer_G_3D.zero_grad()
        self.backward_G_3D()
        self.optimizer_G_3D.step()
        self.set_requires_grad([self.netD_3D], True)

    def save_networks(self, epoch):
        BaseModel.save_networks(self, epoch)
        epoch_path = os.path.join(self.save_dir, 'epoch_%s' % epoch)
        random_vec_path = os.path.join(self.save_dir, 'random_vec.pt')
        os.makedirs(epoch_path, exist_ok=True)
        if not os.path.exists(random_vec_path):  # save random z vector
            noise = torch.zeros(self.opt.vis_batch_num, self.opt.batch_size, self.opt.nz_shape, 1, 1, 1).float().normal_()
            torch.save(noise, random_vec_path)
        else:
            noise = torch.load(random_vec_path)

        for batch_id in range(noise.shape[0]):   # save cached results
            batchz_shape = noise[batch_id, :, :, :, :, :].to(self.device)
            with torch.no_grad():
                pred = self.netG_3D(batchz_shape)
            batch_name = os.path.join(epoch_path, 'vali_%04d' % batch_id)
            np.savez(batch_name, pred=pred.cpu().numpy())

    def sample(self, k=10, interp_traj=2, step=10):
        if k % 2 != 0:
            k += 1
        noise = torch.zeros(2, self.opt.nz_shape, 1, 1, 1).float().normal_().to(self.device)
        self.netG_3D.eval()
        sample_shapes = []
        for idx in range(k // 2):
            with torch.no_grad():
                noise.normal_(0, 1)
                pred = self.netG_3D(noise)
                pred = pred.transpose(2, 3)
                pred = torch.flip(pred, [2])
                pred_numpy = pred.data.cpu().numpy()
                sample_shapes.append(pred_numpy[0, 0, :, :, :])
                sample_shapes.append(pred_numpy[1, 0, :, :, :])
        interp_traj_list = []
        noise_end = noise.clone()
        with torch.no_grad():
            for k in range(interp_traj // 2):
                noise_end.normal_(0, 1)
                noise.normal_(0, 1)
                traj_1 = []
                traj_2 = []
                for alpha in torch.linspace(0, 1, step, device=noise.device):
                    interpz_shape = alpha * noise + (1 - alpha * noise_end)
                    pred = self.netG_3D(interpz_shape)
                    pred = pred.transpose(2, 3)
                    pred = torch.flip(pred, [2])
                    pred_numpy = pred.data.cpu().numpy()
                    traj_1.append(pred_numpy[0, 0, :, :, :])
                    traj_2.append(pred_numpy[1, 0, :, :, :])
                interp_traj_list.append(traj_1[:])
                interp_traj_list.append(traj_2[:])
        return sample_shapes, interp_traj_list
