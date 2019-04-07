from .base_model import BaseModel
from .networks import _cal_kl, GANLoss
from util.image_pool import ImagePool
import itertools
import torch
from .networks_3d import _calc_grad_penalty


class TextureRealModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_cycle_A', type=float, default=10.0, help='weight for forward cycle')
        parser.add_argument('--lambda_cycle_B', type=float, default=25.0, help='weight for backward cycle')
        parser.add_argument('--lambda_z', type=float, default=1.0, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl_real', type=float, default=0.001, help='weight for KL loss, real')
        parser.add_argument('--lambda_mask', type=float, default=2.5, help='mask loss')
        parser.add_argument('--gp_norm', type=float, default=1.0, help='WGANGP gradient penality norm')
        parser.add_argument('--gp_type', type=str, default='mixed', help='WGANGP graident penalty type')
        parser.add_argument('--lambda_gp', type=float, default=10, help='WGANGP gradient penality coefficient')

        return parser

    def __init__(self, opt, base_init=True):
        assert opt.input_nc == 1 and opt.output_nc == 3
        if base_init:
            BaseModel.__init__(self, opt)
        self.nz_texture = opt.nz_texture
        self.use_df = opt.use_df or opt.dataset_mode.find('df') >= 0
        self.vae = opt.lambda_kl_real > 0.0
        self.bg_B = 1
        self.bg_A = -1
        if self.isTrain:
            self.model_names += ['G_AB', 'G_BA', 'D_A', 'D_B', 'E']
        else:
            self.model_names += ['G_AB', 'G_BA', 'E']

        # load/define networks: define G
        self.netG_AB = self.define_G(opt.input_nc, opt.output_nc, opt.nz_texture, ext='AB')
        self.netG_BA = self.define_G(opt.output_nc, opt.input_nc, 0, ext='BA')
        self.netE = self.define_E(opt.output_nc, self.vae)
        # define D
        if opt.isTrain:
            self.netD_A = self.define_D(opt.output_nc, ext='A')
            self.netD_B = self.define_D(opt.input_nc, ext='B')
        self.setup_DR(opt)
        self.visual_names += ['real_A', 'mask_A', 'fake_B', 'rec_A', 'real_B', 'mask_B', 'fake_A', 'rec_B']
        self.loss_names += ['G', 'G_AB', 'G_BA', 'cycle_A', 'cycle_B', 'cycle_z', 'D_A', 'D_B']
        self.cuda_names += ['input_B', 'rot_mat', 'mask_B', 'voxel_real', 'z_texture']
        if opt.gan_mode == 'wgangp':
            self.loss_names += ['gp_A', 'gp_B']

        if opt.lambda_kl_real > 0.0:
            self.loss_names += ['kl_real', 'mu_enc', 'var_enc']
        if opt.lambda_mask > 0.0:
            self.loss_names += ['mask_B']

        if opt.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.critGAN = GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.critCycle = torch.nn.L1Loss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters(), self.netE.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        self.input_B = input[0]['image']
        self.mask_B = input[0]['real_im_mask']
        self.rot_mat = input[0]['rotation_matrix']
        self.bs = self.input_B.size(0)
        self.is_skip = self.bs < self.opt.batch_size
        if self.is_skip:
            return
        self.z_texture = self.get_z_random(self.bs, self.nz_texture)
        self.voxel_real = input[1]['voxel']
        self.move_to_cuda()
        mask_A_full, real_A_full = self.get_depth(self.voxel_real, self.rot_mat, use_df=self.use_df)
        self.mask_A, self.real_A, self.mask_B, self.real_B = self.crop_image(mask_A_full, real_A_full, self.mask_B, self.input_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B, self.loss_gp_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A, self.loss_gp_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_basic(self, netD, real, fake):
        loss_D_real = self.critGAN(netD(real.detach()), True)  # real
        loss_D_fake = self.critGAN(netD(fake.detach()), False)  # fake
        loss_D = loss_D_real + loss_D_fake  # combined loss
        loss_D.backward()  # backward

        if self.opt.gan_mode == 'wgangp':
            loss_gp, _ = _calc_grad_penalty(netD, real, fake.detach(), self.device, type='mixed', constant=self.opt.gp_norm, ll=self.opt.lambda_gp)
            loss_gp.backward(retain_graph=True)
        else:
            loss_gp = 0.0

        return loss_D, loss_gp

    def backward_GE(self):
        # GAN loss D_A(G_A(A))
        self.fake_B = self.apply_mask(self.netG_AB(self.real_A, self.z_texture), self.mask_A, self.bg_B)
        self.loss_G_AB = self.critGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.fake_A = self.apply_mask(self.netG_BA(self.real_B), self.mask_B, self.bg_A)
        self.loss_G_BA = self.critGAN(self.netD_B(self.fake_A), True)
        # Forward image cycle loss
        self.rec_A = self.apply_mask(self.netG_BA(self.fake_B), self.mask_A, self.bg_A)
        self.loss_cycle_A = self.critCycle(self.rec_A, self.real_A.detach()) * self.opt.lambda_cycle_A
        # Backward latent cycle loss
        self.z_encoded, mu1, logvar1 = self.encode(self.real_B, self.vae)
        if self.opt.lambda_kl_real > 0.0:
            self.loss_mu_enc = torch.mean(torch.abs(mu1))
            self.loss_var_enc = torch.mean(logvar1.exp())
        self.rec_B = self.apply_mask(self.netG_AB(self.fake_A, self.z_encoded), self.mask_B, self.bg_B)
        self.loss_cycle_B = self.critCycle(self.rec_B, self.real_B) * self.opt.lambda_cycle_B
        # latent cycle loss
        z_predict, mu2, logvar2 = self.encode(self.fake_B, self.vae)
        self.loss_cycle_z = self.critCycle(z_predict, self.z_texture) * self.opt.lambda_z
        self.loss_kl_real = _cal_kl(mu1, logvar1, self.opt.lambda_kl_real)
        # mask B consistency loss
        self.loss_mask_B = self.critCycle(self.fake_A, self.mask_B * 2 - 1) * self.opt.lambda_mask
        # combined loss
        self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_cycle_A + self.loss_cycle_B \
            + self.loss_cycle_z + self.loss_kl_real + self.loss_mask_B
        self.loss_G.backward()

    def update_D(self):
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def update_G(self):
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_GE()
        self.optimizer_G.step()
