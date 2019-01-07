from .texture_real_model import TextureRealModel
from .shape_gan_model import ShapeGANModel


class FullModel(TextureRealModel, ShapeGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        TextureRealModel.modify_commandline_options(parser, is_train)
        ShapeGANModel.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        TextureRealModel.__init__(self, opt, base_init=True)
        ShapeGANModel.__init__(self, opt, base_init=False)
        if opt.lambda_GAN_3D == 0.0:
            self.loss_names = [x for x in self.loss_names if '3D' not in x]
        if opt.print_grad:
            self.loss_names += ['2d_grad', '3d_grad']
            self.loss_2d_grad = 0.0
            self.loss_3d_grad = 0.0
            self.grad_hook_gen, self.grad_stats = self.dict_grad_hook_factory(add_func=lambda x: {'mean': x.data.mean(), 'std': x.data.std()})

    def set_input(self, input):
        self.input_B = input[0]['image']
        self.mask_B = input[0]['real_im_mask']
        self.rot_mat = input[0]['rotation_matrix']
        self.bs = self.input_B.size(0)
        self.z_texture = self.get_z_random(self.bs, self.nz_texture)
        self.z_shape = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
        self.voxel_real = input[1]['voxel']
        self.move_to_cuda()

        mask_A_full, real_A_full, _ = self.safe_render(self.netG_3D, self.bs, self.nz_shape, flipped=False)

        self.mask_A, self.real_A, self.mask_B, self.real_B = self.crop_image(mask_A_full, real_A_full, self.mask_B, self.input_B)

    def update_D(self):
        TextureRealModel.update_D(self)
        if self.opt.lambda_GAN_3D > 0.0:
            ShapeGANModel.update_D(self)

    def update_G(self):
        self.optimizer_G_3D.zero_grad()
        TextureRealModel.update_G(self)
        self.optimizer_G_3D.step()
        if self.opt.lambda_GAN_3D > 0.0:
            ShapeGANModel.update_G(self)
        if self.opt.print_grad:
            self.loss_2d_grad = self.grad_stats['2d_grad']['std'] * 1e3
            if self.opt.lambda_GAN_3D > 0.0:
                self.loss_3d_grad = self.grad_stats['3d_grad']['std'] * 1e3

    def save_networks(self, epoch):
        ShapeGANModel.save_networks(self, epoch)

    def sample(self, k=10, interp_traj=2, step=10):
        ShapeGANModel(self, k, interp_traj, step)
