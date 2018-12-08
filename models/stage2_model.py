import torch
from .stage2_real_model import Stage2RealModel


class Stage2Model(Stage2RealModel):
    def name(self):
        return 'Stage2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Stage2RealModel.modify_commandline_options(parser, is_train)
        return parser

    def initialize(self, opt, base_init=True):
        assert opt.input_nc == 1 and opt.output_nc == 3
        Stage2RealModel.initialize(self, opt, base_init)
        self.nz_shape = opt.nz_shape
        self.netG_3D = self.define_G_3D()
        self.netG_3D.eval()
        self.model_names.append('G_3D')
        self.cuda_names.append('z_shape')
        self.cuda_names.remove('voxel_real')
        self.deduplicate_names()

    def set_input(self, input):
        self.input_B = input[0]['image']
        self.mask_B = input[0]['real_im_mask']
        self.rot_mat = input[0]['rotation_matrix']
        self.bs = self.input_B.size(0)
        self.is_skip = self.bs < self.opt.batch_size
        if self.is_skip:
            return
        self.z_texture = self.get_z_random(self.bs, self.nz_texture)
        self.z_shape = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
        self.move_to_cuda()
        with torch.no_grad():
            mask_A_full, real_A_full, _ = self.safe_render(self.netG_3D, self.bs, self.nz_shape, flipped=False)
            self.mask_A, self.real_A, self.mask_B, self.real_B = self.crop_to_fine_size(mask_A_full, real_A_full, self.mask_B, self.input_B)
