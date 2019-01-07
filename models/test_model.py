from .base_model import BaseModel
import numpy as np
import torch


class TestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.vae = True
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['mask', 'depth', 'image']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_AB', 'G_3D']
        self.model_names += 'E'
        self.cuda_names = ['z_shape', 'z_texture', 'rot_mat']
        self.use_df = opt.use_df or opt.dataset_mode.find('df') >= 0

        self.netG_3D = self.define_G_3D()
        self.netG_AB = self.define_G(opt.input_nc, opt.output_nc, opt.nz_texture, ext='AB')
        self.netE = self.define_E(opt.output_nc, self.vae)
        self.is_loaded = True
        self.n_views = opt.n_views
        self.bs = opt.batch_size
        self.nz_shape = opt.nz_shape
        self.nz_texture = opt.nz_texture
        self.n_shapes = opt.n_shapes
        self.setup_DR(opt)
        self.bg_B = 1
        self.bg_A = -1
        self.random_view = opt.random_view
        self.interp_shape = opt.interp_shape
        self.interp_texture = opt.interp_texture
        self.critCycle = torch.nn.L1Loss().to(self.device)
        with torch.no_grad():
            self.z0_s = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
            self.z1_s = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
            self.z0_t = self.get_z_random(self.bs, self.nz_texture).view(self.bs, self.nz_texture, 1, 1, 1)
            self.z1_t = self.get_z_random(self.bs, self.nz_texture).view(self.bs, self.nz_texture, 1, 1, 1)

        self.count = 0

    def set_input(self, input, reset_shape=False, reset_texture=False):
        self.input_B = input[0]['image'].to(self.device)
        self.mask_B = input[0]['real_im_mask'].to(self.device)
        if reset_shape or not hasattr(self, 'voxel'):
            self.voxel = input[1]['voxel'].to(self.device)
        if reset_texture or not hasattr(self, 'z_texture'):
            with torch.no_grad():
                self.z_texture, mu, var = self.encode(self.input_B, vae=self.vae)
                self.z_texture = mu

    def set_posepool(self, posepool):
        self.posepool = posepool
        self.total_views = posepool.shape[0]
        if not self.random_view:
            self.elevation = (posepool[:, 0] // 3) * 3
            self.azimuth = (posepool[:, 1] // 5) * 5
            views = np.zeros((self.n_views, 2))
            hist, bins = np.histogram(self.azimuth, bins=range(-90, 91, 5))
            sort_ids = hist.argsort()[::-1][: self.n_views]
            top_a = bins[sort_ids]
            top_a.sort()
            for n, az in enumerate(top_a):
                ids = np.where(self.azimuth == az)
                ele = self.elevation[ids]
                values, counts = np.unique(ele, return_counts=True)
                counts[0] = 0
                id2 = np.argmax(counts)
                views[n, 0] = values[id2]
                views[n, 1] = az
            self.views = views

    def reset_shape(self, reset=True):
        if reset or not hasattr(self, 'z_shape'):
            if self.interp_shape:
                alpha = self.count / float(self.n_shapes)
                self.z_shape = (1 - alpha) * self.z0_s + alpha * self.z1_s
            else:
                self.z_shape = self.get_z_random(self.bs, self.nz_shape).view(self.bs, self.nz_shape, 1, 1, 1)
            self.z_shape = self.z_shape.to(self.device)

    def reset_texture(self, reset=True):
        if reset or not hasattr(self, 'z_texture'):
            if self.interp_texture:
                alpha = self.count / float(self.n_shapes)
                self.z_texture = (1 - alpha) * self.z0_t + alpha * self.z1_t
            else:
                self.z_texture = self.get_z_random(self.bs, self.nz_texture)

            self.z_texture = self.z_texture.to(self.device)

    def reset_view(self, reset=False):
        if self.random_view:
            if reset or not hasattr(self, 'views'):
                rand_ids = np.random.randint(self.total_views, size=self.n_views)
                self.views = self.posepool[rand_ids, :]

    def sample_3d(self):
        with torch.no_grad():
            self.voxel = self.netG_3D(self.z_shape)

    def sample_2d(self, view_id, extra=False):
        assert view_id >= 0 and view_id <= self.n_views
        view = self.views[view_id, :]
        with torch.no_grad():
            self.rot_mat = self.azele2matrix(az=view[1], ele=view[0]).unsqueeze(0).repeat(self.bs, 1, 1)
            self.rot_mat = self.rot_mat.to(self.device)
            self.mask, self.depth = self.get_depth(self.voxel, self.rot_mat, use_df=self.use_df)
            self.image = self.apply_mask(self.netG_AB(self.depth, self.z_texture), self.mask, self.bg_B)
        if extra:
            return self.image, self.mask, self.depth
        else:
            return self.image

    @staticmethod
    def azele2matrix(az=0, ele=0):
        R0 = torch.zeros([3, 3])
        R = torch.zeros([3, 4])
        R0[0, 1] = 1
        R0[1, 0] = -1
        R0[2, 2] = 1
        az = az * np.pi / 180
        ele = ele * np.pi / 180
        cos = np.cos
        sin = np.sin
        R_ele = torch.FloatTensor(
            [[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]])
        R_az = torch.FloatTensor(
            [[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R_rot = torch.mm(R_az, R_ele)
        R_all = torch.mm(R_rot, R0)
        R[:3, :3] = R_all
        return R

    def update_D(self):
        pass

    def update_G(self):
        pass
