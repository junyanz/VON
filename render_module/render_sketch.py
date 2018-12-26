try:
    from .vtn.vtn.functions import grid_sample3d, affine_grid3d
except ImportError:
    from vtn.vtn.functions import grid_sample3d, affine_grid3d
try:
    from .calc_prob.calc_prob.functions.calc_prob import CalcStopProb
except ImportError:
    from calc_prob.calc_prob.functions.calc_prob import CalcStopProb
from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from scipy import ndimage


def azele2matrix(az=0, ele=0):
    R0 = torch.zeros([3, 3])
    R0[0, 1] = 1
    R0[1, 0] = -1
    R0[2, 2] = 1
    az = az * np.pi / 180
    ele = ele * np.pi / 180
    cos = np.cos
    sin = np.sin
    R_ele = torch.from_numpy(np.array([[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]])).float()
    R_az = torch.from_numpy(np.array([[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])).float()
    R_rot = torch.mm(R_az, R_ele)
    R_all = torch.mm(R_rot, R0)
    return R_all


class GetRotationMatrix(nn.Module):
    def __init__(self, az_min=-np.pi / 2, az_max=np.pi / 2, ele_min=0, ele_max=2 * np.pi / 9):
        super().__init__()
        self.az_max = az_max
        self.az_min = az_min
        self.ele_max = ele_max
        self.ele_min = ele_min

    def forward(self, angles_in):
        is_cuda = angles_in.is_cuda
        assert(angles_in.shape[1] == 2)
        bn = angles_in.shape[0]
        az_in = angles_in[:, 0]
        ele_in = angles_in[:, 1]
        az_in = torch.clamp(az_in, self.az_min, self.az_max)
        ele_in = torch.clamp(ele_in, self.ele_min, self.ele_max)
        az_sin = torch.sin(az_in)
        az_cos = torch.cos(az_in)
        ele_sin = torch.sin(ele_in)
        ele_cos = torch.cos(ele_in)
        R_az = self.create_Raz(az_cos, az_sin)
        R_ele = self.create_Rele(ele_cos, ele_sin)
        # print(R_ele)
        R_rot = torch.bmm(R_az, R_ele)
        R_0 = angles_in.data.new(
            bn, 3, 3).zero_()
        R_0[:, 0, 1] = 1
        R_0[:, 1, 0] = -1
        R_0[:, 2, 2] = 1
        R_0 = R_0.requires_grad_(True)
        if is_cuda:
            R_0 = R_0.cuda()
        R = torch.bmm(R_rot, R_0)
        zeros = angles_in.data.new_zeros([bn, 3, 1]).zero_().requires_grad_(True)
        return torch.cat((R, zeros), dim=2)

    def create_Rele(self, ele_cos, ele_sin):
        bn = ele_cos.shape[0]
        one = Variable(ele_cos.data.new(bn, 1, 1).fill_(1))
        zero = Variable(ele_cos.data.new(bn, 1, 1).zero_())
        ele_cos = ele_cos.view(bn, 1, 1)
        ele_sin = ele_sin.view(bn, 1, 1)
        c1 = torch.cat((one, zero, zero), dim=1)
        c2 = torch.cat((zero, ele_cos, ele_sin), dim=1)
        c3 = torch.cat((zero, -ele_sin, ele_cos), dim=1)
        return torch.cat((c1, c2, c3), dim=2)

    def create_Raz(self, cos, sin):
        bn = cos.shape[0]
        one = Variable(cos.data.new(bn, 1, 1).fill_(1))
        zero = Variable(cos.data.new(bn, 1, 1).zero_())
        cos = cos.view(bn, 1, 1)
        sin = sin.view(bn, 1, 1)
        c1 = torch.cat((cos, sin, zero), dim=1)
        c2 = torch.cat((-sin, cos, zero), dim=1)
        c3 = torch.cat((zero, zero, one), dim=1)
        return torch.cat((c1, c2, c3), dim=2)


class FineSizeCroppingLayer(nn.Module):
    '''
    crop the input list of images to specified size.
    '''

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, random_number):
        '''
        random number is from 0 to 1
        '''
        N, C, H, W = x.shape
        output_size = self.output_size
        assert(H >= output_size and W >= output_size)
        h_beg = round((H - output_size) * random_number)
        w_beg = round((W - output_size) * random_number)
        return x[:, :, h_beg:h_beg + output_size, w_beg:w_beg + output_size]


class CroppingLayer(nn.Module):
    '''
    crop and pad output to have consistant size
    '''

    def __init__(self, output_size, sil_th=0.8, padding_pct=0.03, no_largest=False):
        super().__init__()
        self.output_size = output_size
        self.threshold = nn.Threshold(sil_th, 0)
        size = [output_size, output_size]
        h = torch.arange(0, size[0]).float() / (size[0] - 1.0) * 2.0 - 1.0
        w = torch.arange(0, size[1]).float() / (size[1] - 1.0) * 2.0 - 1.0
        # create grid
        grid = torch.zeros(size[0], size[1], 2)
        grid[:, :, 0] = w.unsqueeze(0).repeat(size[0], 1)
        grid[:, :, 1] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
        # expand to match batch size
        grid = grid.unsqueeze(0)
        self.register_buffer('grid', grid)
        self.kernel = np.ones((5, 5), np.uint8)
        self.padding_pct = padding_pct
        self.th = sil_th
        self.no_largest = no_largest

    def forward(self, exp_sil, exp_depth):

        sil, depth, _, _ = self.crop_depth_sil(exp_sil, exp_depth)
        return sil, depth

    def bbox_from_sil(self, exp_sil, padding_pct=0.03):
        n, c, h, w = exp_sil.shape
        assert c == 1  # sil only
        mask_th = exp_sil.data.cpu().numpy()
        # find largest connected component:
        mask_th_binary = np.where(mask_th < self.th, 0.0, 1.0)
        # lefttop_h, lefttop_w, rightbottom_h, rightbottom_w
        bbox = np.zeros([n, 4]).astype(int)
        mask_largest_batch = torch.FloatTensor(n, c, h, w).zero_()
        for x in range(n):
            if self.no_largest:
                nz = np.nonzero(mask_th_binary[x, 0, :, :])
                bbox[x, 0] = np.min(nz[0])
                bbox[x, 1] = np.min(nz[1])
                bbox[x, 2] = np.max(nz[0])
                bbox[x, 3] = np.max(nz[1])
                mask_largest_batch[x, 0, :, :] = torch.from_numpy(mask_th_binary[x, 0, :, :].astype(np.float32)).float()
            else:
                mask_th_binary_pad = np.pad(mask_th_binary[x, 0, :, :], ((1, 1),), 'constant', constant_values=0).astype(np.uint8)
                labeled, nr_objects = ndimage.measurements.label(mask_th_binary_pad)
                counts = np.bincount(labeled.flatten())
                largest = np.argmax(counts[1:]) + 1
                mask_largest = np.where(labeled == largest, 1, 0)[1:-1, 1:-1]
                mask_largest_batch[x, 0, :, :] = torch.from_numpy(mask_largest.astype(np.float32)).float()
                # mask_th2 = cv2.morphologyEx(mask_th[x], cv2.MORPH_OPEN, self.kernel)
                # nz = np.nonzero(mask_th2[0, :, :])
                nz = np.nonzero(mask_largest)
                bbox[x, 0] = np.min(nz[0])
                bbox[x, 1] = np.min(nz[1])
                bbox[x, 2] = np.max(nz[0])
                bbox[x, 3] = np.max(nz[1])

        return bbox, mask_largest_batch

    def crop_depth_sil(self, exp_sil_full, exp_depth_full, is_debug=False):
        # also keeps track of coordinate change
        # output a

        bbox, mask_largest_batch = self.bbox_from_sil(exp_sil_full)
        mask_largest_batch = exp_sil_full.new_tensor(mask_largest_batch)

        bbox = bbox.astype(np.int32)
        exp_sil = exp_sil_full * mask_largest_batch
        exp_depth = exp_depth_full * mask_largest_batch
        exp_depth = exp_depth + 3 * (1 - mask_largest_batch)
        n, c, h, w = exp_sil.shape
        new_sil = []
        new_depth = []

        shape_stat = []
        for x in range(n):
            h = bbox[x, 2] + 1 - bbox[x, 0]
            w = bbox[x, 3] + 1 - bbox[x, 1]
            h = int(h)
            w = int(w)
            cropped_sil = exp_sil[x, 0, bbox[x, 0]:bbox[x, 0] + h,
                                  bbox[x, 1]:bbox[x, 1] + w]
            cropped_sil = cropped_sil.contiguous().view(1, 1, h, w)
            cropped_depth = exp_depth[x, 0, bbox[x, 0]:bbox[x, 0] +
                                      h, bbox[x, 1]:bbox[x, 1] + w]
            cropped_depth = cropped_depth.contiguous().view(1, 1, h, w)
            if h > w:
                dim = h
                m_sil = nn.ConstantPad2d(((h - w) // 2, (h - w) // 2, 0, 0), 0)
                m_depth = nn.ConstantPad2d(
                    ((h - w) // 2, (h - w) // 2, 0, 0), 3)
            else:
                dim = w
                m_sil = nn.ConstantPad2d((0, 0, (w - h) // 2, (w - h) // 2), 0)
                m_depth = nn.ConstantPad2d(
                    (0, 0, (w - h) // 2, (w - h) // 2), 3)
            pad = int(np.floor(dim * self.padding_pct))
            space_pad_depth = nn.ConstantPad2d((pad, pad, pad, pad), 3)
            space_pad_sil = nn.ConstantPad2d((pad, pad, pad, pad), 0)
            sq_depth = space_pad_depth(m_depth(cropped_depth))
            sq_sil = space_pad_sil(m_sil(cropped_sil))
            shape_stat.append(sq_depth.shape)
            new_sil.append(grid_sample(sq_sil, self.grid))
            new_depth.append(grid_sample(sq_depth, self.grid))
        new_sil = torch.cat(new_sil, dim=0)
        new_depth = torch.cat(new_depth, dim=0)
        if is_debug:
            return sq_sil, cropped_depth, bbox, sq_depth.shape
        else:
            return new_sil, new_depth, bbox, shape_stat


class VoxelRenderLayer(nn.Module):
    def __init__(self, voxel_shape, camera_distance=2.0, fl=0.050, w=0.0612, res=128, nsamples_factor=1.5):
        super().__init__()
        self.camera_distance = camera_distance
        self.fl = fl
        self.w = w
        self.voxel_shape = voxel_shape
        self.nsamples_factor = nsamples_factor
        self.res = res
        self.register_buffer('grid', self.grid_gen())
        self.grid_sampler3d = grid_sample3d
        self.calc_stop_prob = CalcStopProb().apply
        self.affine_grid3d = affine_grid3d

    def forward(self, voxel_in, rotation_matrix=None):
        if rotation_matrix is None:
            voxel_rot = voxel_in
        else:
            voxel_rot_grid = self.affine_grid3d(
                rotation_matrix, voxel_in.shape)
            voxel_rot = self.grid_sampler3d(voxel_in, voxel_rot_grid)
        voxel_align = self.grid_sampler3d(voxel_rot, self.grid)
        voxel_align = voxel_align.permute(0, 1, 3, 4, 2)
        voxel_align = torch.clamp(voxel_align, 1e-4, 1 - (1e-4))
        voxel_align = voxel_align.contiguous()
        stop_prob = self.calc_stop_prob(voxel_align)
        exp_depth = torch.matmul(
            stop_prob, self.depth_weight)
        back_groud_prob = torch.prod((1.0) - voxel_align, dim=4)
        back_groud_prob = torch.clamp(back_groud_prob, 1e-4, 1 - (1e-4))
        back_groud_prob = back_groud_prob * (self.camera_distance + 1.0)
        exp_depth = exp_depth + back_groud_prob
        exp_sil = torch.sum(stop_prob, dim=4)
        return torch.transpose(exp_sil, 2, 3), torch.transpose(exp_depth, 2, 3)

    def grid_gen(self, numtype=np.float32):
        n, c, sx, sy, sz = self.voxel_shape
        nsamples = int(sz * self.nsamples_factor)
        res = self.res
        w = self.w
        dist = self.camera_distance
        self.register_buffer('depth_weight', torch.linspace(
            dist - 1, dist + 1, nsamples))
        fl = self.fl
        grid = np.zeros([n, nsamples, res, res, 3], dtype=numtype)
        h_linspace = np.linspace(w / 2, -w / 2, res)
        w_linspace = np.linspace(w / 2, -w / 2, res)
        H, W = np.meshgrid(h_linspace, w_linspace)
        cam = np.array([[[-dist, 0, 0]]])
        grid_vec = np.zeros([res, res, 3], dtype=numtype)
        grid_vec[:, :, 1] = W
        grid_vec[:, :, 2] = H
        grid_vec[:, :, 0] = -(dist - fl)
        grid_vec = grid_vec - cam
        self.grid_vec = grid_vec
        grid_vec_a = grid_vec * ((dist - 1) / fl)
        grid_vec_b = grid_vec * ((dist + 1) / fl)
        for idn in range(n):
            for ids in range(nsamples):
                grid[idn, ids, :, :, :] = grid_vec_b - \
                    (1 - (ids / nsamples)) * (grid_vec_b - grid_vec_a)
        grid = grid + cam
        return torch.from_numpy(grid.astype(numtype))
