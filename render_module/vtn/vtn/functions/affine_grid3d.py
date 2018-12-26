# functions/add.py
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class AffineGridGen3DFunction(Function):
    """
    Generate a 3D affine grid of size (batch*sz1*sz2*sz3*3)
    The affine grid is defined by a 3x4 matrix theta.
    The grid is initialized as a grid in [-1,1] in all dimensions,
    then transformed by matrix multiplication by theta.

    When theta is set to eye(3,4), the grid should match the original grid in a box.
    """
    @staticmethod
    def forward(ctx, theta, size):          # note that ctx is pytorch context
        assert type(size) == torch.Size
        assert len(size) == 5, 'Grid size should be specified by size of tensor to interpolate (5D)'
        assert theta.dim() == 3 and theta.size()[1:] == torch.Size([3, 4]), '3D affine transformation defined by a 3D matrix of batch*3*4'
        assert theta.size(0) == size[0], 'batch size mismatch'
        N, C, sz1, sz2, sz3 = size
        ctx.size = size
        ctx.is_cuda = theta.is_cuda
        theta = theta.contiguous()
        base_grid = theta.new(N, sz1, sz2, sz3, 4)
        linear_points = torch.linspace(-1, 1, sz1) if sz1 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, 0] = linear_points.view(1, -1, 1, 1).expand_as(base_grid[:, :, :, :, 0])
        linear_points = torch.linspace(-1, 1, sz2) if sz2 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, 1] = linear_points.view(1, 1, -1, 1).expand_as(base_grid[:, :, :, :, 1])
        linear_points = torch.linspace(-1, 1, sz3) if sz3 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, 2] = linear_points.view(1, 1, 1, -1).expand_as(base_grid[:, :, :, :, 2])
        base_grid[:, :, :, :, 3] = 1
        ctx.base_grid = base_grid

        grid = torch.bmm(base_grid.view(N, sz1 * sz2 * sz3, 4), theta.transpose(1, 2))
        grid = grid.view(N, sz1, sz2, sz3, 3)
        return grid

    @staticmethod
    @once_differentiable    # Used so that backward runs on tensors instead of variables. Note that this disables use of gradient of gradients.
    def backward(ctx, grad_output):
        N, C, sz1, sz2, sz3 = ctx.size
        assert grad_output.size() == torch.Size([N, sz1, sz2, sz3, 3])
        assert ctx.is_cuda == grad_output.is_cuda
        grad_output = grad_output.contiguous()
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, sz1 * sz2 * sz3, 4).transpose(1, 2),
            grad_output.view(N, sz1 * sz2 * sz3, 3))
        grad_theta = grad_theta.transpose(1, 2)         # actually one transpose would do, but we choose to follow pytorch imp here.
        return grad_theta, None


def affine_grid3d(theta, size):
    return AffineGridGen3DFunction.apply(theta, size)
