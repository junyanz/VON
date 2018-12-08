from torch import nn
from ..functions.affine_grid3d import affine_grid3d


class AffineGridGen3D(nn.Module):
    def forward(self, theta, size):
        return affine_grid3d(theta, size)
