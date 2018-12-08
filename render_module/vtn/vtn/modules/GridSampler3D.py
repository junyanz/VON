from torch import nn
from ..functions.grid_sample3d import grid_sample3d


class GridSampler3D(nn.Module):
    def forward(self, theta, size):
        return grid_sample3d(theta, size)
