import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .. import utils

EPSILON = 1e-6


class BEVT_04(nn.Module):
    """
    Changes from BEVT:
        Condensed bottleneck along channel dimension,
        Increased dropout probability,
    """

    def __init__(self, in_height, z_max, z_min, cell_size):
        super().__init__()

        # [B, C, H, W] --> [B, C_condensed, H, W]
        self.conv_c = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
        )
        # [B, C_condensed, W, H] --> [B, C_condensed, W, 1]
        self.linear_v = nn.Sequential(
            nn.Linear(in_height, 1),
            nn.GroupNorm(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        # [B, 1, C_condensed, W] --> [B, Z, C_condensed, W]
        depth = (z_max - z_min) / cell_size

        self.z_extend = nn.Conv2d(1, int(depth), kernel_size=1)

        self.bev_expand_c = nn.Conv2d(32, 256, kernel_size=1)

        self.z_max = z_max
        self.z_min = z_min
        self.cell_size = cell_size

    def forward(self, features, calib, grid):
        # print('    BEV input shape:', features.shape)
        # Condense channel dimensions
        # [B, C, H, W] --> [B, C_cond, H, W]
        condensed_c = self.conv_c(features)

        # Reshape input tensor then condense input
        # features along the y dimension (height)
        # [B, C_cond, H, W] --> [B, C_cond, W, H] --> [B, C_cond, W, 1]
        bottleneck = self.linear_v(condensed_c.permute(0, 1, 3, 2))

        # condense collapsed v features along channels
        # [B, C, W, 1] --> [B, 1, W, C] --> [B, 1, W, C_condensed]
        # bottleneck = self.linear_c(condensed_h.permute((0, 3, 2, 1)))

        # Expand the bottleneck along the z dimension (depth)
        # [B, 1, C, W] --> [B, Z, C, W] --> [B, C, Z, W]
        bev_polar_feats = self.z_extend(bottleneck.permute(0, 3, 1, 2)).permute(
            0, 2, 1, 3
        )
        # bev_polar_feats_gn = self.gn_polar(bev_polar_feats)

        # print('     BEV polar feats shape:', bev_polar_feats.shape)

        # Normalise grid to [-1, 1]
        norm_grid = self.normalise_grid(grid, calib)

        # TODO compute features within voxels of 2D grid instead of at coordinates

        bev_cart_feats = F.grid_sample(bev_polar_feats, norm_grid, align_corners=True)

        bev_cart_feats_expand = self.bev_expand_c(bev_cart_feats)

        # print('     BEV cart feats shape:', bev_cart_feats.shape)
        return bev_cart_feats_expand

    def normalise_grid(self, grid, calib):
        """
        :param grid: BEV grid in with coords range
        [grid_h1, grid_h2] and [-grid_w/2, grid_w/2]
        :param calib:
        :return:
        """

        f, cu = calib[:, 0, 0], calib[:, 0, 2]
        batch_size = len(calib)

        # Compute positive x dimension at z_max and z_min
        # Computed x dimension is half grid width
        x_zmax = self.z_max / f * cu
        x_zmin = self.z_min / f * cu

        # Compute normalising constant for each row along the z-axis
        sample_res_z = (self.z_max - self.z_min) / self.cell_size
        sample_res_x = grid.shape[2] / self.cell_size

        norm_z = (
            2
            * (grid[..., 1] - grid[..., 1].min())
            / (grid[..., 1].max() - grid[..., 1].min())
            - 1
        )

        norm_scale_x = torch.stack(
            [
                torch.linspace(float(x_zmin[i]), float(x_zmax[i]), int(sample_res_z))
                for i in range(batch_size)
            ]
        )

        grid_ones = torch.ones_like(grid)
        grid_ones[..., 0] *= norm_scale_x.view(batch_size, -1, 1).cuda()

        # Normalise grid to [-1, 1]
        norm_grid = grid / grid_ones

        norm_grid[..., 1] = norm_z

        # print('     norm grid', norm_grid[...,0].max(), norm_grid[...,0].min(),
        #       norm_grid[...,1].max(), norm_grid[...,1].min())

        return norm_grid


class BEVT_H(nn.Module):
    def __init__(
        self,
        in_height,
        z_max,
        z_min,
        cell_size,
        additions_linear=False,
        additions_conv=False,
        kernel_h=9,
        stride_h=3,
        padding_h=4,
    ):
        super().__init__()

        self.horizontal = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=[3, kernel_h],
                stride=[1, stride_h],
                padding=[1, padding_h],
            ),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
        )

        # [B, C, W, H] --> [B, C, W, 1]
        if additions_linear:
            self.linear = nn.Sequential(
                nn.Linear(in_height, 1),
                nn.GroupNorm(16, 256),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            )
        else:
            self.linear = nn.Linear(in_height, 1)

        # [B, 1, C, W] --> [B, Z, C, W]
        depth = (z_max - z_min) / cell_size

        self.additions_conv = additions_conv
        if additions_conv:
            self.z_extend = nn.Sequential(
                nn.Conv2d(1, int(depth), kernel_size=1),
            )
            self.gn_polar = nn.GroupNorm(16, 256)
        else:
            self.z_extend = nn.Conv2d(1, int(depth), kernel_size=1)

        self.z_max = z_max
        self.z_min = z_min
        self.cell_size = cell_size

    def forward(self, features, calib, grid):

        # Convolve horizontally first
        features = self.horizontal(features)

        # Reshape input tensor then collapse input
        # features along the y dimension (height)
        # [B, C, H, W] --> [B, C, W, H] --> [B, C, W, 1]
        bottleneck = self.linear(features.permute(0, 1, 3, 2))

        # Expand the bottleneck along the z dimension (depth)
        # [B, 1, C, W] --> [B, Z, C, W] --> [B, C, Z, W]
        bev_polar_feats = self.z_extend(bottleneck.permute(0, 3, 1, 2)).permute(
            0, 2, 1, 3
        )
        # bev_polar_feats_gn = self.gn_polar(bev_polar_feats)

        # print('     BEV polar feats shape:', bev_polar_feats.shape)

        # Normalise grid to [-1, 1]
        norm_grid = self.normalise_grid(grid, calib)

        if self.additions_conv:
            bev_polar_feats_gn = self.gn_polar(bev_polar_feats)
            # Sample BEV polar features at grid locations
            # [B, C, Z, W] --> [B, C, Z, X]
            bev_cart_feats = F.grid_sample(
                bev_polar_feats_gn, norm_grid, align_corners=True
            )
            # print('     BEV cart feats shape:', bev_cart_feats.shape)
            return bev_cart_feats
        else:
            bev_cart_feats = F.grid_sample(
                bev_polar_feats, norm_grid, align_corners=True
            )
            # print('     BEV cart feats shape:', bev_cart_feats.shape)
            return bev_cart_feats

    def normalise_grid(self, grid, calib):
        """
        :param grid: BEV grid in with coords range
        [grid_h1, grid_h2] and [-grid_w/2, grid_w/2]
        :param calib:
        :return:
        """

        f, cu = calib[:, 0, 0], calib[:, 0, 2]
        batch_size = len(calib)

        # Compute positive x dimension at z_max and z_min
        # Computed x dimension is half grid width
        x_zmax = self.z_max / f * cu
        x_zmin = self.z_min / f * cu

        # Compute normalising constant for each row along the z-axis
        sample_res_z = (self.z_max - self.z_min) / self.cell_size
        sample_res_x = grid.shape[2] / self.cell_size

        norm_z = (
            2
            * (grid[..., 1] - grid[..., 1].min())
            / (grid[..., 1].max() - grid[..., 1].min())
            - 1
        )

        norm_scale_x = torch.stack(
            [
                torch.linspace(float(x_zmin[i]), float(x_zmax[i]), int(sample_res_z))
                for i in range(batch_size)
            ]
        )

        grid_ones = torch.ones_like(grid)
        grid_ones[..., 0] *= norm_scale_x.view(batch_size, -1, 1).cuda()

        # Normalise grid to [-1, 1]
        norm_grid = grid / grid_ones

        norm_grid[..., 1] = norm_z

        return norm_grid


class BEVT(nn.Module):
    def __init__(
        self,
        in_height,
        z_max,
        z_min,
        cell_size,
        additions_linear=False,
        additions_conv=False,
    ):
        super().__init__()

        # [B, C, W, H] --> [B, C, W, 1]
        if additions_linear:
            self.linear = nn.Sequential(
                nn.Linear(in_height, 1),
                nn.GroupNorm(16, 256),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            )
        else:
            self.linear = nn.Linear(in_height, 1)

        # [B, 1, C, W] --> [B, Z, C, W]
        depth = (z_max - z_min) / cell_size

        self.additions_conv = additions_conv
        if additions_conv:
            self.z_extend = nn.Sequential(
                nn.Conv2d(1, int(depth), kernel_size=1),
            )
            self.gn_polar = nn.GroupNorm(16, 256)
        else:
            self.z_extend = nn.Conv2d(1, int(depth), kernel_size=1)

        self.z_max = z_max
        self.z_min = z_min
        self.cell_size = cell_size

    def forward(self, features, calib, grid):
        # print('    BEV input shape:', features.shape)
        # Reshape input tensor then collapse input
        # features along the y dimension (height)
        # [B, C, H, W] --> [B, C, W, H] --> [B, C, W, 1]
        bottleneck = self.linear(features.permute(0, 1, 3, 2))

        # Expand the bottleneck along the z dimension (depth)
        # [B, 1, C, W] --> [B, Z, C, W] --> [B, C, Z, W]
        bev_polar_feats = self.z_extend(bottleneck.permute(0, 3, 1, 2)).permute(
            0, 2, 1, 3
        )
        # bev_polar_feats_gn = self.gn_polar(bev_polar_feats)

        # print('     BEV polar feats shape:', bev_polar_feats.shape)

        # Normalise grid to [-1, 1]
        norm_grid = self.normalise_grid(grid, calib)

        if self.additions_conv:
            bev_polar_feats_gn = self.gn_polar(bev_polar_feats)
            # Sample BEV polar features at grid locations
            # [B, C, Z, W] --> [B, C, Z, X]
            bev_cart_feats = F.grid_sample(
                bev_polar_feats_gn, norm_grid, align_corners=True
            )
            # print('     BEV cart feats shape:', bev_cart_feats.shape)
            return bev_cart_feats
        else:
            bev_cart_feats = F.grid_sample(
                bev_polar_feats, norm_grid, align_corners=True
            )
            # print('     BEV cart feats shape:', bev_cart_feats.shape)
            return bev_cart_feats

    def normalise_grid(self, grid, calib):
        """
        :param grid: BEV grid in with coords range
        [grid_h1, grid_h2] and [-grid_w/2, grid_w/2]
        :param calib:
        :return:
        """

        f, cu = calib[:, 0, 0], calib[:, 0, 2]
        batch_size = len(calib)

        # Compute positive x dimension at z_max and z_min
        # Computed x dimension is half grid width
        x_zmax = self.z_max / f * cu
        x_zmin = self.z_min / f * cu

        # Compute normalising constant for each row along the z-axis
        sample_res_z = (self.z_max - self.z_min) / self.cell_size
        sample_res_x = grid.shape[2] / self.cell_size

        norm_z = (
            2
            * (grid[..., 1] - grid[..., 1].min())
            / (grid[..., 1].max() - grid[..., 1].min())
            - 1
        )

        norm_scale_x = torch.stack(
            [
                torch.linspace(float(x_zmin[i]), float(x_zmax[i]), int(sample_res_z))
                for i in range(batch_size)
            ]
        )

        grid_ones = torch.ones_like(grid)
        grid_ones[..., 0] *= norm_scale_x.view(batch_size, -1, 1).cuda()

        # Normalise grid to [-1, 1]
        norm_grid = grid / grid_ones

        norm_grid[..., 1] = norm_z

        return norm_grid


class sample_polar2cart(nn.Module):
    def __init__(
        self,
        z_max,
        z_min,
        cell_size,
    ):
        super().__init__()

        self.z_max = z_max
        self.z_min = z_min
        self.cell_size = cell_size

    def forward(self, features, calib, grid):

        # Normalise grid to [-1, 1]
        norm_grid = self.normalise_grid(grid, calib)

        bev_cart_feats = F.grid_sample(features, norm_grid, align_corners=True)
        return bev_cart_feats

    def normalise_grid(self, grid, calib):
        """
        :param grid: BEV grid in with coords range
        [grid_h1, grid_h2] and [-grid_w/2, grid_w/2]
        :param calib:
        :return:
        """

        f, cu = calib[:, 0, 0], calib[:, 0, 2]
        batch_size = len(calib)

        # Compute positive x dimension at z_max and z_min
        # Computed x dimension is half grid width
        x_zmax = self.z_max / f * cu
        x_zmin = self.z_min / f * cu

        # Compute normalising constant for each row along the z-axis
        sample_res_z = (self.z_max - self.z_min) / self.cell_size
        sample_res_x = grid.shape[2] / self.cell_size

        norm_z = (
            2
            * (grid[..., 1] - grid[..., 1].min())
            / (grid[..., 1].max() - grid[..., 1].min())
            - 1
        )

        norm_scale_x = torch.stack(
            [
                torch.linspace(float(x_zmin[i]), float(x_zmax[i]), int(sample_res_z))
                for i in range(batch_size)
            ]
        )

        grid_ones = torch.ones_like(grid)
        grid_ones[..., 0] *= norm_scale_x.view(batch_size, -1, 1).cuda()

        # Normalise grid to [-1, 1]
        norm_grid = grid / grid_ones

        norm_grid[..., 1] = norm_z

        return norm_grid


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
