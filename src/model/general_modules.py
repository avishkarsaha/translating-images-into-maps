import torch
from torch import nn
from torch.nn import functional as F

from src.model import resnet


class FeatureGating(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.fc = nn.Linear(channel_dim, channel_dim, bias=True)

    def forward(self, x):
        """Feature gating as used in S3D-G."""
        spatiotemporal_average = torch.mean(x, dim=[2, 3, 4])
        W = self.fc(spatiotemporal_average)
        W = torch.sigmoid(W)
        return W[:, :, None, None, None] * x


class STConv3D(nn.Module):
    """
    Saptiotemporal convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ks_size,
        kt_size,
        ps_size,
        pt_size,
        num_groups=17,
    ):
        super().__init__()

        self.spatial = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks_size,
            padding=ps_size,
        )
        self.bn1 = nn.GroupNorm(num_groups, out_channels)
        self.temporal = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kt_size,
            padding=pt_size,
        )
        self.bn2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.spatial(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.temporal(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class S3D_TemporalBlock_Comp_Var(nn.Module):
    """
    Separable 3D convolutions with local and global context

    Parameters:

    Input:
        Tensor of shape [N, C, 3, D, W]
    Output:
        Tensor of shape [N, C, 1, D, W]

        References
        1. https://arxiv.org/pdf/2003.06409.pdf and
        2. https://arxiv.org/pdf/1712.04851.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
        gating=False,
        num_groups=17,
        temporal_range=2,
    ):
        super().__init__()

        comp_channels = in_channels // 3
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)
        t = temporal_range
        # Global context
        s1, s2, s4 = 1, 2, 4
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[t, 5, 5], stride=[s1, s1, s1], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            # nn.Upsample(
            #     size=[1, in_size_z, in_size_x],
            #     mode="trilinear"
            # )
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[t, 5, 5], stride=[s2, s2, s2], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[t, 5, 5], stride=[s4, s4, s4], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        self.dynamics_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[t, 5, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 2],
                num_groups=num_groups,
            ),
        )
        self.dynamics_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[t, 5, 1],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 0],
                num_groups=num_groups,
            ),
        )
        self.dynamics_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[t, 1, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 0, 2],
                num_groups=num_groups,
            ),
        )
        self.compress_out = nn.Conv3d(int(comp_channels * 6), out_channels, 1)

        # Feature gating mechanism
        self.gating = gating
        if self.gating:
            self.feature_gating = FeatureGating(out_channels)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        motion_x = self.dynamics_x(x)
        motion_z = self.dynamics_z(x)
        motion_xz = self.dynamics_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        out = self.compress_out(
            torch.cat(
                [motion_x, motion_z, motion_xz, context_s1, context_s2, context_s4],
                dim=1,
            )
        )

        if self.gating:
            out = self.feature_gating(out)

        # Add input of last frame
        out = out + x[:, :, -1:]

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)
        return out


class S3D_TemporalBlock_Comp_100(nn.Module):
    """
    Separable 3D convolutions with local and global context
    Kernels are twice the size of S3D_TemporalBlock_Comp to account for
    the larger spatial size of features (100x100)

    Input:
        Tensor of shape [N, C, 2, D, W]
    Output:
        Tensor of shape [N, C, 1, D, W]

        References
        1. https://arxiv.org/pdf/2003.06409.pdf and
        2. https://arxiv.org/pdf/1712.04851.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
        gating=False,
        num_groups=17,
    ):
        super().__init__()

        comp_channels = in_channels // 3
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)

        # Global context
        s1, s2, s4 = 1, 2, 4
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 9, 9], stride=[s1, s1, s1], padding=[0, 4, 4]),
            nn.Conv3d(in_channels, comp_channels, 1),
            # nn.Upsample(
            #     size=[1, in_size_z, in_size_x],
            #     mode="trilinear"
            # )
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 9, 9], stride=[s2, s2, s2], padding=[0, 4, 4]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 9, 9], stride=[s4, s4, s4], padding=[0, 4, 4]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        self.dynamics_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 9, 9],
                kt_size=[2, 9, 9],
                ps_size=[0, 4, 4],
                pt_size=[0, 4, 4],
                num_groups=num_groups,
            ),
        )
        self.dynamics_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 9, 9],
                kt_size=[2, 9, 1],
                ps_size=[0, 4, 4],
                pt_size=[0, 4, 0],
                num_groups=num_groups,
            ),
        )
        self.dynamics_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 9, 9],
                kt_size=[2, 1, 9],
                ps_size=[0, 4, 4],
                pt_size=[0, 0, 4],
                num_groups=num_groups,
            ),
        )
        self.compress_out = nn.Conv3d(int(comp_channels * 6), out_channels, 1)

        # Feature gating mechanism
        self.gating = gating
        if self.gating:
            self.feature_gating = FeatureGating(out_channels)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        motion_x = self.dynamics_x(x)
        motion_z = self.dynamics_z(x)
        motion_xz = self.dynamics_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        out = self.compress_out(
            torch.cat(
                [motion_x, motion_z, motion_xz, context_s1, context_s2, context_s4],
                dim=1,
            )
        )

        if self.gating:
            out = self.feature_gating(out)

        # Add input of last frame
        out = out + x[:, :, 1:2]

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)
        return out


class S3D_TemporalBlock_Comp(nn.Module):
    """
    Separable 3D convolutions with local and global context

    Parameters:

    Input:
        Tensor of shape [N, C, 2, D, W]
    Output:
        Tensor of shape [N, C, 1, D, W]

        References
        1. https://arxiv.org/pdf/2003.06409.pdf and
        2. https://arxiv.org/pdf/1712.04851.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
        gating=False,
        num_groups=17,
    ):
        super().__init__()

        comp_channels = in_channels // 3
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)

        # Global context
        s1, s2, s4 = 1, 2, 4
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s1, s1, s1], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            # nn.Upsample(
            #     size=[1, in_size_z, in_size_x],
            #     mode="trilinear"
            # )
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s2, s2, s2], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s4, s4, s4], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        self.dynamics_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 5, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 2],
                num_groups=num_groups,
            ),
        )
        self.dynamics_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 5, 1],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 0],
                num_groups=num_groups,
            ),
        )
        self.dynamics_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 1, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 0, 2],
                num_groups=num_groups,
            ),
        )
        self.compress_out = nn.Conv3d(int(comp_channels * 6), out_channels, 1)

        # Feature gating mechanism
        self.gating = gating
        if self.gating:
            self.feature_gating = FeatureGating(out_channels)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        motion_x = self.dynamics_x(x)
        motion_z = self.dynamics_z(x)
        motion_xz = self.dynamics_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        out = self.compress_out(
            torch.cat(
                [motion_x, motion_z, motion_xz, context_s1, context_s2, context_s4],
                dim=1,
            )
        )

        if self.gating:
            out = self.feature_gating(out)

        # Add input of last frame
        out = out + x[:, :, 1:2]

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)
        return out


class S3D_TemporalBlock_Agg(nn.Module):
    """
    Aggregates > 2 frames into one
    Separable 3D convolutions with local and global context
    Input of last frame is concatenated instead of added
    Parameters:

    Input:
        Tensor of shape [N, C, 2, D, W]
    Output:
        Tensor of shape [N, C, 1, D, W]

        References
        1. https://arxiv.org/pdf/2003.06409.pdf and
        2. https://arxiv.org/pdf/1712.04851.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
        gating=False,
        n_groups=17,
        n_in_frames=4,
    ):
        super().__init__()

        comp_channels = in_channels // 3
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)

        # Global context
        s1, s2, s4 = 1, 2, 4
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=[n_in_frames, 5, 5], stride=[s1, s1, s1], padding=[0, 2, 2]
            ),
            nn.Conv3d(in_channels, comp_channels, 1),
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=[n_in_frames, 5, 5], stride=[s2, s2, s2], padding=[0, 2, 2]
            ),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=[n_in_frames, 5, 5], stride=[s4, s4, s4], padding=[0, 2, 2]
            ),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        self.dynamics_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[4, 5, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 2],
                num_groups=n_groups,
            ),
        )
        self.dynamics_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[4, 5, 1],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 0],
                num_groups=n_groups,
            ),
        )
        self.dynamics_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[4, 1, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 0, 2],
                num_groups=n_groups,
            ),
        )
        self.compress_out = nn.Conv3d(
            int(comp_channels * 6) + in_channels, out_channels, 1
        )

        # Feature gating mechanism
        self.gating = gating
        if self.gating:
            self.feature_gating = FeatureGating(out_channels)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        motion_x = self.dynamics_x(x)
        motion_z = self.dynamics_z(x)
        motion_xz = self.dynamics_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        # Concatenate all streams along with input of last frame
        out = self.compress_out(
            torch.cat(
                [
                    motion_x,
                    motion_z,
                    motion_xz,
                    context_s1,
                    context_s2,
                    context_s4,
                    x[:, :, 1:2],
                ],
                dim=1,
            )
        )

        if self.gating:
            out = self.feature_gating(out)

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)
        return out


class S3D_TemporalBlock_Cat(nn.Module):
    """
    Separable 3D convolutions with local and global context
    Input of last frame is concatenated instead of added
    Parameters:

    Input:
        Tensor of shape [N, C, 2, D, W]
    Output:
        Tensor of shape [N, C, 1, D, W]

        References
        1. https://arxiv.org/pdf/2003.06409.pdf and
        2. https://arxiv.org/pdf/1712.04851.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
        gating=False,
        num_groups=17,
    ):
        super().__init__()

        comp_channels = in_channels // 3
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)

        # Global context
        s1, s2, s4 = 1, 2, 4
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s1, s1, s1], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            # nn.Upsample(
            #     size=[1, in_size_z, in_size_x],
            #     mode="trilinear"
            # )
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s2, s2, s2], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[2, 5, 5], stride=[s4, s4, s4], padding=[0, 2, 2]),
            nn.Conv3d(in_channels, comp_channels, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        self.dynamics_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 5, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 2],
                num_groups=num_groups,
            ),
        )
        self.dynamics_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 5, 1],
                ps_size=[0, 2, 2],
                pt_size=[0, 2, 0],
                num_groups=num_groups,
            ),
        )
        self.dynamics_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channels, 1),
            STConv3D(
                in_channels=comp_channels,
                out_channels=comp_channels,
                ks_size=[1, 5, 5],
                kt_size=[2, 1, 5],
                ps_size=[0, 2, 2],
                pt_size=[0, 0, 2],
                num_groups=num_groups,
            ),
        )
        self.compress_out = nn.Conv3d(
            int(comp_channels * 6) + in_channels, out_channels, 1
        )

        # Feature gating mechanism
        self.gating = gating
        if self.gating:
            self.feature_gating = FeatureGating(out_channels)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        motion_x = self.dynamics_x(x)
        motion_z = self.dynamics_z(x)
        motion_xz = self.dynamics_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        # Concatenate all streams along with input of last frame
        out = self.compress_out(
            torch.cat(
                [
                    motion_x,
                    motion_z,
                    motion_xz,
                    context_s1,
                    context_s2,
                    context_s4,
                    x[:, :, 1:2],
                ],
                dim=1,
            )
        )

        if self.gating:
            out = self.feature_gating(out)

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)
        return out


class TemporalBlock(nn.Module):
    """Spatio-temporal 3D convolutions with local and global context
    From https://arxiv.org/pdf/2003.06409.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size_x,
        in_size_z,
    ):
        super().__init__()

        # Global context
        s1, s2, s4 = 1, 2, 4
        num_global_streams = 3
        comp_channel_global = in_channels // num_global_streams
        self.global_s1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[3, 5, 5], stride=[s1, s1, s1], padding=[1, 2, 2]),
            nn.Conv3d(in_channels, comp_channel_global, 1),
        )
        self.global_s2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[3, 5, 5], stride=[s2, s2, s2], padding=[1, 2, 2]),
            nn.Conv3d(in_channels, comp_channel_global, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )
        self.global_s4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=[3, 5, 5], stride=[s4, s4, s4], padding=[1, 2, 2]),
            nn.Conv3d(in_channels, comp_channel_global, 1),
            nn.Upsample(
                size=[1, in_size_z, in_size_x], mode="trilinear", align_corners=True
            ),
        )

        # Local context
        num_local_streams = 4
        comp_channel_local = in_channels // num_local_streams
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, comp_channel_local, 1),
            nn.Conv3d(
                in_channels=comp_channel_local,
                out_channels=comp_channel_local,
                kernel_size=[1, 5, 5],
                padding=[0, 2, 2],
            ),
        )
        self.motion_x = nn.Sequential(
            nn.Conv3d(in_channels, comp_channel_local, 1),
            nn.Conv3d(
                in_channels=comp_channel_local,
                out_channels=comp_channel_local,
                kernel_size=[3, 1, 5],
                padding=[1, 0, 2],
            ),
        )
        self.motion_z = nn.Sequential(
            nn.Conv3d(in_channels, comp_channel_local, 1),
            nn.Conv3d(
                in_channels=comp_channel_local,
                out_channels=comp_channel_local,
                kernel_size=[3, 5, 1],
                padding=[1, 2, 0],
            ),
        )
        self.motion_xz = nn.Sequential(
            nn.Conv3d(in_channels, comp_channel_local, 1),
            nn.Conv3d(
                in_channels=comp_channel_local,
                out_channels=comp_channel_local,
                kernel_size=[3, 5, 5],
                padding=[1, 2, 2],
            ),
        )

        self.compress_out = nn.Conv3d(
            comp_channel_global + comp_channel_local, out_channels, 1
        )

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        spatial = self.spatial(x)
        dyn_x = self.motion_x(x)
        dyn_z = self.motion_z(x)
        dyn_xz = self.motion_xz(x)

        context_s1 = self.global_s1(x)
        context_s2 = self.global_s2(x)
        context_s4 = self.global_s4(x)

        out = self.compress_out(
            torch.cat([dyn_x, dyn_z, dyn_xz, context_s1, context_s2, context_s4], dim=1)
        )

        # Add input of last frame
        out = out + x[:, :, 1:2]

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)

        return out


class SpaTemConv_00(nn.Module):
    """Spatio-temporal 3D convolutions with local context only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kx_size,
        kz_size,
        kxz_size,
    ):
        super().__init__()

        # Local context
        comp_channels = in_channels // 2
        self.compress_in = nn.Conv3d(in_channels, comp_channels, 1)
        # self.conv_spatial = nn.Conv3d(
        #     in_channels=comp_channels,
        #     out_channels=comp_channels,
        #     kernel_size=ks_size,
        #     padding=[0, 2, 2],
        # )
        self.conv_motion_x = nn.Conv3d(
            in_channels=comp_channels,
            out_channels=comp_channels,
            kernel_size=kx_size,
            padding=[0, 0, 2],
        )
        self.conv_motion_z = nn.Conv3d(
            in_channels=comp_channels,
            out_channels=comp_channels,
            kernel_size=kz_size,
            padding=[0, 2, 0],
        )
        self.conv_motion_xz = nn.Conv3d(
            in_channels=comp_channels,
            out_channels=comp_channels,
            kernel_size=kxz_size,
            padding=[0, 2, 2],
        )
        self.compress_out = nn.Conv3d(int(comp_channels * 3), out_channels, 1)

    def forward(self, x):
        # # Permute input dimensions [N, T, C, H, W] --> [N, C, T, H, W]
        # x = x.permute(0, 2, 1, 3, 4)

        out = self.compress_in(x)
        # spatial = self.conv_spatial(out)
        motion_x = self.conv_motion_x(out)
        motion_z = self.conv_motion_z(out)
        motion_xz = self.conv_motion_xz(out)
        out = self.compress_out(torch.cat([motion_x, motion_z, motion_xz], dim=1))

        # Permute back to input shape [N, C, T, H, W] --> [N, T, C, H, W]
        # out = out.permute(0, 2, 1, 3, 4)

        return out


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class WidthDownsamplers(nn.Module):
    """
    Width Downsampler for ResNet features of different sizes
    """

    def __init__(self, widths, heights, final_w):
        super().__init__()

        # Downsample width to under size then upsample to size
        self.downsamplers = []

        for width, height in zip(widths, heights):
            scale_w = (width // final_w) + 1
            if scale_w > 1:
                # If feature map width is larger than final width
                # then conv down below final size and then interpolate to
                # final size
                downsample = nn.Sequential(
                    # Downsample with convolutions to below final size
                    # to preserve information
                    nn.Conv2d(
                        256,
                        256,
                        kernel_size=int(2 * scale_w + 1),
                        stride=int(scale_w),
                        padding=int((scale_w + 1) // 2),
                    ),
                    # Upsample to final width
                    nn.Upsample(size=[int(height), int(final_w)], mode="nearest"),
                    nn.GroupNorm(16, 256),
                    nn.ReLU(),
                )
            else:
                # If feature map width is smaller than final width
                # then interpolate up to final size
                downsample = nn.Sequential(
                    # Upsample to final width
                    nn.Upsample(size=[int(height), int(final_w)], mode="nearest"),
                    nn.GroupNorm(16, 256),
                    nn.ReLU(),
                )

            self.downsamplers.append(downsample)

        self.downsamplers = nn.ModuleList(self.downsamplers)

    def forward(self, f8, f16, f32, f64):

        f8 = self.downsamplers[0](f8)
        f16 = self.downsamplers[1](f16)
        f32 = self.downsamplers[2](f32)
        f64 = self.downsamplers[3](f64)

        return f8, f16, f32, f64


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q,
        k,
        v,
        d_k,
    ):

        attn = torch.matmul(q, k) / torch.sqrt(d_k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias
        )
        qh = q.view(
            q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads
        )
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


class AggregateNode_Cat(nn.Module):
    """
    Aggregates two inputs by matching their batch dim, concatenating their channel dims
    and then passing through a restnet block
    """

    def __init__(self, dim1, dim2, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            dim1 + dim2, dim2, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.gn1 = nn.GroupNorm(dim2 // 8, dim2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            dim2, dim2, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.gn2 = nn.GroupNorm(dim2 // 8, dim2)

    def forward(self, x1, x2):
        identity = x2

        out = self.conv1(torch.cat([x2, x1], dim=1))
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out


class AggregateNode_Add(nn.Module):
    """
    Aggregates two inputs by matching their batch dim and projecting
    to match their channel dims
    and then passing through a restnet block
    """

    def __init__(self, dim1, dim2, kernel_size=2):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=1),
            nn.GroupNorm(dim2 // 8, dim2),
        )

        self.conv1 = nn.Conv2d(dim1 + dim2, dim2, kernel_size=3, stride=1)
        self.gn1 = nn.GroupNorm(dim2 // 8, dim2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim2, dim2, kernel_size=3, stride=1)
        self.gn2 = nn.GroupNorm(dim2 // 8, dim2)

    def forward(self, x1, x2):

        x = self.project(x1)  # project to same dim as x2
        x = self.relu(x)

        x = x + x2  # add projected x1 and x2

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += x
        out = self.relu(out)

        return out


class ResNetBlock_GN(nn.Module):
    """
    ResNet block with group norm
    """

    def __init__(self, dim2, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            dim2, dim2, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.gn1 = nn.GroupNorm(dim2 // 8, dim2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            dim2, dim2, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.gn2 = nn.GroupNorm(dim2 // 8, dim2)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DLA_Node(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, padding=1, norm="BatchNorm"
    ):
        super().__init__()

        self.conv = _resnet_bblock_enc_layer(
            in_planes, out_planes, stride=1, num_blocks=2
        )

        if norm == "GroupNorm":
            self.bn = nn.GroupNorm(out_planes // 8, out_planes)
        elif norm == "BatchNorm":
            self.bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x


class DLA_Node_old(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, padding=1, norm="BatchNorm"
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, padding=padding
        )

        if norm == "GroupNorm":
            self.bn = nn.GroupNorm(out_planes // 8, out_planes)
        elif norm == "BatchNorm":
            self.bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class IDA_up(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.ConvTranspose2d(
            out_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
        )

    def forward(self, x):
        x = self.project(x)
        x = self.upsample(x)
        return x


class Down(nn.Module):
    """Downscaling using residual BasicBlocks with stride of 2"""

    def __init__(self, in_planes, out_planes, num_blocks):
        super().__init__()
        stride = 2

        # Condense identity channels
        condense = nn.Sequential(
            resnet.conv1x1(in_planes, out_planes, stride=1),
            nn.BatchNorm2d(out_planes),
        )

        # Process and condense channels
        self.conv = nn.Sequential(
            *[
                resnet.BasicBlock(in_planes, out_planes, stride=1, downsample=condense)
                for _ in range(1)
            ]
        )

        # Process and downsample using stride
        downsample = nn.Sequential(
            resnet.conv1x1(out_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
        )
        self.down = nn.Sequential(
            *[
                resnet.BasicBlock(
                    out_planes, out_planes, stride=stride, downsample=downsample
                )
                for _ in range(1)
            ]
        )
        self.conv2 = _resnet_bblock_down_layer(
            out_planes, out_planes, stride=1, num_blocks=num_blocks
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.down(out)
        return out


class Down_old(nn.Module):
    """Downscaling using residual BasicBlocks with stride of 2"""

    def __init__(self, in_planes, out_planes, num_blocks, stride=2):
        super().__init__()

        # Condense identity channels
        condense = nn.Sequential(
            resnet.conv1x1(in_planes, out_planes, stride=1),
            nn.BatchNorm2d(out_planes),
        )

        # Process and condense channels
        self.conv = nn.Sequential(
            *[
                resnet.BasicBlock(in_planes, out_planes, stride=1, downsample=condense)
                for _ in range(num_blocks - 1)
            ]
        )

        # Process and downsample using stride
        downsample = nn.Sequential(
            resnet.conv1x1(out_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
        )
        self.down = nn.Sequential(
            *[
                resnet.BasicBlock(
                    out_planes, out_planes, stride=stride, downsample=downsample
                )
                for _ in range(1)
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.down(out)
        return out


class Up(nn.Module):
    """Upsampling using residual BasicBlocks with stride of 2 for use in upsampling nodes
    with skip-connections"""

    def __init__(
        self,
        in_planes,
        out_planes,
        out_size=[200, 200],
        num_blocks=2,
        bilinear=True,
        kernel_size=3,
    ):
        super().__init__()
        expansion = 1
        stride = 2

        if bilinear:
            self.up = nn.ConvTranspose2d(
                in_planes // 2,
                in_planes // 2,
                kernel_size=kernel_size,
                padding=1,
                stride=2,
            )
        else:
            self.up = nn.Upsample(size=out_size, mode="nearest")

        self.conv = _resnet_bblock_enc_layer_old(
            in_planes, out_planes, stride=1, num_blocks=num_blocks
        )

    def forward(self, x1, x2):
        out = self.up(x1)
        out = self.conv(torch.cat([out, x2], dim=1))
        return out


class Upsample(nn.Module):
    """Upsampling using residual BasicBlocks with stride of 2"""

    def __init__(
        self,
        in_planes,
        out_planes,
        out_size=[50, 50],
        num_blocks=2,
        bilinear=False,
        kernel_size=3,
    ):
        super().__init__()
        expansion = 1
        stride = 2

        if bilinear:
            self.up = nn.ConvTranspose2d(
                in_planes,
                in_planes,
                kernel_size=kernel_size,
                padding=1,
                stride=2,
            )
        else:
            self.up = nn.Upsample(size=out_size, mode="nearest")

        self.conv = _resnet_bblock_enc_layer(
            in_planes, out_planes, stride=1, num_blocks=num_blocks
        )

    def forward(self, x1):
        out = self.up(x1)
        out = self.conv(out)
        return out


class Upsample_old(nn.Module):
    """Upsampling using residual BasicBlocks with stride of 2"""

    def __init__(
        self,
        in_planes,
        out_planes,
        out_size=[50, 50],
        num_blocks=2,
        bilinear=False,
        kernel_size=3,
    ):
        super().__init__()
        expansion = 1
        stride = 2

        if bilinear:
            self.up = nn.ConvTranspose2d(
                in_planes,
                in_planes,
                kernel_size=kernel_size,
                padding=1,
                stride=2,
            )
        else:
            self.up = nn.Upsample(size=out_size, mode="nearest")

        self.conv = _resnet_bblock_enc_layer_old(
            in_planes, out_planes, stride=1, num_blocks=num_blocks
        )

    def forward(self, x1):
        out = self.up(x1)
        out = self.conv(out)
        return out


def _resnet_bblock_enc_layer(inplanes, planes, stride=2, num_blocks=1):
    """Create a ResNet layer for the topdown network specifically"""
    expansion = 1

    downsample = nn.Sequential(
        resnet.conv1x1(inplanes, planes * expansion, stride),
        nn.BatchNorm2d(planes * expansion),
    )
    return nn.Sequential(
        *[resnet.BasicBlock(inplanes, inplanes) for _ in range(num_blocks)],
        resnet.BasicBlock(inplanes, planes, stride, downsample),
    )


def _resnet_bblock_enc_layer_old(inplanes, planes, stride=2, num_blocks=1):
    """Create a ResNet layer for the topdown network specifically"""
    expansion = 1

    downsample = nn.Sequential(
        resnet.conv1x1(inplanes, planes * expansion, stride),
        nn.BatchNorm2d(planes * expansion),
    )
    return nn.Sequential(
        resnet.BasicBlock(inplanes, inplanes),
        resnet.BasicBlock(inplanes, planes, stride, downsample),
    )


def _resnet_bblock_down_layer(inplanes, planes, stride=2, num_blocks=1):
    """Create a ResNet layer for the topdown network specifically"""
    expansion = 1

    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            resnet.conv1x1(inplanes, planes * expansion, stride),
            nn.BatchNorm2d(planes * expansion),
        )
        return nn.Sequential(
            *[
                resnet.BasicBlock(inplanes, planes, stride, downsample)
                for _ in range(num_blocks)
            ]
        )
    else:
        return nn.Sequential(
            *[resnet.BasicBlock(inplanes, planes) for _ in range(num_blocks)]
        )


def _resnet_bblock_up_layer(inplanes, planes, out_size, num_blocks):
    """Create a ResNet layer for the topdown network specifically"""

    return nn.Sequential(
        nn.Upsample(size=out_size, mode="nearest"),
        *[resnet.BasicBlock(inplanes, planes) for _ in range(num_blocks)],
    )
