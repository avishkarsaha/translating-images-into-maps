import math
import numpy as np
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone_utils import resnet_fpn_backbone, resnet_fpn_backbone_roddick
from src.model import resnet
from src.model.bev_transform import BEVT, sample_polar2cart
from src.model.general_modules import S3D_TemporalBlock_Comp, Flatten, WidthDownsamplers, MLP, MultiHeadAttentionMap, \
    AggregateNode_Cat, ResNetBlock_GN, Identity, DLA_Node, DLA_Node_old, IDA_up, Down, Down_old, Up, Upsample_old, \
    _resnet_bblock_enc_layer, _resnet_bblock_enc_layer_old
from src.model.positional_encoding import (
    PositionalEncoding,
    PositionalEncoding2DwDropout,
    PositionalEncoding_00,
    PositionalEncoding3D,
    PositionEmbeddingSine,
)
from src.model.transformer import (
    Transformer,
    Transformer_00,
    TransformerMonotonic,
)
from src.model.axial_attention import AxialAttention


class PyrOccTranDetrMonoInv_S_0904_old(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Flip features along height dimension (for monotonic attention later)
        feat8 = feat8.flip(dims=[2])
        feat16 = feat16.flip(dims=[2])
        feat32 = feat32.flip(dims=[2])
        feat64 = feat64.flip(dims=[2])

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetrMono_S_0904_old(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = TransformerMonotonic(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_AxialAttention(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    Axial attention applied to image plane feature maps
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # Axial attention for image-plane features
        self.axial_attn = nn.ModuleList(
            [
                nn.Sequential(
                    AxialAttention(
                        dim=256,  # embedding dimension
                        dim_index=1,  # where is the embedding dimension
                        heads=4,  # number of heads for multi-head attention
                        num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                    ),
                    nn.GroupNorm(256 // 8, 256),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply axial attention to image-plane features
        feat8, feat16, feat32, feat64 = [
            checkpoint(attn, f)
            for attn, f in zip(self.axial_attn, [feat8, feat16, feat32, feat64])
        ]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_rep100x100_out100x100(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_rep100x100_out100x100_swa(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, input):
        image, calib, grid = input

        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_prenorm_old_rep100x100_out100x100(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_M_0904_old_rep100x100_out100x100(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    Axial attention output is for last in sequence
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        warm_up_frames=3,
        n_channels_dynamics=256,
        n_groups_dynamics=17,
        lookahead_frames=0,
        dla_l1_n_channels=32,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Dynamics with Axial Attention
        self.axial_attn = AxialAttention(
            dim=256,  # embedding dimension
            dim_index=2,  # where is the embedding dimension
            heads=4,  # number of heads for multi-head attention
            num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
        )
        self.dynamics_norm = nn.Sequential(
            nn.GroupNorm(256 // 8, 256), nn.ReLU(inplace=True),
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        # Reshape input from [B, T, C, H, W] to [B x T, C, H, W]
        B, T, C, H, W = image.shape
        N = B * T
        image = image.view(N, C, H, W)
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply axial attention to learn dynamics
        # Get context frames on which to condition each estimate frame
        # [B x T, C, H, W] --> [B, T, C, H, W]
        context_frames = torch.stack(torch.split(bev, [T for _ in range(B)], dim=0))

        # Apply axial attention and then take first frame of every batch
        dynamics = checkpoint(self.axial_attn, context_frames)[:, 0]
        dynamics = checkpoint(self.dynamics_norm, dynamics)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, dynamics)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_M_10_0904_old_rep50x50_out100x100(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    Spatial at 100x100
    Dynamics learnt at smaller scale of 50x50, channel number kept same
    Axial attention output is for last frame in sequence instead of first
    like when dynamics are learnt at 100x100
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        warm_up_frames=3,
        n_channels_dynamics=256,
        n_groups_dynamics=17,
        lookahead_frames=False,
        dla_l1_n_channels=32,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc_img = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.pos_enc_polar = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Dynamics with Axial Attention
        # Downsample spatially to 50x50
        self.pre_dynamics = Down_old(256, n_channels_dynamics, num_blocks=2)
        self.axial_attn = AxialAttention(
            dim=256,  # embedding dimension
            dim_index=2,  # where is the embedding dimension
            heads=4,  # number of heads for multi-head attention
            num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
        )
        self.dynamics_norm = nn.Sequential(
            nn.GroupNorm(256 // 8, 256), nn.ReLU(inplace=True),
        )
        # Upsample dynamics back to 100x100
        self.post_dynamics = Upsample_old(
            256,
            n_channels_dynamics,
            out_size=[100, 100],
            num_blocks=2,
            bilinear=True,
            kernel_size=4,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        # Reshape input from [B, T, C, H, W] to [B x T, C, H, W]
        B, T, C, H, W = image.shape
        N = B * T
        image = image.view(N, C, H, W)
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        # Add positional encodings to transformer enc/dec inputs
        pe_feat8 = self.pos_enc_img(feat8, feat8.shape[-2], feat8.shape[-1])
        pe_feat16 = self.pos_enc_img(feat16, feat16.shape[-2], feat16.shape[-1])
        pe_feat32 = self.pos_enc_img(feat32, feat32.shape[-2], feat32.shape[-1])
        pe_feat64 = self.pos_enc_img(feat64, feat64.shape[-2], feat64.shape[-1])

        pe_tgt8 = self.pos_enc_polar(tgt8.unsqueeze(1), tgt8.shape[-2], tgt8.shape[-1])
        pe_tgt16 = self.pos_enc_polar(
            tgt16.unsqueeze(1), tgt16.shape[-2], tgt16.shape[-1]
        )
        pe_tgt32 = self.pos_enc_polar(
            tgt32.unsqueeze(1), tgt32.shape[-2], tgt32.shape[-1]
        )
        pe_tgt64 = self.pos_enc_polar(
            tgt64.unsqueeze(1), tgt64.shape[-2], tgt64.shape[-1]
        )

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.trans_reshape(pe_tgt8),
            self.trans_reshape(qe8),
            self.trans_reshape(pe_feat8),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.trans_reshape(pe_tgt16),
            self.trans_reshape(qe16),
            self.trans_reshape(pe_feat16),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.trans_reshape(pe_tgt32),
            self.trans_reshape(qe32),
            self.trans_reshape(pe_feat32),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.trans_reshape(pe_tgt64),
            self.trans_reshape(qe64),
            self.trans_reshape(pe_feat64),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply axial attention to learn dynamics
        # Downsample BEV copy first
        bev_down = checkpoint(self.pre_dynamics, bev)
        # Get context frames on which to condition each estimate frame
        # [B x T, C, H, W] --> [B, T, C, H, W]
        context_frames = torch.stack(
            torch.split(bev_down, [T for _ in range(B)], dim=0)
        )

        # Apply axial attention and then take first frame of every batch
        dynamics = self.axial_attn(context_frames)[:, -1]
        dynamics = checkpoint(self.dynamics_norm, dynamics)

        # Upsample dynamics and add to bev
        dynamics_up = checkpoint(self.post_dynamics, dynamics)
        dynamics_bev = bev.view(B, T, -1, 100, 100)[:, -1] + dynamics_up

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, dynamics_bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTran_S_0904_old_rep100x100_out100x100(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = nn.Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
        )
        self.tbev16 = nn.Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
        )
        self.tbev32 = nn.Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
        )
        self.tbev64 = nn.Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]

        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        tgt8 = (tgt8.unsqueeze(-1) + self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1) + self.query_embed(tgt16.long())).permute(
            0, 3, 1, 2
        )
        tgt32 = (tgt32.unsqueeze(-1) + self.query_embed(tgt32.long())).permute(
            0, 3, 1, 2
        )
        tgt64 = (tgt64.unsqueeze(-1) + self.query_embed(tgt64.long())).permute(
            0, 3, 1, 2
        )

        bev8 = checkpoint(
            self.tbev8,
            self.pos_enc(self.trans_reshape(feat8)),
            self.pos_enc(self.trans_reshape(tgt8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.pos_enc(self.trans_reshape(feat16)),
            self.pos_enc(self.trans_reshape(tgt16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.pos_enc(self.trans_reshape(feat32)),
            self.pos_enc(self.trans_reshape(tgt32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.pos_enc(self.trans_reshape(feat64)),
            self.pos_enc(self.trans_reshape(tgt64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_2Aux(nn.Module):
    """
    PyrOccTranDetr_S_0904_old_rep100x100_out100x100 with auxiliary semantic segmentation
    loss on the transformer encoder and auxiliary loss on inter-plane attention maps
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_enc=True,
            return_intermediate_dec=True,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_enc=True,
            return_intermediate_dec=True,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def aux_reshape(self, input, N):
        NxW, H1, H2 = input.shape

        x = input.unsqueeze(0).view(N, NxW // N, H1, H2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8, enc8, dec8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16, enc16, dec16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        # reshape auxiliary outputs
        enc8 = self.aux_reshape(enc8, N)
        enc16 = self.aux_reshape(enc16, N)
        dec8 = self.aux_reshape(dec8, N)
        dec16 = self.aux_reshape(dec16, N)

        return (
            [
                output_s1.squeeze(2),
                output_s2.squeeze(2),
                output_s4.squeeze(2),
                output_s8.squeeze(2),
            ],
            [enc8, enc16, dec8, dec16],
        )


class PyrOccTranDetr_S_0904_old_2DPosEnc(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc_img = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.pos_enc_polar = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        # Add positional encodings to transformer enc/dec inputs
        pe_feat8 = self.pos_enc_img(feat8, feat8.shape[-2], feat8.shape[-1])
        pe_feat16 = self.pos_enc_img(feat16, feat16.shape[-2], feat16.shape[-1])
        pe_feat32 = self.pos_enc_img(feat32, feat32.shape[-2], feat32.shape[-1])
        pe_feat64 = self.pos_enc_img(feat64, feat64.shape[-2], feat64.shape[-1])

        pe_tgt8 = self.pos_enc_polar(tgt8.unsqueeze(1), tgt8.shape[-2], tgt8.shape[-1])
        pe_tgt16 = self.pos_enc_polar(
            tgt16.unsqueeze(1), tgt16.shape[-2], tgt16.shape[-1]
        )
        pe_tgt32 = self.pos_enc_polar(
            tgt32.unsqueeze(1), tgt32.shape[-2], tgt32.shape[-1]
        )
        pe_tgt64 = self.pos_enc_polar(
            tgt64.unsqueeze(1), tgt64.shape[-2], tgt64.shape[-1]
        )

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.trans_reshape(pe_tgt8),
            self.trans_reshape(qe8),
            self.trans_reshape(pe_feat8),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.trans_reshape(pe_tgt16),
            self.trans_reshape(qe16),
            self.trans_reshape(pe_feat16),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.trans_reshape(pe_tgt32),
            self.trans_reshape(qe32),
            self.trans_reshape(pe_feat32),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.trans_reshape(pe_tgt64),
            self.trans_reshape(qe64),
            self.trans_reshape(pe_feat64),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_2DPosEnc_Big(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc_img = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.pos_enc_polar = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        # Add positional encodings to transformer enc/dec inputs
        pe_feat8 = self.pos_enc_img(feat8, feat8.shape[-2], feat8.shape[-1])
        pe_feat16 = self.pos_enc_img(feat16, feat16.shape[-2], feat16.shape[-1])
        pe_feat32 = self.pos_enc_img(feat32, feat32.shape[-2], feat32.shape[-1])
        pe_feat64 = self.pos_enc_img(feat64, feat64.shape[-2], feat64.shape[-1])

        pe_tgt8 = self.pos_enc_polar(tgt8.unsqueeze(1), tgt8.shape[-2], tgt8.shape[-1])
        pe_tgt16 = self.pos_enc_polar(
            tgt16.unsqueeze(1), tgt16.shape[-2], tgt16.shape[-1]
        )
        pe_tgt32 = self.pos_enc_polar(
            tgt32.unsqueeze(1), tgt32.shape[-2], tgt32.shape[-1]
        )
        pe_tgt64 = self.pos_enc_polar(
            tgt64.unsqueeze(1), tgt64.shape[-2], tgt64.shape[-1]
        )

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.trans_reshape(pe_tgt8),
            self.trans_reshape(qe8),
            self.trans_reshape(pe_feat8),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.trans_reshape(pe_tgt16),
            self.trans_reshape(qe16),
            self.trans_reshape(pe_feat16),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.trans_reshape(pe_tgt32),
            self.trans_reshape(qe32),
            self.trans_reshape(pe_feat32),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.trans_reshape(pe_tgt64),
            self.trans_reshape(qe64),
            self.trans_reshape(pe_feat64),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_2DPosEncImg(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.pos_enc_img = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        # Add positional encodings to transformer enc/dec inputs
        pe_feat8 = self.pos_enc_img(feat8, feat8.shape[-2], feat8.shape[-1])
        pe_feat16 = self.pos_enc_img(feat16, feat16.shape[-2], feat16.shape[-1])
        pe_feat32 = self.pos_enc_img(feat32, feat32.shape[-2], feat32.shape[-1])
        pe_feat64 = self.pos_enc_img(feat64, feat64.shape[-2], feat64.shape[-1])

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.trans_reshape(pe_feat8),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.pos_enc(self.trans_reshape(tgt16)),
            self.trans_reshape(qe16),
            self.trans_reshape(pe_feat16),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.pos_enc(self.trans_reshape(tgt32)),
            self.trans_reshape(qe32),
            self.trans_reshape(pe_feat32),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.pos_enc(self.trans_reshape(tgt64)),
            self.trans_reshape(qe64),
            self.trans_reshape(pe_feat64),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )


class PyrOccTranDetr_S_0904_old_2DPosEncBEV(nn.Module):
    """
    BEV prediction with single-image inputs using the 0900 architecture.
    """

    def __init__(
        self,
        num_classes=11,
        frontend="resnet50",
        grid_res=1.0,
        pretrained=True,
        img_dims=[1600, 900],
        z_range=[1.0, 6.0, 13.0, 26.0, 51.0],
        h_cropped=[60.0, 60.0, 60.0, 60.0],
        dla_norm="GroupNorm",
        additions_BEVT_linear=False,
        additions_BEVT_conv=False,
        dla_l1_n_channels=32,
        n_enc_layers=2,
        n_dec_layers=2,
    ):

        super().__init__()

        self.image_height = img_dims[1]
        self.image_width = img_dims[0]
        self.z_range = z_range

        # Cropped feature map heights
        h_cropped = torch.tensor(h_cropped)

        # Image heights
        feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        crop = feat_h > h_cropped
        h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
            ~crop
        ).float()
        cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        self.cropped_h = cropped_h

        # Construct frontend network
        self.frontend = resnet_fpn_backbone(
            backbone_name=frontend, pretrained=pretrained
        )

        # BEV transformation using Transformer
        self.pos_enc = PositionalEncoding(256, 0.1, 1000)
        self.pos_enc_polar = PositionalEncoding2DwDropout(256, 0.1, 1000)
        self.query_embed = nn.Embedding(100, 256)
        self.tbev8 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev16 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev32 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.tbev64 = Transformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # BEV Polar to Cartesian Sampler
        self.sample8 = sample_polar2cart(z_range[4], z_range[3], grid_res)
        self.sample16 = sample_polar2cart(z_range[3], z_range[2], grid_res)
        self.sample32 = sample_polar2cart(z_range[2], z_range[1], grid_res)
        self.sample64 = sample_polar2cart(z_range[1], z_range[0], grid_res)

        # Batch normalisation to BEV outputs
        self.bev_bn = nn.Sequential(nn.GroupNorm(16, 256), nn.ReLU(),)

        # Topdown DLA
        n_channels = np.array(2 ** np.arange(4) * dla_l1_n_channels, dtype=int)
        self.topdown_down_s1 = _resnet_bblock_enc_layer_old(
            256, n_channels[0], stride=1, num_blocks=2
        )
        self.topdown_down_s2 = Down_old(n_channels[0], n_channels[1], num_blocks=2)
        self.topdown_down_s4 = Down_old(n_channels[1], n_channels[2], num_blocks=2)
        self.topdown_down_s8 = Down_old(n_channels[2], n_channels[3], num_blocks=2)

        # Aggregate Nodes
        self.node_1_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_2_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_2_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)
        self.node_3_s4 = DLA_Node_old(n_channels[3], n_channels[2], norm=dla_norm)
        self.node_3_s2 = DLA_Node_old(n_channels[2], n_channels[1], norm=dla_norm)
        self.node_3_s1 = DLA_Node_old(n_channels[1], n_channels[0], norm=dla_norm)

        # Upsample and project
        self.up_node_1_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_2_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s1 = IDA_up(n_channels[1], n_channels[0], kernel_size=4)
        self.up_node_3_s2 = IDA_up(n_channels[2], n_channels[1], kernel_size=4)
        self.up_node_3_s4 = IDA_up(n_channels[3], n_channels[2], kernel_size=3)

        # Identity mappings
        self.id_node_1_s1 = Identity()
        self.id_node_2_s1 = Identity()
        self.id_node_2_s2 = Identity()
        self.id_node_3_s1 = Identity()
        self.id_node_3_s2 = Identity()
        self.id_node_3_s4 = Identity()

        # Detection head
        self.head_s8 = nn.Conv2d(n_channels[3], num_classes, kernel_size=3, padding=1)
        self.head_s4 = nn.Conv2d(n_channels[2], num_classes, kernel_size=3, padding=1)
        self.head_s2 = nn.Conv2d(n_channels[1], num_classes, kernel_size=3, padding=1)
        self.head_s1 = nn.Conv2d(n_channels[0], num_classes, kernel_size=3, padding=1)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

        # Grid z range
        # Calculate Z_indices
        z_diff = np.diff(np.array(z_range))
        z_idx = np.cumsum(z_diff)
        self.register_buffer("z_idx", (torch.tensor(z_idx) / grid_res).int())
        self.register_buffer("h_start", h_crop_idx_start.int())
        self.register_buffer("h_end", h_crop_idx_end.int())

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image, calib, grid):
        N = image.shape[0]
        # Normalize by mean and std-dev
        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Frontend outputs
        feats = self.frontend(image)

        # Crop feature maps to certain height
        feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # Apply Transformer
        tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
            -1, self.z_idx[-1] - self.z_idx[-2], -1
        )
        tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
            -1, self.z_idx[-2] - self.z_idx[-3], -1
        )
        tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
            -1, self.z_idx[-3] - self.z_idx[-4], -1
        )
        tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        # Add positional encodings to transformer enc/dec inputs
        pe_tgt8 = self.pos_enc_polar(tgt8.unsqueeze(1), tgt8.shape[-2], tgt8.shape[-1])
        pe_tgt16 = self.pos_enc_polar(
            tgt16.unsqueeze(1), tgt16.shape[-2], tgt16.shape[-1]
        )
        pe_tgt32 = self.pos_enc_polar(
            tgt32.unsqueeze(1), tgt32.shape[-2], tgt32.shape[-1]
        )
        pe_tgt64 = self.pos_enc_polar(
            tgt64.unsqueeze(1), tgt64.shape[-2], tgt64.shape[-1]
        )

        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat8),
            self.trans_reshape(pe_tgt8),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat8)),
        )
        bev16 = checkpoint(
            self.tbev16,
            self.trans_reshape(feat16),
            self.trans_reshape(pe_tgt16),
            self.trans_reshape(qe16),
            self.pos_enc(self.trans_reshape(feat16)),
        )
        bev32 = checkpoint(
            self.tbev32,
            self.trans_reshape(feat32),
            self.trans_reshape(pe_tgt32),
            self.trans_reshape(qe32),
            self.pos_enc(self.trans_reshape(feat32)),
        )
        bev64 = checkpoint(
            self.tbev64,
            self.trans_reshape(feat64),
            self.trans_reshape(pe_tgt64),
            self.trans_reshape(qe64),
            self.pos_enc(self.trans_reshape(feat64)),
        )

        # Resample polar BEV to Cartesian
        bev8 = self.sample8(self.bev_reshape(bev8, N), calib, grid[:, self.z_idx[2] :])
        bev16 = self.sample16(
            self.bev_reshape(bev16, N), calib, grid[:, self.z_idx[1] : self.z_idx[2]]
        )
        bev32 = self.sample32(
            self.bev_reshape(bev32, N), calib, grid[:, self.z_idx[0] : self.z_idx[1]]
        )
        bev64 = self.sample64(
            self.bev_reshape(bev64, N), calib, grid[:, : self.z_idx[0]]
        )

        bev = torch.cat([bev64, bev32, bev16, bev8], dim=2)

        # Apply DLA on topdown
        down_s1 = checkpoint(self.topdown_down_s1, bev)
        down_s2 = checkpoint(self.topdown_down_s2, down_s1)
        down_s4 = checkpoint(self.topdown_down_s4, down_s2)
        down_s8 = checkpoint(self.topdown_down_s8, down_s4)

        node_1_s1 = checkpoint(
            self.node_1_s1,
            torch.cat([self.id_node_1_s1(down_s1), self.up_node_1_s1(down_s2)], dim=1),
        )
        node_2_s2 = checkpoint(
            self.node_2_s2,
            torch.cat([self.id_node_2_s2(down_s2), self.up_node_2_s2(down_s4)], dim=1),
        )
        node_2_s1 = checkpoint(
            self.node_2_s1,
            torch.cat(
                [self.id_node_2_s1(node_1_s1), self.up_node_2_s1(node_2_s2)], dim=1
            ),
        )
        node_3_s4 = checkpoint(
            self.node_3_s4,
            torch.cat([self.id_node_3_s4(down_s4), self.up_node_3_s4(down_s8)], dim=1),
        )
        node_3_s2 = checkpoint(
            self.node_3_s2,
            torch.cat(
                [self.id_node_3_s2(node_2_s2), self.up_node_3_s2(node_3_s4)], dim=1
            ),
        )
        node_3_s1 = checkpoint(
            self.node_3_s1,
            torch.cat(
                [self.id_node_3_s1(node_2_s1), self.up_node_3_s1(node_3_s2)], dim=1
            ),
        )

        # Predict encoded outputs
        batch, _, depth_s8, width_s8 = down_s8.size()
        _, _, depth_s4, width_s4 = node_3_s4.size()
        _, _, depth_s2, width_s2 = node_3_s2.size()
        _, _, depth_s1, width_s1 = node_3_s1.size()
        output_s8 = self.head_s8(down_s8).view(batch, -1, 1, depth_s8, width_s8)
        output_s4 = self.head_s4(node_3_s4).view(batch, -1, 1, depth_s4, width_s4)
        output_s2 = self.head_s2(node_3_s2).view(batch, -1, 1, depth_s2, width_s2)
        output_s1 = self.head_s1(node_3_s1).view(batch, -1, 1, depth_s1, width_s1)

        return (
            output_s1.squeeze(2),
            output_s2.squeeze(2),
            output_s4.squeeze(2),
            output_s8.squeeze(2),
        )
