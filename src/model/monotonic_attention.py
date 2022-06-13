import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicAttention(nn.Module):
    """
    Monotonic Multihead Attention with Infinite Lookback
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        energy_bias=True,
    ):
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.k_proj_mono = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj_mono = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_soft = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj_soft = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.eps = 1e-6

        self.noise_mean = 0.0
        self.noise_var = 1.0

        self.energy_bias_init = -2.0
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if energy_bias is True
            else 0
        )

        self.reset_parameters()

        self.k_in_proj = {"monotonic": self.k_proj_mono, "soft": self.k_proj_soft}
        self.q_in_proj = {"monotonic": self.q_proj_mono, "soft": self.q_proj_soft}
        self.v_in_proj = {"output": self.v_proj}

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj_mono.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj_soft.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_mono.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_soft.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))

        else:
            nn.init.xavier_uniform_(self.k_proj_mono.weight)
            nn.init.xavier_uniform_(self.k_proj_soft.weight)
            nn.init.xavier_uniform_(self.q_proj_mono.weight)
            nn.init.xavier_uniform_(self.q_proj_soft.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def input_projections(self, query, key, value, name):
        """
        Prepare inputs for multihead attention
        ============================================================
        Expected input size
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        name: monotonic or soft
        """

        if query is not None:
            bsz = query.size(1)
            q = self.q_in_proj(query)
            q *= self.scaling
            q = (
                q.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            q = None

        if key is not None:
            bsz = key.size(1)
            k = self.k_in_proj(key)
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            k = None

        if value is not None:
            bsz = value.size(1)
            v = self.v_proj(value)
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            v = None

        return q, k, v

    def p_choose(self, query, key, key_padding_mask=None):
        """
        Calculating step wise prob for reading and writing
        1 to read, 0 to write
        ============================================================
        Expected input size
        query: bsz, tgt_len, embed_dim
        key: bsz, src_len, embed_dim
        value: bsz, src_len, embed_dim
        key_padding_mask: bsz, src_len
        attn_mask: bsz, src_len
        query: bsz, tgt_len, embed_dim
        """

        # prepare inputs
        q_proj, k_proj, _ = self.input_projections(query, key, None, "monotonic")

        # attention energy
        attn_energy = self.attn_energy(q_proj, k_proj, key_padding_mask)

        noise = 0

        if self.training:
            # add noise here to encourage discretness
            noise = (
                torch.normal(self.noise_mean, self.noise_var, attn_energy.size())
                .type_as(attn_energy)
                .to(attn_energy.device)
            )

        p_choose = torch.sigmoid(attn_energy + noise)
        _, _, tgt_len, src_len = p_choose.size()

        # p_choose: bsz * self.num_heads, tgt_len, src_len
        return p_choose.view(-1, tgt_len, src_len)

    def attn_energy(self, q_proj, k_proj, key_padding_mask=None):
        """
        Calculating monotonic energies
        ============================================================
        Expected input size
        q_proj: bsz * num_heads, tgt_len, self.head_dim
        k_proj: bsz * num_heads, src_len, self.head_dim
        key_padding_mask: bsz, src_len
        attn_mask: tgt_len, src_len
        """
        bsz, tgt_len, embed_dim = q_proj.size()
        bsz = bsz // self.num_heads
        src_len = k_proj.size(1)

        attn_energy = torch.bmm(q_proj, k_proj.transpose(1, 2)) + self.energy_bias

        attn_energy = attn_energy.view(bsz, self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            attn_energy = attn_energy.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(),
                float("-inf"),
            )

        return attn_energy

    def expected_alignment_train(self, p_choose, key_padding_mask):
        """
        Calculating expected alignment for MMA
        Mask is not need because p_choose will be 0 if masked
        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij
        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))
        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        """

        # p_choose: bsz * num_heads, tgt_len, src_len
        bsz_num_heads, tgt_len, src_len = p_choose.size()

        # cumprod_1mp : bsz * num_heads, tgt_len, src_len
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=self.eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, self.eps, 1.0)

        init_attention = p_choose.new_zeros([bsz_num_heads, 1, src_len])
        init_attention[:, :, 0] = 1.0

        previous_attn = [init_attention]

        for i in range(tgt_len):
            # p_choose: bsz * num_heads, tgt_len, src_len
            # cumprod_1mp_clamp : bsz * num_heads, tgt_len, src_len
            # previous_attn[i]: bsz * num_heads, 1, src_len
            # alpha_i: bsz * num_heads, src_len
            alpha_i = (
                p_choose[:, i]
                * cumprod_1mp[:, i]
                * torch.cumsum(previous_attn[i][:, 0] / cumprod_1mp_clamp[:, i], dim=1)
            ).clamp(0, 1.0)
            previous_attn.append(alpha_i.unsqueeze(1))

        # alpha: bsz * num_heads, tgt_len, src_len
        alpha = torch.cat(previous_attn[1:], dim=1)

        if self.mass_preservation:
            # Last token has the residual probabilities
            alpha[:, :, -1] = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0.0, 1.0)

        assert not torch.isnan(alpha).any(), "NaN detected in alpha."

        return alpha

    def expected_attention(
        self, alpha, query, key, value, key_padding_mask, incremental_state
    ):
        # monotonic attention, we will calculate milk here
        bsz_x_num_heads, tgt_len, src_len = alpha.size()
        bsz = int(bsz_x_num_heads / self.num_heads)

        q, k, _ = self.input_projections(query, key, None, "soft")
        soft_energy = self.attn_energy(q, k, key_padding_mask)

        assert list(soft_energy.size()) == [bsz, self.num_heads, tgt_len, src_len]

        soft_energy = soft_energy.view(bsz * self.num_heads, tgt_len, src_len)

        if incremental_state is not None:
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            monotonic_step = monotonic_cache["step"] + 1
            step_offset = 0
            if key_padding_mask is not None:
                if key_padding_mask[:, 0].any():
                    # left_pad_source = True:
                    step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
            monotonic_step += step_offset
            mask = lengths_to_mask(
                monotonic_step.view(-1), soft_energy.size(2), 1
            ).unsqueeze(1)

            soft_energy = soft_energy.masked_fill(~mask.bool(), float("-inf"))
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_sum = exp_soft_energy.sum(dim=2)
            beta = exp_soft_energy / exp_soft_energy_sum.unsqueeze(2)

        else:
            # bsz * num_heads, tgt_len, src_len
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_cumsum = torch.cumsum(exp_soft_energy, dim=2)

            if key_padding_mask is not None:
                if key_padding_mask.any():
                    exp_soft_energy_cumsum = (
                        exp_soft_energy_cumsum.view(
                            -1, self.num_heads, tgt_len, src_len
                        )
                        .masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(1), self.eps
                        )
                        .view(-1, tgt_len, src_len)
                    )

            inner_items = alpha / exp_soft_energy_cumsum

            beta = exp_soft_energy * torch.cumsum(
                inner_items.flip(dims=[2]), dim=2
            ).flip(dims=[2])

            beta = self.dropout_module(beta)

        assert not torch.isnan(beta).any(), "NaN detected in beta."

        return beta

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        *args,
        **kwargs,
    ):

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        # stepwise prob
        # p_choose: bsz * self.num_heads, tgt_len, src_len
        p_choose = self.p_choose(query, key, key_padding_mask)

        # expected alignment alpha
        # bsz * self.num_heads, tgt_len, src_len

        alpha = self.expected_alignment_train(p_choose, key_padding_mask)

        # expected attention beta
        # bsz * self.num_heads, tgt_len, src_len
        beta = self.expected_attention(
            alpha, query, key, value, key_padding_mask, incremental_state
        )

        attn_weights = beta

        _, _, v_proj = self.input_projections(None, None, value, "output")
        attn = torch.bmm(attn_weights.type_as(v_proj), v_proj)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, alpha, beta, p_choose


def exclusive_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    Implementing exclusive cumprod.
    There is cumprod in pytorch, however there is no exclusive mode.
    cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
    exclusive means cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor_size = list(tensor.size())
    tensor_size[dim] = 1
    return_tensor = safe_cumprod(
        torch.cat([torch.ones(tensor_size).type_as(tensor), tensor], dim=dim),
        dim=dim,
        eps=eps,
    )

    if dim == 0:
        return return_tensor[:-1]
    elif dim == 1:
        return return_tensor[:, :-1]
    elif dim == 2:
        return return_tensor[:, :, :-1]
    else:
        raise RuntimeError("Cumprod on dimension 3 and more is not implemented")


def safe_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    An implementation of cumprod to prevent precision issue.
    cumprod(x)
    = [x1, x1x2, x1x2x3, ....]
    = [exp(log(x1)), exp(log(x1) + log(x2)), exp(log(x1) + log(x2) + log(x3)), ...]
    = exp(cumsum(log(x)))
    """

    if (tensor + eps < 0).any().item():
        raise RuntimeError(
            "Safe cumprod can only take non-negative tensors as input."
            "Consider use torch.cumprod if you want to calculate negative values."
        )

    log_tensor = torch.log(tensor + eps)
    cumsum_log_tensor = torch.cumsum(log_tensor, dim)
    exp_cumsum_log_tensor = torch.exp(cumsum_log_tensor)
    return exp_cumsum_log_tensor


def lengths_to_mask(lengths, max_len: int, dim: int = 0, negative_mask: bool = False):
    """
    Convert a tensor of lengths to mask
    For example, lengths = [[2, 3, 4]], max_len = 5
    mask =
       [[1, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]]
    """
    assert len(lengths.size()) <= 2
    if len(lengths) == 2:
        if dim == 1:
            lengths = lengths.t()
        lengths = lengths
    else:
        lengths = lengths.unsqueeze(1)

    # lengths : batch_size, 1
    lengths = lengths.view(-1, 1)

    batch_size = lengths.size(0)
    # batch_size, max_len
    mask = torch.arange(max_len).expand(batch_size, max_len).type_as(lengths) < lengths

    if negative_mask:
        mask = ~mask

    if dim == 0:
        # max_len, batch_size
        mask = mask.t()

    return mask


def moving_sum(x, start_idx: int, end_idx: int):
    """
    From MONOTONIC CHUNKWISE ATTENTION
    https://arxiv.org/pdf/1712.05382.pdf
    Equation (18)
    x = [x_1, x_2, ..., x_N]
    MovingSum(x, start_idx, end_idx)_n = Sigma_{m=n−(start_idx−1)}^{n+end_idx-1} x_m
    for n in {1, 2, 3, ..., N}
    x : src_len, batch_size
    start_idx : start idx
    end_idx : end idx
    Example
    src_len = 5
    batch_size = 3
    x =
       [[ 0, 5, 10],
        [ 1, 6, 11],
        [ 2, 7, 12],
        [ 3, 8, 13],
        [ 4, 9, 14]]
    MovingSum(x, 3, 1) =
       [[ 0,  5, 10],
        [ 1, 11, 21],
        [ 3, 18, 33],
        [ 6, 21, 36],
        [ 9, 24, 39]]
    MovingSum(x, 1, 3) =
       [[ 3, 18, 33],
        [ 6, 21, 36],
        [ 9, 24, 39],
        [ 7, 17, 27],
        [ 4,  9, 14]]
    """
    assert start_idx > 0 and end_idx > 0
    assert len(x.size()) == 2
    src_len, batch_size = x.size()
    # batch_size, 1, src_len
    x = x.t().unsqueeze(1)
    # batch_size, 1, src_len
    moving_sum_weight = x.new_ones([1, 1, end_idx + start_idx - 1])

    moving_sum = (
        torch.nn.functional.conv1d(
            x, moving_sum_weight, padding=start_idx + end_idx - 1
        )
        .squeeze(1)
        .t()
    )
    moving_sum = moving_sum[end_idx:-start_idx]

    assert src_len == moving_sum.size(0)
    assert batch_size == moving_sum.size(1)

    return moving_sum
