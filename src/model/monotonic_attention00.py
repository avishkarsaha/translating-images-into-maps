import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicEnergy(nn.Module):
    def __init__(
        self,
        kdim,
        qdim,
        adim,
        atype,
        n_heads,
        init_r,
        bias=True,
        param_init="",
        conv1d=False,
        conv_kernel_size=5,
    ):
        """Energy function for the monotonic attenion.
        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of monotonic attention heads
            init_r (int): initial value for offset r
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method
            conv1d (bool): use 1D causal convolution for energy calculation
            conv_kernel_size (int): kernel size for 1D convolution
        """
        super().__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.key = None
        self.mask = None

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)

        if atype == "add":
            self.w_key = nn.Linear(kdim, adim)
            self.v = nn.Linear(adim, n_heads, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)
        elif atype == "scaled_dot":
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        else:
            raise NotImplementedError(atype)

        self.r = nn.Parameter(torch.Tensor([init_r]))
        logger.info("init_r is initialized with %d" % init_r)

        self.conv1d = None
        if conv1d:
            self.conv1d = CausalConv1d(
                in_channels=kdim,
                out_channels=kdim,
                kernel_size=conv_kernel_size,
                param_init=param_init,
            )
            # padding=(conv_kernel_size - 1) // 2

        if atype == "add":
            self.v = nn.utils.weight_norm(self.v, name="weight", dim=0)
            # initialization
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == "scaled_dot":
            if param_init == "xavier_uniform":
                self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            "===== Initialize %s with Xavier uniform distribution ====="
            % self.__class__.__name__
        )
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False, boundary_leftmost=0):
        """Compute monotonic energy.
        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Returns:
            e (FloatTensor): `[B, H_ma, qlen, klen]`
        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            # 1d conv
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key))
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H_ma, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat(
                    [1, self.n_heads, 1, 1]
                )  # `[B, H_ma, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (
                    self.mask.size(),
                    (bs, self.n_heads, qlen, klen),
                )

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, H_ma, qlen, d_k]`
        m = self.mask

        if self.atype == "add":
            k = self.key.unsqueeze(2)  # `[B, H_ma, 1, klen, d_k]`
            # Truncate encoder memories
            if boundary_leftmost > 0:
                k = k[:, :, :, boundary_leftmost:]
                klen = k.size(3)
                if m is not None:
                    m = m[:, :, :, boundary_leftmost:]
            e = torch.relu(k + query.unsqueeze(3))  # `[B, H_ma, qlen, klen, d_k]`
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)  # `[B, qlen, klen, H_ma]`
        elif self.atype == "scaled_dot":
            k = self.key.transpose(3, 2)
            e = torch.matmul(query, k) / self.scale

        if self.r is not None:
            e = e + self.r
        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        assert e.size() == (bs, self.n_heads, qlen, klen), (
            e.size(),
            (bs, self.n_heads, qlen, klen),
        )
        return e


class ChunkEnergy(nn.Module):
    def __init__(self, kdim, qdim, adim, atype, n_heads=1, bias=True, param_init=""):
        """Energy function for the chunkwise attention.
        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of chunkwise attention heads
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method
        """
        super().__init__()

        self.key = None
        self.mask = None

        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)

        if atype == "add":
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, n_heads, bias=False)
        elif atype == "scaled_dot":
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            if param_init == "xavier_uniform":
                self.reset_parameters(bias)
        else:
            raise NotImplementedError(atype)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            "===== Initialize %s with Xavier uniform distribution ====="
            % self.__class__.__name__
        )
        # NOTE: see https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(
        self,
        key,
        query,
        mask,
        cache=False,
        boundary_leftmost=0,
        boundary_rightmost=10e6,
    ):
        """Compute chunkwise energy.
        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Returns:
            e (FloatTensor): `[B, H_ca, qlen, klen]`
        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()  # `[B, H_ca, klen, d_k]`
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat(
                    [1, self.n_heads, 1, 1]
                )  # `[B, H_ca, qlen, klen]`
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (
                    self.mask.size(),
                    (bs, self.n_heads, qlen, klen),
                )

        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()  # `[B, H_ca, qlen, d_k]`
        m = self.mask

        if self.atype == "add":
            k = self.key.unsqueeze(2)  # `[B, H_ca, 1, klen, d_k]`
            # Truncate
            k = k[:, :, :, boundary_leftmost:boundary_rightmost]
            klen = k.size(3)
            if m is not None:
                m = m[:, :, :, boundary_leftmost:boundary_rightmost]

            r = torch.relu(k + query.unsqueeze(3))  # `[B, H_ca, qlen, klen, d_k]`
            r = (
                r.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            )  # `[B, qlen, klen, H_ca * d_k]`
            r = self.v(r).permute(0, 3, 1, 2).contiguous()  # `[B, H_ca, qlen, klen]`
        elif self.atype == "scaled_dot":
            k = self.key.transpose(3, 2)
            r = torch.matmul(query, k) / self.scale

        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=r.dtype).numpy().dtype).min)
            r = r.masked_fill_(m == 0, NEG_INF)
        assert r.size() == (bs, self.n_heads, qlen, klen), (
            r.size(),
            (bs, self.n_heads, qlen, klen),
        )
        return r


class MonotonicAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(MonotonicAttention, self).__init__()

        self.n_heads = n_heads

        self.monotonic_energy = MonotonicEnergy(
            kdim=embed_dim,
            qdim=embed_dim,
            adim=embed_dim,
            atype="scaled_dot",
            n_heads=n_heads,
            init_r=0,
            param_init="xavier_uniform",
        )

        self.chunk_energy = ChunkEnergy(
            kdim=embed_dim,
            qdim=embed_dim,
            adim=embed_dim,
            atype="scaled_dot",
            n_heads=n_heads,
            param_init="xavier_uniform",
        )
        self.dropout_attn = nn.Dropout(p=dropout)  # for beta
        self.bd_offset = 0

    def forward(self, key, value, query, aw_prev=None, mask=None, cache=None):

        bs, klen = key.size()[:2]
        qlen = query.size(1)

        if aw_prev is None:
            # aw_prev = [1, 0, 0 ... 0]
            aw_prev = key.new_zeros(bs, self.n_heads_ma, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.n_heads_ma, 1, 1)

        # Compute monotonic energy
        e_ma = self.monotonic_energy(
            key, query, mask, cache=cache, boundary_leftmost=self.bd_offset
        )  # `[B, H_ma, qlen, klen]`

        p_choose = torch.sigmoid(
            add_gaussian_noise(e_ma, self.noise_std)
        )  # `[B, H_ma, qlen, klen]`

        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose = safe_cumprod(
            1 - p_choose, eps=self.eps
        )  # `[B, H_ma, qlen, klen]`

        # Compute recurrence relation solution
        alpha = []
        for i in range(qlen):
            denom = (
                1
                if self.no_denom
                else torch.clamp(
                    cumprod_1mp_choose[:, :, i : i + 1], min=self.eps, max=1.0
                )
            )
            aw_prev = (
                p_choose[:, :, i : i + 1]
                * cumprod_1mp_choose[:, :, i : i + 1]
                * torch.cumsum(aw_prev / denom, dim=-1)
            )  # `[B, H_ma, 1, klen]`
            # Mask the right part from the trigger point
            alpha.append(aw_prev)

        alpha = (
            torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
        )  # `[B, H_ma, qlen, klen]`
        alpha_masked = alpha.clone()

        # Compute chunk energy
        e_ca = self.chunk_energy(
            key,
            query,
            mask,
        )  # `[B, (H_ma*)H_ca, qlen, ken]`

        beta = efficient_chunkwise_attention(
            alpha_masked,
            e_ca,
            mask,
            chunk_size=-1,
            n_heads_chunk=self.n_heads,
            sharpening_factor=1.0,
            share_chunkwise_attention=True,
        )
        beta = self.dropout_attn(beta)

        value = self.w_value(value).view(
            bs, -1, self.n_heads_ma * self.n_heads_ca, self.d_k
        )
        value = value.transpose(2, 1).contiguous()  # `[B, H_ma * H_ca, klen, d_k]`
        cv = torch.matmul(
            alpha if self.w == 1 else beta, value
        )  # `[B, H_ma * H_ca, qlen, d_k]`
        cv = (
            cv.transpose(2, 1)
            .contiguous()
            .view(bs, -1, self.n_heads_ma * self.n_heads_ca * self.d_k)
        )
        cv = self.w_out(cv)  # `[B, qlen, adim]`


def add_gaussian_noise(xs, std):
    """Add Gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.
    Args:
        x (FloatTensor): `[B, H, qlen, klen]`
    Returns:
        x (FloatTensor): `[B, H, qlen, klen]`
    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].
    Args:
        x (FloatTensor): `[B, H, qlen, klen]`
    Returns:
        x (FloatTensor): `[B, H, qlen, klen]`
    """
    return torch.cumsum(
        torch.cat(
            [x.new_zeros(x.size(0), x.size(1), x.size(2), 1), x[:, :, :, :-1]], dim=-1
        ),
        dim=-1,
    )


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].
    Args:
        x (FloatTensor): `[B, H, qlen, klen]`
    Returns:
        x (FloatTensor): `[B, H, qlen, klen]`
    """
    return torch.cumprod(
        torch.cat(
            [x.new_ones(x.size(0), x.size(1), x.size(2), 1), x[:, :, :, :-1]], dim=-1
        ),
        dim=-1,
    )


def efficient_chunkwise_attention(
    alpha,
    u,
    mask,
    chunk_size,
    n_heads_chunk,
    sharpening_factor,
    share_chunkwise_attention,
):
    """Compute chunkwise attention efficiently by clipping logits at training time.
    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        n_heads_chunk (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`
    """
    bs, n_heads_mono, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)  # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if n_heads_chunk > 1:
        alpha = alpha.repeat([1, 1, n_heads_chunk, 1, 1])
    if n_heads_mono > 1 and not share_chunkwise_attention:
        u = u.view(bs, n_heads_mono, n_heads_chunk, qlen, klen)
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=-1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    if chunk_size == -1:
        # infinite lookback attention
        # inner_items = alpha * sharpening_factor / torch.cumsum(softmax_exp, dim=-1)
        # beta = softmax_exp * torch.cumsum(inner_items.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        # beta = beta.masked_fill(mask.unsqueeze(1), 0)
        # beta = beta / beta.sum(dim=-1, keepdim=True)

        softmax_denominators = torch.cumsum(softmax_exp, dim=-1)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(
            alpha * sharpening_factor / softmax_denominators, back=0, forward=klen - 1
        )
    else:
        softmax_denominators = moving_sum(softmax_exp, back=chunk_size - 1, forward=0)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(
            alpha * sharpening_factor / softmax_denominators,
            back=0,
            forward=chunk_size - 1,
        )
    return beta.view(bs, -1, qlen, klen)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.
    Args:
        x (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`
        back (int):
        forward (int):
    Returns:
        x_sum (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`
    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.view(-1, klen)
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(
        x, pad=[back, forward]
    )  # `[B * H_ma * H_ca * qlen, back + klen + forward]`
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum
