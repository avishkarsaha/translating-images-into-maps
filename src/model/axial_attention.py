"""
Copied from https://github.com/lucidrains/axial-attention.git
"""

import torch
from torch import nn
from operator import itemgetter


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


# helper classes


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.fn(x) * self.g


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x) + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, emb_dim_index, dimensions):
        super().__init__()
        parameters = []
        total_dimensions = len(dimensions) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        for axial_dim, axial_dim_index in zip(dimensions, ax_dim_indexes):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = emb_dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            parameters.append(parameter)

        self.params = nn.ParameterList(parameters)

    def forward(self, x):
        for param in self.params:
            x = x + param
        return x


# classic multi-head attention


def attention(q, k, v, h):
    b, t, d = q.shape
    e = d // h

    merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
    q, k, v = map(merge_heads, (q, k, v))
    dots = torch.einsum("bie,bje->bij", q, k) * ((d // h) ** -0.5)
    dots = dots.softmax(dim=-1)
    out = torch.einsum("bij,bje->bie", dots, v)
    out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = (
            lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        )
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum("bie,bje->bij", q, k) * ((d // h) ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum("bij,bje->bie", dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class InducedSetAttention(nn.Module):
    def __init__(self, num_queries, dim, heads, dim_heads=None):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.attn_in = SelfAttention(dim, heads)
        self.attn_out = SelfAttention(dim, heads)

    def forward(self, x):
        b = x.shape[0]
        q = self.queries.expand(b, -1, -1)
        q_out = self.attn_in(q, x)
        out = self.attn_out(x, q_out)
        return out


# axial attention class


class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_dimensions=2,
        heads=8,
        dim_heads=None,
        dim_index=-1,
        sum_axial_out=True,
    ):
        assert (
            dim % heads
        ) == 0, "hidden dimension must be divisible by number of heads"
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = (
            dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        )

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(
                PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads))
            )

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert (
            len(x.shape) == self.total_dimensions
        ), "input tensor does not have the correct number of dimensions"
        assert (
            x.shape[self.dim_index] == self.dim
        ), "input tensor does not have the correct input dimension"

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        axial_attn = self.axial_attentions[0]
        out = axial_attn(x)
        for axial_attn in self.axial_attentions[1:]:
            out = axial_attn(out)
        return out
