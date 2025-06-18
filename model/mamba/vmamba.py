import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import sys
import numpy as np
from model.refinement import RefineGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from model.encoder import *

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

sys.path.append(".")
from .csm_triton import cross_scan_fn, cross_merge_fn
# try:
#     from .csm_triton import cross_scan_fn, cross_merge_fn
# except:
#     from csm_triton import cross_scan_fn, cross_merge_fn

from .csms6s import selective_scan_fn, selective_scan_flop_jit
# try:
#     from .csms6s import selective_scan_fn, selective_scan_flop_jit
# except:
#     from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
from .mamba2.ssd_minimal import selective_scan_chunk_fn
# try:
#     from .mamba2.ssd_minimal import selective_scan_chunk_fn
# except:
#     from mamba2.ssd_minimal import selective_scan_chunk_fn

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def generate_scan_forward_fmap(x):
    B, C, H, W = x.size()
    L = H * W
    # x 划分象限
    x_quadrant1 = x[:, :, :H//2, W//2:].contiguous().view(B, -1, L//4)
    x_quadrant2 = x[:, :, :H//2, :W//2].contiguous().view(B, -1, L//4)
    x_quadrant3 = x[:, :, H//2:, :W//2].contiguous().view(B, -1, L//4)
    x_quadrant4 = x[:, :, H//2:, W//2:].contiguous().view(B, -1, L//4)
    zz_path_quadrant, zz_path_rev_quadrant = generate_zigzag_forw_back(L//4, device=x.device)
    # 从中心到边缘扫描路径 1
    x_quadrant1_loc1 = forward_permutation(x_quadrant1, zz_path_quadrant[4])
    x_quadrant2_loc1 = forward_permutation(x_quadrant2, zz_path_quadrant[6])
    x_quadrant3_loc1 = forward_permutation(x_quadrant3, zz_path_quadrant[2])
    x_quadrant4_loc1 = forward_permutation(x_quadrant4, zz_path_quadrant[0])
    xs_loc1 = torch.cat([x_quadrant2_loc1, x_quadrant1_loc1, x_quadrant3_loc1, x_quadrant4_loc1], dim=-1)
    # 从中心到边缘扫描路径 2
    x_quadrant1_loc2 = forward_permutation(x_quadrant1, zz_path_quadrant[5])
    x_quadrant2_loc2 = forward_permutation(x_quadrant2, zz_path_quadrant[7])
    x_quadrant3_loc2 = forward_permutation(x_quadrant3, zz_path_quadrant[3])
    x_quadrant4_loc2 = forward_permutation(x_quadrant4, zz_path_quadrant[1])
    xs_loc2 = torch.cat([x_quadrant2_loc2, x_quadrant1_loc2, x_quadrant3_loc2, x_quadrant4_loc2], dim=-1)
    # 从边缘到中心扫描路径 1
    x_quadrant1_glb1 = forward_permutation(x_quadrant1, zz_path_quadrant[2])
    x_quadrant2_glb1 = forward_permutation(x_quadrant2, zz_path_quadrant[0])
    x_quadrant3_glb1 = forward_permutation(x_quadrant3, zz_path_quadrant[4])
    x_quadrant4_glb1 = forward_permutation(x_quadrant4, zz_path_quadrant[6])
    xs_glb1 = torch.cat([x_quadrant2_glb1, x_quadrant1_glb1, x_quadrant3_glb1, x_quadrant4_glb1], dim=-1)
    # 从边缘到中心扫描路径 2
    x_quadrant1_glb2 = forward_permutation(x_quadrant1, zz_path_quadrant[3])
    x_quadrant2_glb2 = forward_permutation(x_quadrant2, zz_path_quadrant[1])
    x_quadrant3_glb2 = forward_permutation(x_quadrant3, zz_path_quadrant[5])
    x_quadrant4_glb2 = forward_permutation(x_quadrant4, zz_path_quadrant[7])
    xs_glb2 = torch.cat([x_quadrant2_glb2, x_quadrant1_glb2, x_quadrant3_glb2, x_quadrant4_glb2], dim=-1)
    # 空间连续性检查扫描路径 8
    xin = x.view(B, -1, L)
    zz_path, zz_path_rev = generate_zigzag_forw_back(L, device=x.device)
    xs_space_continuity_1 = forward_permutation(xin, zz_path[0])
    xs_space_continuity_2 = forward_permutation(xin, zz_path[1])
    xs_space_continuity_3 = forward_permutation(xin, zz_path[2])
    xs_space_continuity_4 = forward_permutation(xin, zz_path[3])
    xs_space_continuity_5 = forward_permutation(xin, zz_path[4])
    xs_space_continuity_6 = forward_permutation(xin, zz_path[5])
    xs_space_continuity_7 = forward_permutation(xin, zz_path[6])
    xs_space_continuity_8 = forward_permutation(xin, zz_path[7])
    xs = torch.stack([xs_loc1, xs_loc2, xs_glb1, xs_glb2, 
                        xs_space_continuity_1, xs_space_continuity_2, xs_space_continuity_3, xs_space_continuity_4,
                        xs_space_continuity_5, xs_space_continuity_6, xs_space_continuity_7, xs_space_continuity_8], dim=1)
    return xs

def generate_scan_backward_fmap(xs, H, W):
    B, K, C, L = xs.size()
    zz_path_quadrant, zz_path_rev_quadrant = generate_zigzag_forw_back(L//4, device=xs.device)
    # x 分组：局部组、全局组、空间连续性检查组
    xs_loc, xs_glb, xs_space_continuity = xs[:, :2], xs[:, 2:4], xs[:, 4:]

    # 划分象限
    x_quadrant2_loc, x_quadrant1_loc, x_quadrant3_loc, x_quadrant4_loc = torch.chunk(xs_loc, 4, dim=-1)
    x_quadrant2_glb, x_quadrant1_glb, x_quadrant3_glb, x_quadrant4_glb = torch.chunk(xs_glb, 4, dim=-1)
    
    # 从中心到边缘扫描路径 1
    x_quadrant1_loc1 = backward_permutation(x_quadrant1_loc[:, 0], zz_path_rev_quadrant[4]).view(B, -1, H//2, W//2)
    x_quadrant2_loc1 = backward_permutation(x_quadrant2_loc[:, 0], zz_path_rev_quadrant[6]).view(B, -1, H//2, W//2)
    x_quadrant3_loc1 = backward_permutation(x_quadrant3_loc[:, 0], zz_path_rev_quadrant[2]).view(B, -1, H//2, W//2)
    x_quadrant4_loc1 = backward_permutation(x_quadrant4_loc[:, 0], zz_path_rev_quadrant[0]).view(B, -1, H//2, W//2)
    xs_loc1 = torch.cat([torch.cat([x_quadrant2_loc1, x_quadrant1_loc1], dim=3), torch.cat([x_quadrant3_loc1, x_quadrant4_loc1], dim=3)], dim=2).contiguous()
    xs_loc1 = xs_loc1.view(B, -1, L)
    # 从中心到边缘扫描路径 2
    x_quadrant1_loc2 = backward_permutation(x_quadrant1_loc[:, 1], zz_path_rev_quadrant[5]).view(B, -1, H//2, W//2)
    x_quadrant2_loc2 = backward_permutation(x_quadrant2_loc[:, 1], zz_path_rev_quadrant[7]).view(B, -1, H//2, W//2)
    x_quadrant3_loc2 = backward_permutation(x_quadrant3_loc[:, 1], zz_path_rev_quadrant[3]).view(B, -1, H//2, W//2)
    x_quadrant4_loc2 = backward_permutation(x_quadrant4_loc[:, 1], zz_path_rev_quadrant[1]).view(B, -1, H//2, W//2)
    xs_loc2 = torch.cat([torch.cat([x_quadrant2_loc2, x_quadrant1_loc2], dim=3), torch.cat([x_quadrant3_loc2, x_quadrant4_loc2], dim=3)], dim=2).contiguous()
    xs_loc2 = xs_loc2.view(B, -1, L)
    # 从边缘到中心扫描路径 1
    x_quadrant1_glb1 = backward_permutation(x_quadrant1_glb[:, 0], zz_path_rev_quadrant[2]).view(B, -1, H//2, W//2)
    x_quadrant2_glb1 = backward_permutation(x_quadrant2_glb[:, 0], zz_path_rev_quadrant[0]).view(B, -1, H//2, W//2)
    x_quadrant3_glb1 = backward_permutation(x_quadrant3_glb[:, 0], zz_path_rev_quadrant[4]).view(B, -1, H//2, W//2)
    x_quadrant4_glb1 = backward_permutation(x_quadrant4_glb[:, 0], zz_path_rev_quadrant[6]).view(B, -1, H//2, W//2)
    xs_glb1 = torch.cat([torch.cat([x_quadrant2_glb1, x_quadrant1_glb1], dim=3), torch.cat([x_quadrant3_glb1, x_quadrant4_glb1], dim=3)], dim=2).contiguous()
    xs_glb1 = xs_glb1.view(B, -1, L)
    # 从边缘到中心扫描路径 2
    x_quadrant1_glb2 = backward_permutation(x_quadrant1_glb[:, 1], zz_path_rev_quadrant[3]).view(B, -1, H//2, W//2)
    x_quadrant2_glb2 = backward_permutation(x_quadrant2_glb[:, 1], zz_path_rev_quadrant[1]).view(B, -1, H//2, W//2)
    x_quadrant3_glb2 = backward_permutation(x_quadrant3_glb[:, 1], zz_path_rev_quadrant[5]).view(B, -1, H//2, W//2)
    x_quadrant4_glb2 = backward_permutation(x_quadrant4_glb[:, 1], zz_path_rev_quadrant[7]).view(B, -1, H//2, W//2)
    xs_glb2 = torch.cat([torch.cat([x_quadrant2_glb2, x_quadrant1_glb2], dim=3), torch.cat([x_quadrant3_glb2, x_quadrant4_glb2], dim=3)], dim=2).contiguous()
    xs_glb2 = xs_glb2.view(B, -1, L)
    # 空间连续性检查扫描路径 8
    zz_path, zz_path_rev = generate_zigzag_forw_back(L, device=xs.device)
    xs_space_continuity_1 = backward_permutation(xs_space_continuity[:, 0], zz_path_rev[0])
    xs_space_continuity_2 = backward_permutation(xs_space_continuity[:, 1], zz_path_rev[1])
    xs_space_continuity_3 = backward_permutation(xs_space_continuity[:, 2], zz_path_rev[2])
    xs_space_continuity_4 = backward_permutation(xs_space_continuity[:, 3], zz_path_rev[3])
    xs_space_continuity_5 = backward_permutation(xs_space_continuity[:, 4], zz_path_rev[4])
    xs_space_continuity_6 = backward_permutation(xs_space_continuity[:, 5], zz_path_rev[5])
    xs_space_continuity_7 = backward_permutation(xs_space_continuity[:, 6], zz_path_rev[6])
    xs_space_continuity_8 = backward_permutation(xs_space_continuity[:, 7], zz_path_rev[7])
    xout = xs_loc1 + xs_loc2 + xs_glb1 + xs_glb2 + xs_space_continuity_1 + xs_space_continuity_2 + xs_space_continuity_3 + xs_space_continuity_4 + xs_space_continuity_5 + xs_space_continuity_6 + xs_space_continuity_7 + xs_space_continuity_8
    return xout

def forward_permutation(xz_main, _perm):
    return xz_main[:, :, _perm].contiguous()  # [B,C,L]

def backward_permutation(o_main, _perm_rev):
    return o_main[:, :, _perm_rev].contiguous()  # [B,C,L]


def generate_zigzag_forw_back(num_patches, device, scan_type='zigzagN8'):
    patch_side_len = int(math.sqrt(num_patches))
    if (
        scan_type.startswith("zigzagN")
    ):
        if scan_type.startswith("zigzagN"):
            # 三原则扫描，12条路径
            zigzag_num = int(scan_type.replace("zigzagN", ""))
            if zigzag_num == 12:
                _zz_paths = zigzag_3p_path(N=patch_side_len)
            elif zigzag_num == 8:
                _zz_paths = zigzag_3p_path(N=patch_side_len)[4:]
            if scan_type.startswith("zigzagN"):
                zz_paths = _zz_paths[:zigzag_num]
                assert (
                    len(zz_paths) == zigzag_num
                ), f"{len(zz_paths)} != {zigzag_num}"
            else:
                raise ValueError("scan_type should be xx")
        else:
            raise ValueError(f"scan_type {scan_type} doenst match")
        #############
        zz_paths_rev = [reverse_permut_np(_) for _ in zz_paths]
        # zz_paths = zz_paths * depth
        # zz_paths_rev = zz_paths_rev * depth
        zz_paths = [torch.from_numpy(_).to(device) for _ in zz_paths]
        zz_paths_rev = [torch.from_numpy(_).to(device) for _ in zz_paths_rev]
        assert len(zz_paths) == len(
            zz_paths_rev
        ), f"{len(zz_paths)} != {len(zz_paths_rev)}"
    else:
        raise ValueError("scan_type doesn't match")
    return zz_paths, zz_paths_rev

def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse

def zigzag_path(N):
    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths

def zigzag4_path(N):
    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    half_N = N // 2
    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, half_N - 1, 1, -1),
        (half_N - 1, 0, -1, 1),
        (half_N - 1, half_N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(half_N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(half_N, start_row, start_col, dir_row, dir_col))

    # paths 总共有8条路径
    offset_dict = {
        '0': 0,
        '1': half_N,
        '2': 2 * half_N * half_N,
        '3': (2 * half_N + 1) * half_N,
    }

    codebooks = {}  # 记录4个字典，每个字典保存各个象限扫描路径的对应关系
    for group, offset in offset_dict.items():
        code_dict = {}
        for row in range(half_N):
            for col in range(half_N):
                key = row * half_N + col
                code_dict[f'{key}'] = key + offset + row * half_N
        codebooks[group] = code_dict

    group_path_dicts = {}
    for group, code_dict in codebooks.items():
        group_path_dicts[group] = []
        for _index, path in enumerate(paths):
            # list(map(lambda x:x+5, ilist))
            group_path = list(map(lambda x:code_dict[f'{x}'], path))
            group_path_dicts[group].append(np.array(group_path))

    return group_path_dicts

def zigzag_3p_path(N):
    half_N = N // 2
    path_4q = []
    group_path_dicts = zigzag4_path(N)
    group_index_pairs = [
        [('0', 6), ('1', 4), ('2', 2), ('3', 0), ],
        [('0', 7), ('1', 5), ('2', 3), ('3', 1), ],
        [('0', 0), ('1', 2), ('2', 4), ('3', 6), ],
        [('0', 1), ('1', 3), ('2', 5), ('3', 7), ],
    ]
    for group_index_pair in group_index_pairs:
        group_paths = []
        for group, index in group_index_pair:
            group_path = group_path_dicts[group][index].reshape(half_N, half_N)
            group_paths.append(group_path)
        path_4q.append(np.stack([np.stack(group_paths[:2], axis=1), np.stack(group_paths[2:], axis=1)], axis=0).reshape(N*N))
    path_3p = path_4q + zigzag_path(N)
    return path_3p


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        # self.weight[:, :, None, None] 表示weight增加了最后两个维度，也就是 (in_channel, out_channel, 1, 1) 为 1x1 卷积核
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)  # 把 (b, h, w, c) 通过隔行分割的方式分成 4 个子图，然后在通道维合并，变成 (b, h/2, w/2, c*4)
        x = self.norm(x)
        x = self.reduction(x)  # 把 (b, h/2, w/2, c*4) 映射为 (b, h/2, w/2, 2*dim)，缩减维度大小

        return x


class PatchCombining2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_combining_pad_channel_first if channel_first else self._patch_combining_pad_channel_last
        self.increase = Linear(dim // 4, (dim // 2) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(dim // 4)

    @staticmethod
    def _patch_combining_pad_channel_last(x: torch.Tensor):
        H, W, C = x.shape[-3:]
        C_out = C//4
        x_out = torch.zeros((2*H, 2*W, C_out), dtype=x.dtype(), device=x.device)
        x0, x1, x2, x3 = torch.chunk(x, chunks=4, dim=-1)
        x_out[..., 0::2, 0::2, :] = x0  # ... H/2 W/2 C_out
        x_out[..., 1::2, 0::2, :] = x1  # ... H/2 W/2 C_out
        x_out[..., 0::2, 1::2, :] = x2  # ... H/2 W/2 C_out
        x_out[..., 1::2, 1::2, :] = x3  # ... H/2 W/2 C_out
        return x_out

    @staticmethod
    def _patch_combining_pad_channel_first(x: torch.Tensor):
        C, H, W = x.shape[-3:]
        C_out = C//4
        x_out = torch.zeros((C_out, 2*H, 2*W), dtype=x.dtype(), device=x.device)
        x0, x1, x2, x3 = torch.chunk(x, chunks=4, dim=1)
        x_out[..., 0::2, 0::2] = x0  # ... H/2 W/2
        x_out[..., 1::2, 0::2] = x1  # ... H/2 W/2
        x_out[..., 0::2, 1::2] = x2  # ... H/2 W/2
        x_out[..., 1::2, 1::2] = x3  # ... H/2 W/2
        return x_out

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.increase(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.view(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:  # 对空间维度进行 softmax
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:  # 对通道维度进行 softmax
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # dt_proj 是一个全连接投影，维度变化为  dt_rank -> d_inner，dt_rank 是 delta 的秩

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class S32DBSv0:
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # scan path ============
        scan_path=None,
        rev_scan_path=None,
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        self.k_group = 1
        self.scan_path = scan_path
        self.rev_scan_path = rev_scan_path
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.con_proj = nn.Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,  # groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.con_conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,  # groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, condition: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        c = self.con_proj(condition)  # (b, hc, wc, d)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z) # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        c = c.permute(0, 3, 1, 2).contiguous()
        c = self.con_conv2d(c) # (b, d, hc, wc)
        c = self.act(c)
        selective_scan = partial(selective_scan_fn, backend="oflex")  # backend="mamba"
        
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        xs = forward_permutation(x.view(B, D, L), self.scan_path).view(B, self.k_group, D, L)

        con_c, con_h, con_w = c.size()[1:]
        con_l = con_h * con_w
        if con_h == 1:
            cs = c.view(B, D, -1)  # (b, d, 1)
            cs = cs.unsqueeze(1).contiguous()  # (b, 1, d, 1)
        else:
            cs = forward_permutation(c.view(B, con_c, con_l), self.scan_path).view(B, self.k_group, con_c, con_l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs + cs, self.x_proj_weight)  # c = (dt_rank + d_state * 2)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # 从论文可知，dts, Bs, Cs 是由 x 映射得到
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  # 将 dts 做个简单的全连接映射

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y = 0
        for i in range(K):
            each_out_y = out_y[:, i]
            y += backward_permutation(each_out_y, self.rev_scan_path)
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class S32DBS(nn.Module, S32DBSv0):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # scan path ============
        scan_path=None,
        rev_scan_path=None,
        # ======================
        **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, 
            scan_path=scan_path, rev_scan_path=rev_scan_path,
        )
        self.__initv0__(seq=("seq" in forward_type), **kwargs)


# =====================================================
class VSSBlockBS(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        _SS2D: type = S32DBS,
        num_patches=None,
        scan_idx=None,
        scan_type='zigzagN12',
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            scan_paths, rev_scan_paths = generate_zigzag_forw_back(num_patches, device, scan_type=scan_type)
            self.norm = nn.Sequential(
                        (nn.Identity() if not channel_first else Permute(0, 2, 3, 1)),
                        norm_layer(hidden_dim),  # b, h, w, c
                        (nn.Identity() if not channel_first else Permute(0, 3, 1, 2)),
                    )
            self.op = _SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ==========================
                scan_path=scan_paths[scan_idx],
                rev_scan_path=rev_scan_paths[scan_idx],
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = nn.Sequential(
                        (nn.Identity() if not channel_first else Permute(0, 2, 3, 1)),
                        norm_layer(hidden_dim),
                        (nn.Identity() if not channel_first else Permute(0, 3, 1, 2)),
                    )
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, inputs):
        x = inputs[0]
        c0 = inputs[1]
        if c0 is None:
            c = x
        else:
            c = c0
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x, c)))
            else:
                x = x + self.drop_path(self.op(self.norm(x), self.norm(c)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return [x, c0]

    def forward(self, inputs):
        input, condition = inputs
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, [input, condition])
        else:
            return self._forward([input, condition])


class VSSMBS_Encoder(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        con_in_chans=512,
        num_classes=1000, 
        depths=[2, 2, 8, 2], 
        dims=[96, 192, 384, 768], 
        nums_patches=[64*64, 32*32, 16*16, 8*8],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        scan_type='zigzagN12',
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        _SS2D=S32DBS,
        # =========================
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.con_in_chans = con_in_chans
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        scan_idx_lists = list(map(lambda x:x%12, np.arange(0, sum(depths), 1).tolist()))
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)
        if con_in_chans != 0:
            self.con_patch_embed = _make_patch_embed(con_in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                scan_idx_list=scan_idx_lists[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                num_patches=nums_patches[i_layer],
                scan_type=scan_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
                con_in_chans=con_in_chans,
            ))

        self.latent = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute1=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            head=nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1),  # B,C,H,W
            permute2=(Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity()),  # B,H,W,C
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),  # b,c,h,w
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        scan_idx_list=[0, 0],
        num_patches=None,
        scan_type='zigzagN12',
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        _SS2D=S32DBS,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlockBS(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
                num_patches=num_patches,
                scan_idx=scan_idx_list[d],
                scan_type=scan_type,
            ))
        
        return nn.ModuleList(
            [
                nn.Sequential(*blocks,),
                downsample,  # 对x下采样
                downsample if kwargs['con_in_chans'] != 0 else nn.Identity(),  # 对c下采样
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        concat_feature = []
        x = self.patch_embed(x)  # output_size is 1/4 of input_size
        if c is not None:
            c = self.con_patch_embed(c)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
            if c is not None:
                c = c + pos_embed
        for layer in self.layers:
            x, c = layer[0]([x, c])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            x = layer[1](x)
            concat_feature.append(x)
            if c is not None:
                c = layer[2](c)
        x = self.latent(x)
        return x, c, concat_feature[:-1]

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# Mamba Decoder
class VSSMBS_Decoder(nn.Module):
    def __init__(
        self, 
        out_chans=3, 
        depths=[2, 9, 2, 2], 
        dims=[768, 384, 192, 96], 
        nums_patches=[8*8, 16*16, 32*32, 64*64],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v0",
        scan_type='zigzagN12',
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1,
        norm_layer="LN", # "BN", "LN2D"
        upsample_version: str = "v2", # "v1", "v2", "v3"
        use_checkpoint=False,  
        # =========================
        _SS2D=S32DBS,
        scan_idx_lists=None,
        # =========================
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** ((self.num_layers - 1) - i_layer)) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule，从 0 到 drop_path_rate 取元素总数为 sum(depths) 的等差数列
        if scan_idx_lists is None:
            scan_idx_lists = list(map(lambda x:x%12, np.arange(0, sum(depths), 1).tolist())) 
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        _make_upsample = dict(
            v1=PatchCombining2D, 
            v2=self._make_upsample, 
            v3=self._make_upsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(upsample_version, None)  # 本案例，v3，输出尺寸缩小一半

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 本案例，self.num_layers = 4，前三层分别缩小一半尺寸，最后一层尺寸不变
            upsample = _make_upsample(
                        self.dims[i_layer], 
                        self.dims[i_layer + 1],
                        norm_layer=norm_layer,
                        channel_first=self.channel_first,
                    ) if i_layer < (self.num_layers - 1) else _make_upsample(
                        self.dims[-1], 
                        self.dims[-1],
                        norm_layer=norm_layer,
                        channel_first=self.channel_first,
                    )

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=upsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                scan_idx_list=scan_idx_lists[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                num_patches=nums_patches[i_layer],
                scan_type=scan_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))

        self.inpainting = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute1=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            head1=nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1),  # B,C,H,W
            act1=nn.ELU(),
            head2=nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1),  # B,C,H,W
            act2=nn.ELU(),
            head3=nn.Conv2d(self.num_features, out_chans, kernel_size=3, stride=1, padding=1),  # B,C,H,W
            permute2=(Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity()),  # B,H,W,C
            act3=nn.Tanh(),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_upsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.ConvTranspose2d(dim, out_dim, kernel_size=2, stride=2, padding=1, output_padding=1, dilation=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_upsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.ConvTranspose2d(dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        upsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        scan_idx_list=[0, 0],
        num_patches=None,
        scan_type='zigzagN12',
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        _SS2D=S32DBS,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlockBS(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
                num_patches=num_patches,
                scan_idx=scan_idx_list[d],
                scan_type=scan_type,
            ))
        
        return nn.ModuleList(
            [
                nn.Sequential(*blocks,),
                upsample,
                upsample,
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, concat_feature=None):
        for i, layer in enumerate(self.layers):
            if concat_feature:
                if i < self.num_layers - 2:
                    ef = concat_feature[self.num_layers - 2 - 1 - i]
                    x = x + ef
            x, c = layer[0]([x, c])
            x = layer[1](x)
            c = layer[2](c)
        x = self.inpainting(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "inpainting.norm")
        change_name("head", "inpainting.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)





# ================ Mamba Encoder ==========================

class VSSMBSSegNet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.permute_start = nn.Identity() if opts.channel_first else Permute(0, 2, 3, 1)  # 从 b,c,h,w 转换到 是否channel_first
        self.permute_end = nn.Identity() if opts.channel_first else Permute(0, 3, 1, 2)  # 从 是否channel_first 转换到 b,c,h,w
        self.enc = vmambabs_seg_tiny_s1l8_enc_256(in_chans=3+opts.con_in_chans, con_in_chans=opts.con_in_chans, channel_first=opts.channel_first)
        self.dec = vmambabs_seg_tiny_s1l8_dec_256(opts.channel_first)
        # self.refinement = RefineGenerator(img_dim=3, con_dim=0, cnum=32)

    def forward(self, x, c, mask=None):
        c0 = c.clone()
        x0 = x.clone()
        x = torch.cat([x, c], dim=1)
        x = self.permute_start(x)
        c = self.permute_start(c)
        x, c, concat_feature = self.enc(x, c)
        x = self.dec(x, c, concat_feature)  # b,h,w,c
        x = self.permute_end(x)  # b,c,h,w
        # x2 = self.refinement(x)
        if mask is not None:
            x = x * mask + x0 * (1 - mask)
        return x


class Reshape(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.reshape(*self.dim)

def vmambabs_seg_tiny_s1l8_enc_256(in_chans, con_in_chans, channel_first=False, pretrained=None):
    return VSSMBS_Encoder(
        depths=[2, 2, 8, 4], dims=96, nums_patches=[64*64, 32*32, 16*16, 8*8], drop_path_rate=0.2, 
        patch_size=4, in_chans=in_chans, con_in_chans=con_in_chans,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", scan_type='zigzagN12',
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, pretrained=pretrained, posembed=False, imgsize=256,
    )


def vmambabs_seg_tiny_s1l8_dec_256(first_dim_double=False, channel_first=False):
    return VSSMBS_Decoder(
        depths=[4, 8, 4, 4, 4], dims=48, nums_patches=[8*8, 16*16, 32*32, 64*64, 128*128, 256*256], first_dim_double=first_dim_double,
        drop_path_rate=0.25, 
        out_chans=3, ssm_d_state=1, ssm_ratio=1.0, 
        ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", scan_type='zigzagN12',
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        norm_layer=("ln2d" if channel_first else "ln"), 
        upsample_version="v3", use_checkpoint=False, 
    )

