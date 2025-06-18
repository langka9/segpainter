# import numpy as np

# def zigzag_path(N):
#     def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
#         path = []
#         for i in range(N):
#             for j in range(N):
#                 # If the row number is even, move right; otherwise, move left
#                 col = j if i % 2 == 0 else N - 1 - j
#                 path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
#         return path

#     def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
#         path = []
#         for j in range(N):
#             for i in range(N):
#                 # If the column number is even, move down; otherwise, move up
#                 row = i if j % 2 == 0 else N - 1 - i
#                 path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
#         return path

#     paths = []
#     for start_row, start_col, dir_row, dir_col in [
#         (0, 0, 1, 1),
#         (0, N - 1, 1, -1),
#         (N - 1, 0, -1, 1),
#         (N - 1, N - 1, -1, -1),
#     ]:
#         paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
#         paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

#     for _index, _p in enumerate(paths):
#         paths[_index] = np.array(_p)
#     return paths

# def zigzag4_path(N):
#     def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
#         path = []
#         for i in range(N):
#             for j in range(N):
#                 # If the row number is even, move right; otherwise, move left
#                 col = j if i % 2 == 0 else N - 1 - j
#                 path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
#         return path

#     def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
#         path = []
#         for j in range(N):
#             for i in range(N):
#                 # If the column number is even, move down; otherwise, move up
#                 row = i if j % 2 == 0 else N - 1 - i
#                 path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
#         return path

#     half_N = N // 2
#     paths = []
#     for start_row, start_col, dir_row, dir_col in [
#         (0, 0, 1, 1),
#         (0, half_N - 1, 1, -1),
#         (half_N - 1, 0, -1, 1),
#         (half_N - 1, half_N - 1, -1, -1),
#     ]:
#         paths.append(zigzag_path_lr(half_N, start_row, start_col, dir_row, dir_col))
#         paths.append(zigzag_path_tb(half_N, start_row, start_col, dir_row, dir_col))

#     # paths 总共有8条路径
#     offset_dict = {
#         '0': 0,
#         '1': half_N,
#         '2': 2 * half_N * half_N,
#         '3': (2 * half_N + 1) * half_N,
#     }

#     codebooks = {}  # 记录4个字典，每个字典保存各个象限扫描路径的对应关系
#     for group, offset in offset_dict.items():
#         code_dict = {}
#         for row in range(half_N):
#             for col in range(half_N):
#                 key = row * half_N + col
#                 code_dict[f'{key}'] = key + offset + row * half_N
#         codebooks[group] = code_dict

#     group_path_dicts = {}
#     for group, code_dict in codebooks.items():
#         group_path_dicts[group] = []
#         for _index, path in enumerate(paths):
#             # list(map(lambda x:x+5, ilist))
#             group_path = list(map(lambda x:code_dict[f'{x}'], path))
#             group_path_dicts[group].append(np.array(group_path))

#     return group_path_dicts

# def zigzag_3p_path(N):
#     half_N = N // 2
#     path_4q = []
#     group_path_dicts = zigzag4_path(N)
#     group_index_pairs = [
#         [('0', 6), ('1', 4), ('2', 2), ('3', 0), ],
#         [('0', 7), ('1', 5), ('2', 3), ('3', 1), ],
#         [('0', 0), ('1', 2), ('2', 4), ('3', 6), ],
#         [('0', 1), ('1', 3), ('2', 5), ('3', 7), ],
#     ]
#     for group_index_pair in group_index_pairs:
#         group_paths = []
#         for group, index in group_index_pair:
#             group_path = group_path_dicts[group][index].reshape(half_N, half_N)
#             group_paths.append(group_path)
#         path_4q.append(np.stack([np.stack(group_paths[:2], axis=1), np.stack(group_paths[2:], axis=1)], axis=0).reshape(N*N))
#     path_3p = path_4q + zigzag_path(N)
#     return path_3p

# print(zigzag_3p_path(4))

import torch
import numpy as np

# depths=[2, 2, 9, 2]
# drop_path_rate = 0.1

# dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
# print(dpr)

x_proj = [torch.ones((2, 2))]

b = torch.stack([t for t in x_proj], dim=0)
print(b.size())
