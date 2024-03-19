# 1 32 1 128 @ 1 32 128 7297

import numpy as np

import torch

A = torch.randn(1, 32, 1, 2).float().cuda()
B = torch.randn(1, 32, 2, 23).float().cuda()

A_np = A.cpu().numpy()
B_np = B.cpu().numpy()


C = torch.matmul(A, B)


C_np = np.zeros((1, 32, 1, 23), dtype="float32")
# C_np = np.matmul(A_np, B_np)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A_slice = A_np[i, j]
        B_slice = B_np[i, j]
        k_max =  -99999
        for k in range(A_slice.shape[0]):
            for l in range(B_slice.shape[1]):
                sum = 0
                for m in range(A_slice.shape[1]):
                    sum +=( A_slice[k, m] * B_slice[m, l])
                C_np[i, j, k, l] = sum
                k_max = max(k_max, sum)

        # C_slice = A_slice @ B_slice
        # C_np[i, j] = C_slice
#         for k in range(B.shape[3]):
#             sum = 0
#             A_slice_slice = A_slice[0]
#             B_slice_slice = B_slice[:, k]
#             print(A_slice_slice.shape, B_slice_slice.shape)
#             for l in range(A.shape[2]):
#                 sum += A_slice_slice[l] * B_slice_slice[l]
#             C_np[i, j, 0, k] = sum

print(np.allclose(C.cpu().numpy(), C_np, atol=1e-3, rtol=1e-3))

# print(C)
# print(C_np)

import pdb
pdb.set_trace()