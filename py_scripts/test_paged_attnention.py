import numpy as np

# torch.Size([1, 32, 128]) torch.Size([7297, 32, 128, 1]) torch.Size([7297, 32, 128])
# at_out = scale * torch.matmul(
#     query.reshape(1, query.shape[1], -1, query.shape[-1]),
#     key.reshape(key.shape[0], key.shape[1], key.shape[2], 1).transpose(0, 3),
# )
# # 1, 32, 1, 7297
# at_out = torch.softmax(at_out, dim=-1)
# o = torch.matmul(
#     at_out,
#     value.reshape(1, value.shape[0], value.shape[1], value.shape[2]).transpose(1, 2)
# )
# print(o.shape)
# print(torch.allclose(out.view(-1), o.view(-1), atol=1e-3, rtol=1e-3))

import torch

import math
import random
np.set_printoptions(precision=3, suppress=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# num_seqs = 2
# num_heads = 32
# num_kv_heads = 32  # 暂时没用
# seq_len = 1
# head_size = 128
# context_lens = [23, 71]

num_seqs = 2
num_heads = 32
num_kv_heads = 32
seq_len = 1
head_size = 128
context_lens = [65, 13]


max_context_len = max(context_lens)
block_size = 16
FLT_MIN = np.finfo(np.float32).min
querys = torch.randn(num_seqs, num_heads, seq_len, head_size).float()
scale = float(head_size)**-0.5
x = 16 // 2  # 这里fp16 size 2  所以是个8 pack 我们虽然是使用float16 case，但是模拟用任意数据都行，而为了理解底层计算逻辑
NUM_BLOCKS = 64  # 意思下 这个是可以GPU 内存可以分配出来的最大 kv cache block大小
key_cache_shape = (NUM_BLOCKS, num_heads, head_size // x, block_size, x)
value_cache_shape = (NUM_BLOCKS, num_heads, head_size, block_size)
key_cache_np = np.zeros(key_cache_shape, dtype="float32")
value_cache_np = np.zeros(value_cache_shape, dtype="float32")
# V 1, 32, 6 + 1, 128
# QK^T = 1, 32, 1 128 @ 1, 32, 128, 7297  -> 1, 32, 1, 7297 -> softmax  1, 32, 1, 7297 -> @V  1, 32, 7297, 128  -> 1, 32, 1 128  -> transpose(1, 2)  -> 2, 6, 4096
# keys is transposed already, tranpose(1, 2)
keys = [torch.randn(num_heads, head_size, context_len).float()
        for context_len in context_lens]
values = [torch.randn(num_heads, context_len, head_size).float()
          for context_len in context_lens]


# values = [torch.tensor([i * 0.001 for i in range(num_heads * context_len * head_size)]).float().view(num_heads, context_len, head_size)
#           for context_len in context_lens]

querys_np = querys.numpy()
keys_np = [key.numpy() for key in keys]
values_np = [value.numpy() for value in values]


# 将 key value cache转换为 paged attention的样式
# KCache [num_blocks, num_kv_heads, head_size/x, block_size, x]
# VCache [num_blocks, num_kv_heads, head_size, block_size]
# paged_attention_v1 的入参为 output, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, kv_cache_dtype,
# output,  num_kv_heads, block_tables
block_tables = []
max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
for _ in range(num_seqs):
    block_table = [
        random.randint(0, NUM_BLOCKS - 1)
        for _ in range(max_num_blocks_per_seq)
    ]
    block_tables.append(block_table)  # block_tables 记录的是真实的block偏移

# 传统的key value 转换为 key_cache, value_cache
# num_heads, head_size, context_len -> key_cache_shape = (NUM_BLOCKS, num_heads, head_size // x, block_size, x)
# num_heads, context_len, head_size -> value_cache_shape = (NUM_BLOCKS, num_heads, head_size, block_size)
#


# vllm 中调用op之前block_tables, 当cache 不等的时候会胃部填0 对齐
# tensor([[7384, 7380, 7378, 7376, 7374, 7372,    0],
#         [7383, 7382, 7381, 7379, 7377, 7375, 7373]], device='cuda:0',dtype=torch.int32
for i in range(len(block_tables)):
    block_table = block_tables[i]
    context_len = context_lens[i]
    key = keys_np[i]
    value = values_np[i]
    for context_idx in range(context_len):
        block_num = block_table[context_idx // block_size]
        block_off = context_idx % block_size
        key_slice = key[:, :, context_idx].reshape(
            num_heads, head_size // x, x)
        value_slice = value[:, context_idx, :]
        # print(key_cache_np[block_num, :, block_off, :].shape, key_slice.shape)
        # print(value_cache_np[block_num, :, :, block_off].shape, value_slice.shape)
        key_cache_np[block_num, :, :, block_off, :] = key_slice
        value_cache_np[block_num, :, :, block_off] = value_slice


def ref_single_query_kv_cache(querys, keys, values):
    attn_outs = []
    for seq_idx in range(num_seqs):
        query = querys[seq_idx]
        key = keys[seq_idx]
        value = values[seq_idx]
        attn_out = scale * torch.matmul(query, key)
        attn_out = torch.softmax(attn_out, dim=-1)
        attn_out = torch.matmul(attn_out, value)
        attn_outs.append(attn_out)
    return attn_outs


def ref_plain_paged_attention_v0(querys_np, keys_np, values_np,):
    attn_outs_np = []
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]     # num_heads, 1, head_size
        key = keys_np[seq_idx]         # num_heads, head_size, context_len
        value = values_np[seq_idx]     # num_heads, context_len, head_size
        attn_out = np.zeros(
            (num_heads, seq_len, context_lens[seq_idx]), dtype="float32")          # qk^t out
        # 进行batch gemv, A少了一个维度，因为默认generate 阶段出 1 seq_len 长度
        for num_head_idx in range(num_heads):
            for context_idx in range(context_lens[seq_idx]):
                sum = 0
                for head_size_idx in range(head_size):
                    sum += query[num_head_idx, 0, head_size_idx] * \
                        key[num_head_idx, head_size_idx, context_idx]
                sum *= scale
                attn_out[num_head_idx, 0, context_idx] = sum
        # print(np.allclose(attn_out, attn_outs[seq_idx], atol=1e-3, rtol=1e-3))
        # attn_out shape ( num_head, seq_len, context_len)
        # softmax 最后一个轴，同时使用的safe softmax  实现为 reduce_max sub exp sum div
        for num_head_idx in range(num_heads):
            head_max = FLT_MIN
            # reduce max
            for context_idx in range(context_lens[seq_idx]):
                head_max = max(
                    head_max, attn_out[num_head_idx, 0, context_idx])
            # sub
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] -= head_max
            # exp
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] = math.exp(
                    attn_out[num_head_idx, 0, context_idx])
            # sum
            head_sum = 0
            for context_idx in range(context_lens[seq_idx]):
                head_sum += attn_out[num_head_idx, 0, context_idx]
            # div
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] /= head_sum
        # batch gemm
        # print(np.allclose(attn_out, attn_outs[seq_idx], atol=1e-3, rtol=1e-3))
        # (num_heads, seq_len, context_len) @ (num_heads, context_len, head_size)
        attn_out_final = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        for num_head_idx in range(num_heads):
            for head_size_idx in range(head_size):
                sum = 0
                for context_idx in range(context_lens[seq_idx]):
                    sum += attn_out[num_head_idx, 0, context_idx] * \
                        value[num_head_idx, context_idx, head_size_idx]
                attn_out_final[num_head_idx, 0, head_size_idx] = sum
        # attn_outs_np[seq_idx] = attn_out_final
        attn_outs_np.append(attn_out_final)
        # print(np.allclose(attn_out_final, attn_outs[seq_idx], atol=1e-3, rtol=1e-3))
    return attn_outs_np


def ref_plain_paged_attention_v1(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    attn_outs_np = []
    for seq_idx in range(num_seqs):
        attn_out = np.zeros(
            (num_heads, seq_len, context_lens[seq_idx]), dtype="float32")
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        for num_head_idx in range(num_heads):
            # batch gemv 计算
            for context_idx in range(context_len):
                sum = 0
                block_num = block_table[context_idx // block_size]
                block_off = context_idx % block_size
                for head_sub_idx in range(head_size // x):
                    for x_idx in range(x):
                        sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * key_cache_np[block_num, num_head_idx, head_sub_idx, block_off, x_idx]
                sum *= scale
                attn_out[num_head_idx, 0, context_idx] = sum
        # softmax 计算
        for num_head_idx in range(num_heads):
            head_max = FLT_MIN
            # reduce_max
            for context_idx in range(context_lens[seq_idx]):
                head_max = max(
                    head_max, attn_out[num_head_idx, 0, context_idx])
            # sub
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] -= head_max
            # exp
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] = math.exp(
                    attn_out[num_head_idx, 0, context_idx])
            # sum
            head_sum = 0
            for context_idx in range(context_lens[seq_idx]):
                head_sum += attn_out[num_head_idx, 0, context_idx]
            # div
            for context_idx in range(context_lens[seq_idx]):
                attn_out[num_head_idx, 0, context_idx] /= head_sum

        # attn_outs_np.append(attn_out)
        attn_out_final = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        for num_head_idx in range(num_heads):
            for head_size_idx in range(head_size):
                sum = 0
                for context_idx in range(context_lens[seq_idx]):
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    sum += attn_out[num_head_idx, 0, context_idx] * \
                        value_cache_np[block_num, num_head_idx,
                                       head_size_idx, block_off]
                attn_out_final[num_head_idx, 0, head_size_idx] = sum
        attn_outs_np.append(attn_out_final)
    return attn_outs_np


# V2 主要是 将循环合并掉, 循环在计算过程中会一直占用寄存器 共享内存等资源，同时合并循环也能增加算数强度(在寄存器/共享内存上)
def ref_plain_paged_attention_v2(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    # 参考 https://zhuanlan.zhihu.com/p/673284781 vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现 增加部分cuda kernel 相关注解
    # paged attention 的实质 是block 去算 gemv softmax gemv 这样的方式
    # NUM_THREADS 默认为128, 这个也和head_size 相关, head_size 常为128，则1个线程实际是算一个数，且128的block 在cuda 中也是相当常见
    # grid大小(num_heads, num_seqs) 不同grid之间显然没有依赖，是非常合理的切分
    # 所以 由于attention 的机制原因，也是可以理解为 grid(num_heads * num_seqs) block(head_dim)
    # 这里考虑一下 是否可以开更大的block 从而让 block 做更多的事比如 256, 则grid 可以减半 增加一些， num_head 通常为32等之类的偶数，所以存在可能性，甚至到512
    # 不过由于 cuda 多sm 开小block 更具有性价比 256 的实现也许可以试试
    # x = 8 为 sizeof(half) * x = 16 bytes = 128bit 这个可以使用LDG.E.128 一次加载上来8个数
    # 文中说 128 的线程块 分为 4 warps, 而每个warp 也分成了blk_size 的thread_group
    attn_outs_np = []
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        attn_out_final = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        for num_head_idx in range(num_heads):
            # 调整了attn_out 的位置, 如果这部分占用资源, 相比原先为原来的 1 / num_heads
            # 共享内存占用 max_context_len * sizeof(float16)
            attn_out = np.zeros(
                (seq_len, context_lens[seq_idx]), dtype="float32")
            # batch gemv 计算
            head_max = FLT_MIN
            for context_idx in range(context_len):
                sum = 0
                block_num = block_table[context_idx // block_size]
                block_off = context_idx % block_size
                for head_sub_idx in range(head_size // x):
                    for x_idx in range(x):
                        sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * \
                            key_cache_np[block_num, num_head_idx,
                                         head_sub_idx, block_off, x_idx]
                sum *= scale
                head_max = max(head_max, sum)
                attn_out[0, context_idx] = sum
            # 在计算softmax 这里由于使用的是safe softmax的算法，所以两次reduce block 内必须同步，所以说性能上会慢一点，
            # sub exp sum 这里依赖 head_max 必须要先算完 才能算整体的, flashattention的改进 可以算online softmax 这里就不用等了，一个非常不错的优化点
            head_sum = 0
            for context_idx in range(context_lens[seq_idx]):
                exp_val = math.exp(attn_out[0, context_idx] - head_max)
                attn_out[0, context_idx] = exp_val
                head_sum += exp_val

            for head_size_idx in range(head_size):
                sum = 0
                for context_idx in range(context_lens[seq_idx]):
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    sum += attn_out[0, context_idx] * value_cache_np[block_num,
                                                                     num_head_idx, head_size_idx, block_off] / head_sum
                attn_out_final[num_head_idx, 0, head_size_idx] = sum
        attn_outs_np.append(attn_out_final)
    return attn_outs_np


def ref_plain_paged_attention_v3(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    # 2-pass
    # https://zhuanlan.zhihu.com/p/663932651
    # where N is sequence length  d is head_size  inference N = 1
    # torch.Size([1, 32, 1, 128]) torch.Size([1, 32, 128, 23]) torch.Size([1, 32, 23, 128])
    #                   torch.Size([1, 32, 1, 23])         torch.Size([1, 32, 23, 128])
    #                                       torch.Size([1, 32, 1, 128])
    # flash attention 的实现 online softmax 减少访存 内存访问空间复杂度 s 𝑂(𝑁^2𝑑^2𝑀^−1)
    # 1. 计算softmax 的reduction的时候没有必要访问所有的输入
    # 2. 反向的时候不存储中间结果矩阵
    # 常规算法中safe softmax 的reduce轴 是不可分的，online softmax则是将reduce 分成多个块，
    # paged attention的主要问题是：当 num_heads * num_seqs 小于sm的个数的时候，整体的并行度是不够的
    # 所以文中提出将输入分成多个block 以及 通过tiling 提高softmax 的性能 tiling, recomputation
    # softmax 原本的计算是要最后一维度所有的值，现在将最后一维进行分块计算，
    attn_outs_np = []
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        attn_out_final = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        for num_head_idx in range(num_heads):
            # 调整了attn_out 的位置, 如果这部分占用资源, 相比原先为原来的 1 / num_heads
            # 共享内存占用 max_context_len * sizeof(float16)
            attn_out = np.zeros(
                (seq_len, context_lens[seq_idx]), dtype="float32")
            # batch gemv 计算
            head_max = FLT_MIN
            head_sum = 0
            for context_idx in range(context_len):
                pre_max = head_max
                sum = 0
                block_num = block_table[context_idx // block_size]
                block_off = context_idx % block_size
                # online softmax
                for head_sub_idx in range(head_size // x):
                    for x_idx in range(x):
                        sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * \
                            key_cache_np[block_num, num_head_idx,
                                         head_sub_idx, block_off, x_idx]
                sum *= scale
                head_max = max(head_max, sum)
                attn_out[0, context_idx] = sum
                head_sum = head_sum * \
                    math.exp((pre_max - head_max)) + math.exp(sum - head_max)
            # head_sums = np.zeros((head_size), dtype="float32")
            for head_size_idx in range(head_size):
                sum = 0
                for context_idx in range(context_len):
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    # recompute
                    sum += math.exp(attn_out[0, context_idx] - head_max) * \
                        value_cache_np[block_num, num_head_idx,
                                       head_size_idx, block_off] / head_sum
                attn_out_final[num_head_idx, 0, head_size_idx] += sum
        attn_outs_np.append(attn_out_final)
    return attn_outs_np


def ref_plain_paged_attention_v4(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    # one-pass paged attentio1n
    # online softmax 引入简化paged attention到如下形式
    # https://zhuanlan.zhihu.com/p/663932651
    attn_outs_np = []
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        attn_out = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        for num_head_idx in range(num_heads):
            head_max = FLT_MIN
            head_sum = 0
            sum_values = np.zeros((head_size,), dtype="float32")
            # outer loop
            for context_idx in range(context_len):  # i/N
                # m_{i-1}
                pre_max = head_max
                # d_{i-1}
                pre_sum = head_sum
                qk_sum = 0
                block_num = block_table[context_idx // block_size]
                block_off = context_idx % block_size
                # inter loop 这里query 维度seq_len 为1，对于prefill阶段 这里就是方阵
                # Q (1, 32, N, 128)  K^T (1, 32, 128, N) V (1, 32, N, 128)
                #       QK^T (1, 32, N, N)   V (1, 32, N, 128)
                #                (1, 32, N, 128)
                for head_sub_idx in range(head_size // x):
                    for x_idx in range(x):
                        qk_sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * \
                            key_cache_np[block_num, num_head_idx,
                                         head_sub_idx, block_off, x_idx]
                # x_i
                qk_sum *= scale
                # print("qk_sum4", qk_sum)
                # m_i
                head_max = max(head_max, qk_sum)
                head_sum = pre_sum * \
                    math.exp((pre_max - head_max)) + \
                    math.exp(qk_sum - head_max)  # d_i
                for head_size_idx in range(head_size):
                    sum_values[head_size_idx] = sum_values[head_size_idx] * \
                        pre_sum * math.exp(pre_max - head_max) / head_sum
                # print("sum_values4", sum_values)
                # print("qk_sums4", qk_sums)
                # print(f"before {context_idx} sum_values4", sum_values, "head_sum", head_sum, "qk_sum", qk_sum, "head_max", head_max)
                for head_size_idx in range(head_size):
                    sum_values[head_size_idx] += math.exp(
                        qk_sum - head_max) * value_cache_np[block_num, num_head_idx, head_size_idx, block_off] / head_sum
                # print(f"after {context_idx} sum_values4", sum_values, "head_sum", head_sum, "qk_sum", qk_sum, "head_max", head_max)
                # print("sum_values4",sum_values)
            attn_out[num_head_idx, 0, :] = sum_values
        attn_outs_np.append(attn_out)
    return attn_outs_np


def ref_plain_paged_attention_v5(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    # https://zhuanlan.zhihu.com/p/663932651
    # vllm 中1个warp 算1个block(16 context len)  1 warp 1 blcok   一个block 128 thread 为4warp 所以subcontext_size = 64
    # flash attention 进行了tiling 如果tiling 所以subcontext_size = 64
    # flash attention 最终实现 增加了tiling
    # flash attention1 implementation
    attn_outs_np = []
    subcontext_size = 64
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        attn_out = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        subcontext_num = (context_len + subcontext_size - 1) // subcontext_size
        for num_head_idx in range(num_heads):
            head_max = FLT_MIN      # 记录context_len内当前迭代过程中的max值，也可以理解为 m_{i-1}
            # head sum 也可以理解为 subcontext d_{i-1} 记录的是上次迭代的时候sum出来的值
            head_sum = 0
            sum_values = np.zeros((head_size,), dtype="float32")
            for subcontext_num_idx in range(subcontext_num):    # 1, 2
                qk_sums = np.full((subcontext_size, ), FLT_MIN, dtype="float32")

                for subcontext_idx in range(subcontext_size):   # 1-64
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    qk_sum = 0
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    for head_sub_idx in range(head_size // x):
                        for x_idx in range(x):
                            qk_sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * \
                                key_cache_np[block_num, num_head_idx,
                                             head_sub_idx, block_off, x_idx]

                    qk_sum *= scale
                    # x_i = Q[k, :]K^T[:,ib: ib + b]
                    qk_sums[subcontext_idx] = qk_sum  # x_i i from 0 to 63
                subcontext_pre_max = head_max
                subcontext_pre_sum = head_sum
                subcontext_max = FLT_MIN
                # m_i^{local} = max(x_i) i from 0 to 63$
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    subcontext_max = max(subcontext_max, qk_sums[subcontext_idx])
                # m_i = max(m_{i-1}, m_i^{local})
                head_max = max(head_max, subcontext_max)
                # d_i = d_{i-1} e^{m_{i-1}-m_i}
                subcontext_sum = subcontext_pre_sum * \
                    math.exp(subcontext_pre_max - head_max)
                # d_i += \sum_{j=0}^{b} {e^{x_i[j] - m_i}}
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    subcontext_sum += math.exp(
                        qk_sums[subcontext_idx] - head_max)
                head_sum = subcontext_sum  # d_i
                # sum_values[head_size_idx] 迭代过程中上次结果输出 首先第一步需要将上次的结果修成为当前max值的结果
                # o_i = o_{i-1} \frac {d_{i-1}e^{m_{i-1}-m_i}} {d_i}
                # 如果m_{i-1} - m_i = 0 则本轮没有改最大值，前面的结果也是按照head_max累加的 可以直接加
                # print(qk_sums)
                # 1 2  1 2 128
                for head_size_idx in range(head_size):
                    sum_values[head_size_idx] = sum_values[head_size_idx] * \
                        subcontext_pre_sum * \
                        math.exp(subcontext_pre_max - head_max) / head_sum
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    # print(f"before {subcontext_idx} sum_values5", sum_values, "head_sum", head_sum, "qk_sum", qk_sums[subcontext_idx], "head_max", head_max)
                    # print(subcontext_idx, qk_sums[subcontext_idx], math.exp(qk_sums[subcontext_idx] - head_max))
                    for head_size_idx in range(head_size):
                        sum_values[head_size_idx] += math.exp(qk_sums[subcontext_idx] - head_max) * \
                            value_cache_np[block_num, num_head_idx,
                                           head_size_idx, block_off] / head_sum
                    # print(f"after {subcontext_idx} sum_values5", sum_values, "head_sum", head_sum, "qk_sum", qk_sums[subcontext_idx], "head_max", head_max)
                    # print("sum_values5",sum_values)
            attn_out[num_head_idx, 0, :] = sum_values
        attn_outs_np.append(attn_out)
    return attn_outs_np


def ref_plain_paged_attention_v6(querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len):
    # flash attention v2 相比attention v1 加速了2x 主要改进：
    # 1. 减少非乘法计算
    # 2. 优化qkv for 循环顺序
    # 3. 采用shared memory 减少通信
    attn_outs_np = []
    subcontext_size = 64
    for seq_idx in range(num_seqs):
        query = querys_np[seq_idx]
        block_table = block_tables[seq_idx]
        context_len = context_lens[seq_idx]
        attn_out = np.zeros(
            (num_heads, seq_len, head_size), dtype="float32")
        subcontext_num = (context_len + subcontext_size - 1) // subcontext_size
        for num_head_idx in range(num_heads):
            head_max = FLT_MIN      # 记录context_len内当前迭代过程中的max值，也可以理解为 m_{i-1}
            # head sum 也可以理解为 subcontext d_{i-1} 记录的是上次迭代的时候sum出来的值
            head_sum = 0
            sum_values = np.zeros((head_size,), dtype="float32")
            for subcontext_num_idx in range(subcontext_num):    # 1, 2
                qk_sums = np.full((subcontext_size, ), FLT_MIN, dtype="float32")

                for subcontext_idx in range(subcontext_size):   # 1-64
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    qk_sum = 0
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    for head_sub_idx in range(head_size // x):
                        for x_idx in range(x):
                            qk_sum += query[num_head_idx, 0, head_sub_idx * x + x_idx] * \
                                key_cache_np[block_num, num_head_idx,
                                             head_sub_idx, block_off, x_idx]

                    qk_sum *= scale
                    # x_i = Q[k, :]K^T[:,ib: ib + b]
                    qk_sums[subcontext_idx] = qk_sum  # x_i i from 0 to 63
                subcontext_pre_max = head_max
                subcontext_pre_sum = head_sum
                subcontext_max = FLT_MIN
                # m_i^{local} = max(x_i) i from 0 to 63$
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    subcontext_max = max(subcontext_max, qk_sums[subcontext_idx])
                # m_i = max(m_{i-1}, m_i^{local})
                head_max = max(head_max, subcontext_max)
                # d_i = d_{i-1} e^{m_{i-1}-m_i}
                subcontext_sum = subcontext_pre_sum * \
                    math.exp(subcontext_pre_max - head_max)
                # d_i += \sum_{j=0}^{b} {e^{x_i[j] - m_i}}
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    subcontext_sum += math.exp(
                        qk_sums[subcontext_idx] - head_max)
                head_sum = subcontext_sum  # d_i
                # sum_values[head_size_idx] 迭代过程中上次结果输出 首先第一步需要将上次的结果修成为当前max值的结果
                # o_i = o_{i-1} \frac {d_{i-1}e^{m_{i-1}-m_i}} {d_i}
                # 如果m_{i-1} - m_i = 0 则本轮没有改最大值，前面的结果也是按照head_max累加的 可以直接加
                # print(qk_sums)
                # 1 2  1 2 128
                for head_size_idx in range(head_size):
                    sum_values[head_size_idx] = sum_values[head_size_idx] * \
                        subcontext_pre_sum * \
                        math.exp(subcontext_pre_max - head_max) / head_sum
                for subcontext_idx in range(subcontext_size):
                    context_idx = subcontext_num_idx * subcontext_size + subcontext_idx
                    if context_idx >= context_len:
                        break
                    block_num = block_table[context_idx // block_size]
                    block_off = context_idx % block_size
                    # print(f"before {subcontext_idx} sum_values5", sum_values, "head_sum", head_sum, "qk_sum", qk_sums[subcontext_idx], "head_max", head_max)
                    # print(subcontext_idx, qk_sums[subcontext_idx], math.exp(qk_sums[subcontext_idx] - head_max))
                    for head_size_idx in range(head_size):
                        sum_values[head_size_idx] += math.exp(qk_sums[subcontext_idx] - head_max) * \
                            value_cache_np[block_num, num_head_idx,
                                           head_size_idx, block_off] / head_sum
                    # print(f"after {subcontext_idx} sum_values5", sum_values, "head_sum", head_sum, "qk_sum", qk_sums[subcontext_idx], "head_max", head_max)
                    # print("sum_values5",sum_values)
            attn_out[num_head_idx, 0, :] = sum_values
        attn_outs_np.append(attn_out)
    return attn_outs_np

# print(querys)
print(querys.shape, keys[0].shape, values[0].shape)
print(querys_np.shape, key_cache_np.shape, value_cache_np.shape, )
# attn_outs = ref_single_query_kv_cache(querys, keys, values)
# attn_outs_np_v0 = ref_plain_paged_attention_v0(querys_np, keys_np, values_np)
# attn_outs_np_v1 = ref_plain_paged_attention_v1(
#     querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)
# attn_outs_np_v2 = ref_plain_paged_attention_v2(
#     querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)
# attn_outs_np_v3 = ref_plain_paged_attention_v3(
#     querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)
attn_outs_np_v4 = ref_plain_paged_attention_v4(
    querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)
# attn_outs_np_v5 = ref_plain_paged_attention_v5(
#     querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)
attn_outs_np_v6 = ref_plain_paged_attention_v6(
    querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len)

for num_seq in range(num_seqs):
    print(np.allclose(attn_outs_np_v4[num_seq],
          attn_outs_np_v6[num_seq], atol=1e-3, rtol=1e-3))

# print(attn_outs_np_v4[num_seq], attn_outs_np_v5[num_seq])

# 送入到ops的 kv cache 是刷的0 且是对齐的,
# x = 16 // torch.tensor([], dtype=torch_dtype).element_size()

# key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
#                                                 num_kv_heads, head_size,
#                                                 kv_cache_dtype, dtype, seed,
#                                                 device)
