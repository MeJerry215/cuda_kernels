#include "common.h"

/*
PagedAttention partitions the KV cache of each sequence into KV blocks. Each block contains the key and value
vectors for a fixed number of tokens

self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

hidden  2, 6, 4096 -> 4096, 4096 qproj 2, 6, 4096 -> view bsz,q_len,num_heads,head_dim 2, 6, 32, 128 -> transpose(1, 2) 2, 32, 6, 128
				   -> 4096, 4096 kproj 2, 6, 4096      bsz,q_len,kv_num_heads,head_dim 2, 6, 32, 128 -> transpsoe(1, 2) 2, 32, 6, 128 -> transpose(2, 3) = 2, 32, 128, 6 + 1
				   -> 4096, 4096 vproj 2, 6, 4096      bsz,q_len,kv_num_heads,head_dim 2, 6, 32, 128 -> transpose(1, 2) 2, 32, 6, 128 = 2, 32, 6 + 1, 128

	QK^T = 2, 32, 6, 128 @ 2, 32, 128, 6 -> 2, 32, 6, 7 -> softmax  2, 32, 6, 7 -> @V  2, 32, 6, 128 -> transpose(1, 2) 2, 6, 32, 128, -> 2, 6, 4096


	QK^T = 1, 32, 1 128 @ 1, 32, 128, 7297  -> 1, 32, 1, 7297 -> softmax  1, 32, 1, 7297 -> @V  1, 32, 7297, 128  -> 1, 32, 1 128  -> transpose(1, 2)  -> 2, 6, 4096

torch.Size([1, 32, 128]) torch.Size([7297, 32, 128]) torch.Size([7297, 32, 128])

 seq_lem num_kv_heads, head_size,

https://zhuanlan.zhihu.com/p/673284781
batch_size, heads, seq_len, head_dim  

Since Paged Attention Introduces memory access pattern not effectiently supported by existing system, so
1. Fused reshape and block write, 
2. Fusing block read and attention
3. Fused block copy


block_tables 中存放这当前num_seqs 大小的的blocksize, 一个block 中是 16 连续的

paged attention 分为 V1 和V2 版本 V1 改变自FasterTransformers的MHA实现 V2 参考的flashdecoding方式实现
V1 适合小于8192或者 num_seqs * num_heads>512 的情况
https://github.com/vllm-project/vllm/issues/421  ncu 优化实现

prefill 阶段使用的是xformers的 FMHA

https://zhuanlan.zhihu.com/p/673284781 实现分析
PA使用tensor的维度信息：

out [num_seqs, num_heads, head_size]
Q [num_seqs, num_heads, head_size]
KCache [num_blocks, num_kv_heads, head_size/x, block_size, x]：x表示一个向量化的大小，如float16 -> 16 / sizeof(float16) = 8。
kv cache 7385, 32, 16, 16, 8
VCache [num_blocks, num_kv_heads, head_size, block_size]
[7385, 32, 128, 16]


num_kv_heads, head_size
num_kv_heads, head_size


Paged内存管理相关的辅助数据结构：

blk_size：也就是block_size，是KVCache page的最高维，KVCache是若干个page的集合，每个page存(blk_size, num_head，head_size)个K、V的元素。
head_mapping [num_heads] 用于MQA, GQA，确定用的KV_head
block_tables [num_seqs, max_num_blocks_per_seq] block_tables映射表，表示每个sequence映射到哪几个block上
context_lens [num_seqs] 用于变长


为什么K cache layout 和V cache layout 不同？



grid大小(num_heads, num_seqs)  线程块大小 默认128  warp内32个线程进一步划分为blk_size个thread group


Attention计算softmax(QK^T)V
*/


/*
https://zhuanlan.zhihu.com/p/673284781
paged attention 使用block_size 为128 当kv cache的bock size 为16 时， threadgroup size 为(warp_size / block_size)
所以 4个warp 8个thread group 而1个thread group算1 个token
*/

// querys_np, key_cache_np, value_cache_np, block_tables, context_lens, max_context_len
// querys_np

/*
/satadata/home/lei.zhang/src_code/llm_benchmark/submodules/vllm/csrc/attention/attention_kernels.cu
torch.Size([2, 32, 128]) torch.Size([7385, 32, 16, 16, 8]) torch.Size([7385, 32, 128, 16]) tensor([[7384, 7380, 7378, 7376,    0,    0],
        [7383, 7382, 7381, 7379, 7377, 7375]], device='cuda:0',
       dtype=torch.int32) 16 tensor([59, 89], device='cuda:0', dtype=torch.int32)
*/
template<typename T, int HEAD_SIZE=32, int BLOCK_SIZE = 16, int NUM_THREADS = 128>
__global__ void paged_attention_v0(T* querys, T* key_cache, T* value_cache, int32_t* block_tables, int32_t* context_lens,
    int32_t max_context_len, int32_t num_kv_heads, const float scale, int q_stride, int kv_block_stride, int kv_head_stride)
{
    /*
    threadIdx.x  128
    blockIdx.x   num_head_idx
    blockIdx.y   seq_idx
    */
    const int seq_idx = blockIdx.y;
    const int context_len = context_lens[seq_idx];
    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    int32_t THREAD_GROUP_SIZE =  
}



int main(int argc, char const* argv[])
{
    return 0;
}
