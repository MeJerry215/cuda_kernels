#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <common.h>
// using namespace cub;

// using namespace cub;

template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ half Inf<half>()
{
    return 65504;
}

template<>
__inline__ __device__ float Inf<float>()
{
    return CUDART_INF_F;
}

template<>
__inline__ __device__ double Inf<double>()
{
    return CUDART_INF;
}


template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};


template<>
struct MaxOp<half> {
    __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hmax(a, b); }
};

template<template<typename> typename ReductionOp, typename T, int width = 32>
__inline__ __device__ T WarpReduce(T val)
{
    for (int mask = width / 2; mask > 0; mask /= 2)
        // mask = 16, 8, 4, 2, 1
        // __shfl_down_sync 是不是也能实现这样的功能, 或者说性能能够更快？
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, width));

    return val;
}

// cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::BLOCK_REDUCE_RAKING
template<int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void BlockReduceKernel(half* d_in, half* d_out, int rows, int cols)
{
    // Specialize BlockReduce type for our thread block
    int row_id = blockIdx.x;
    half* inptr = d_in + row_id * cols;
    typedef cub::BlockReduce<half, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    // Shared memory
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    // Per-thread tile data
    half data[ITEMS_PER_THREAD];
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, inptr, data);
    // Start cycle timer
    // clock_t start = clock();
    // Compute sum
    half aggregate = BlockReduceT(temp_storage).Reduce(data, cub::Max());
    // Stop cycle timer
    // clock_t stop = clock();

    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
        d_out[row_id] = aggregate;
}


template<int pack_size, typename packType, int thread_group>
__global__ void reduce_last_padding_impl(half* in, half* out, int rows, int cols)
{
    // constexpr int num_packs = cols_per_thread / pack_size;
    half buf[pack_size] = {-Inf<half>()};
    int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group;
    const int thread_group_stride = pack_size * thread_group;

    for (int64_t row = global_thread_group_id; row < rows; row += step) {
        half max_val  = -Inf<half>();

        for(int col = lane_id * pack_size;  col < cols; col += thread_group_stride) {
            // 当前col 继续独一个pack 不会越界
            packType val;
            half* pval = (half*)&val;
            val = *((packType*)(in + row * cols + col));

            for(int pack_i = 0; pack_i < pack_size; pack_i ++)
                buf[pack_i] = __hmax(buf[pack_i], pval[pack_i]);
        }

        for(int pack_i = 0; pack_i < pack_size; pack_i ++) {
            buf[pack_i] = WarpReduce<MaxOp, half, thread_group>(buf[pack_i]);
            max_val = __hmax(buf[pack_i], max_val);
        }

        // write to share memory
        if (lane_id == 0)
            out[row] = max_val;
    }
}


template<int pack_size, typename packType, int thread_group>
void launch_reduce_last_padding(half* in, half* out, int rows, int cols)
{
    constexpr int block_size = 128;
    constexpr int waves = 32;
    constexpr int thread_groups_per_block = block_size / thread_group;
    assert(block_size % thread_group == 0);
    dim3 block(thread_group, block_size / thread_group);
    int num_blocks = (rows + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    assert(grid_dim_x > 0);
    reduce_last_padding_impl<pack_size, half, thread_group> <<< grid_dim_x, block>>>(in, out, rows, cols);
}

// template<int pack_size, int thread_group, typename packType>
// void reduce_last_padding(half* in, half* out, int rows, int cols) {
//     if (cols <= 1 * pack_size * pack_num) {
//         launch_reduce_last_padding<pack_size, half, 1>(in, out, rows, cols);
//     } else if (cols <= 2 * pack_size * pack_num) {
//         launch_reduce_last_padding<pack_size,  half, 2>(in, out, rows, cols);
//     } else if (cols <= 4* pack_size * pack_num) {
//         launch_reduce_last_padding< pack_size, pack_num, half, 4>(in, out, rows, cols);
//     } else if (cols <= 8 * pack_size * pack_num) {
//         launch_reduce_last_padding< pack_size, pack_num, half, 8>(in, out, rows, cols);
//     } else if (cols <= 16 * pack_size * pack_num) {
//         launch_reduce_last_padding< pack_size, pack_num, half, 16>(in, out, rows, cols);
//     } else if (cols <= 32 * pack_size * pack_num) {
//         launch_reduce_last_padding< pack_size, pack_num, half, 32>(in, out, rows, cols);
//     } else {
//         launch_reduce_last_padding< pack_size, pack_num, half, 32>(in, out, rows, cols);
//     }
// }


template<int thread_group>
void reduce_last_wrapper(half* in, half* out, int rows, int cols)
{
    if (cols <= 2048) {
        if (cols % 8 == 0)
            launch_reduce_last_padding<8, float4, thread_group>(in, out, rows, cols);
        else if (cols % 4 == 0)
            launch_reduce_last_padding<4, float2, thread_group>(in, out, rows, cols);
        else if (cols % 2 == 0 )
            launch_reduce_last_padding<2, float, thread_group>(in, out, rows, cols);
        else
            launch_reduce_last_padding<1, half, thread_group>(in, out, rows, cols);
    } else
        assert(false);
}

int main(int argc, char const* argv[])
{
    constexpr int MAX_ROWS = 32 * 12 * 128, MAX_COLS = 4096;
    REGISTER_BUFF(A, bytes_of<half>(MAX_ROWS * MAX_COLS));
    REGISTER_BUFF(B, bytes_of<half>(MAX_ROWS * MAX_COLS));
    genOrLoad<float>(string("test_float16.bin"), h_A, (size_t)0);
    CHECK_CALL_ERROR(cudaMemcpy(d_A, h_A, bytes_of<half>(MAX_ROWS * MAX_COLS), cudaMemcpyHostToDevice));
    // BlockReduceKernel<16, 1, cub::BLOCK_REDUCE_RAKING><<<32 * 128, 32>>>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // BLOCK_REDUCE_WARP_REDUCTIONS
    // cub::CachingDeviceAllocator  g_allocator(true);
    // cub::DeviceReduce::Max()
    BlockReduceKernel<32, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 32>>>((half*)d_A, (half*)d_B, 32 * 128, 32);
    BlockReduceKernel<32, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 32>>>((half*)d_A, (half*)d_B, 32 * 128, 128);
    BlockReduceKernel<64, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 64>>>((half*)d_A, (half*)d_B, 32 * 128, 128);
    BlockReduceKernel<128, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 128>>>((half*)d_A, (half*)d_B, 32 * 128, 128);
    BlockReduceKernel<64, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 64>>>((half*)d_A, (half*)d_B, 32 * 128, 512);
    BlockReduceKernel<128, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 128>>>((half*)d_A, (half*)d_B, 32 * 128, 512);
    BlockReduceKernel<256, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 256>>>((half*)d_A, (half*)d_B, 32 * 128, 512);
    BlockReduceKernel<512, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 512>>>((half*)d_A, (half*)d_B, 32 * 128, 512);
    BlockReduceKernel<64, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 64>>>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    BlockReduceKernel<128, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 128>>>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    BlockReduceKernel<256, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 256>>>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    BlockReduceKernel<512, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 512>>>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    BlockReduceKernel<1024, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 1024>>>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    BlockReduceKernel<128, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 128>>>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    BlockReduceKernel<256, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 256>>>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    BlockReduceKernel<512, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 512>>>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    BlockReduceKernel<1024, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS> <<< 32 * 128, 1024>>>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // BlockReduceKernel<16,1, cub::BLOCK_REDUCE_WARP_REDUCTIONS><<<32 * 128, 32>>>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 32);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 64);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 128);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 256);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 512);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 1024);
    // reduce_last_wrapper<1>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // reduce_last_wrapper<2>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // reduce_last_wrapper<4>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // reduce_last_wrapper<8>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // reduce_last_wrapper<16>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    // reduce_last_wrapper<32>((half*)d_A, (half*)d_B, 32 * 128, 2048);
    CHECK_CALL_ERROR(cudaMemcpy(h_B, d_B, bytes_of<half>(32 * 128), cudaMemcpyDeviceToHost));
    dump_to_file("a512_src.bin", h_B, bytes_of<half>(32 * 128));
    CHECK_CUDA_ERROR();
    return 0;
}
