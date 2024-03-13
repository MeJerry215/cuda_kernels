#include "common.h"


template<typename T>
struct DefaultComputeType {
    using type = T;
};

template<>
struct DefaultComputeType<half> {
    using type = float;
};


template<typename T>
__inline__ __device__ T Inf();

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

template<typename T, int N>
struct GetPackType {
    using type = typename std::aligned_storage<N* sizeof(T), N* sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
    static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
    __device__ Pack() {
        // do nothing
    }
    PackType<T, N> storage;
    T elem[N];
};


template<typename SRC, typename DST>
struct DirectLoad {
    DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
    template<int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const
    {
        Pack<SRC, N> pack;
        const int64_t offset = (row * row_size + col) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll

        // 但是这里我没有看到vectorsize
        for (int i = 0; i < N; ++i)  dst[i] = static_cast<DST>(pack.elem[i]);
    }
    const SRC* src;
    int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
    DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
    template<int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col)
    {
        Pack<DST, N> pack;
        const int64_t offset = (row * row_size + col) / N;
#pragma unroll

        for (int i = 0; i < N; ++i)  pack.elem[i] = static_cast<DST>(src[i]);

        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
    }
    DST* dst;
    int64_t row_size;
};


/*
safe softmax 实现
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

根据softmax的定义可知，对于任意的输入shape x 在kernel表现中都只存在2个维度，规约轴 和非规约轴  [2, 2, 3, 4] 等价于 [12, 4] 的softmax
假设输入 shape 为 (256, 1024)

计算顺序 也是 (Q, B) reduce_max (Q, 1) -> (Q, B) (Q, 1) broadcast sub (Q, B) -> (Q, B) exp (Q, B)
                            -> (Q, B) reduce sum (Q, 1) -> (Q, B) (Q, 1) broadcast div (Q, B)
    reduce_max          read Q * B              store Q
    broadcast_sub       read Q * B + Q          store Q * B
    exp                 read Q * B              store Q * B
    reduce_sum          read Q * B              store Q
    broadcast_div       read Q * B + Q          store Q * B

总计需要 8 * Q * B + 4 * Q  所以根据 计算 native 的softmax 实现最多只能实现 1 / 8 设备带宽

 设定测试shape 为 32 * 12 * 128, B, B维任意变动
 B in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

 1. 一个 Warp 处理一行的计算，适用于 num_cols <= 1024 情况
 2. 一个 Block 处理一行的计算，借助 Shared Memory 保存中间结果数据，适用于需要的 Shared Memory 资源满足 Kernel Launch 的可启动条件的情况，在本测试环境中是 1024 < num_cols <= 4096
 3. 一个 Block 处理一行的计算，不使用 Shared Memory，重复读输入 x，适用于不支持(1)、（2)的情况 > 4096


*/


template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};


/*
WarpReduce
__shfl_xor_sync 进行的是蝶形交换 width 内 按照 mask长度继续分组 和上一个或者下一个分组的交换

__shfl_xor_sync(0xffffffff, val, 4)
    0   1   2   3   4   5   6   7 |  8   9   10  11  12  13  14  15  16  17  ...
    4   5   6   7   0   1   2   3 | 12  13  14  15  8   9   10  11  20  21  ...

WarpReduce warp 对32个数进行规约
*/
template<template<typename> typename ReductionOp, typename T, int width = 32>
__inline__ __device__ T WarpReduce(T val)
{
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
        // mask = 16, 8, 4, 2, 1
        // __shfl_down_sync 是不是也能实现这样的功能, 或者说性能能够更快？
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, width));

    return val;
}


// template<template<typename> typename ReductionOp, typename T, int block_size>
// __inline__ __device__ T BlockReduce(T val)
// {
//     typedef cub::BlockReduce<T, block_size> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     __shared__ T result_broadcast;
//     T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());

//     if (threadIdx.x == 0)  result_broadcast = result;

//     __syncthreads();
//     return result_broadcast;
// }

template<typename T, int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access, bool padding>
__global__ void warp_sofrmax_padding_impl(DirectLoad<T, typename  DefaultComputeType<T>::type> load,
    DirectStore<typename DefaultComputeType<T>::type, T> store, int rows, int cols)
{
    constexpr int num_packs = cols_per_thread / pack_size;
    using ComputeType = typename DefaultComputeType<T>::type;
    ComputeType buf[rows_per_access][cols_per_thread];
    // 线程为 block (thread_group_width, thread_groups_per_block)
    // grid         num_blocks
    // thread_group_width 计算 < 1024的一行/两行，global_thread_group_id 这应该是计算行索引
    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    // 总共开了多少个线程组 num_global_thread_group * 1/2 近似于 rows
    const int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;

    // 如果开的grid太小，不能算完则需要迭代进行计算下一个step，这里开够足够多的grid 可以没有这个循环，但是可能会有tail effect
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
        // 每个线程只持有rows_per_access 的数据，然后进行warp shlf reduce
        ComputeType thread_max[rows_per_access]; // 1/2
#pragma unroll

        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_max[row_id] = -Inf<ComputeType>();
            ComputeType* row_buf = buf[row_id];
#pragma unroll

            // 每个线程先把自己的 cols_per_thread 算完max，比如cols = 1023
            // 则会切分为 thread_group_width 32 cols_per_pthread  32
            // 所以 最终独自 32个元素求max，最后一把进行warp shlf reduce
            // pack_size = 1 num_packs = 32
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;

                if (!padding || col < cols) {
                    // 这个里面的load 仍然是以循环的方式load的, 所以能不能用上vectorize 暂时还不知道
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll

                    for (int i = 0; i < pack_size; ++i)
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                } else {
#pragma unroll

                    for (int i = 0; i < pack_size; ++i)  row_buf[pack_offset + i] = -Inf<ComputeType>();
                }
            }
        }

        ComputeType warp_max[rows_per_access];
#pragma unroll

        // 已经算完 reduce_max
        for (int row_id = 0; row_id < rows_per_access; ++row_id)
            warp_max[row_id] = WarpReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);

        // 开始算 exp(broadcase sub in, in_max) 以及warp sum
        ComputeType thread_sum[rows_per_access];
#pragma unroll

        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_sum[row_id] = 0;
            ComputeType* row_buf = buf[row_id];
#pragma unroll

            for (int i = 0; i < cols_per_thread; ++i) {
                row_buf[i] = expf(row_buf[i] - warp_max[row_id]);
                thread_sum[row_id] += row_buf[i];
            }
        }

        ComputeType warp_sum[rows_per_access];
#pragma unroll

        for (int row_id = 0; row_id < rows_per_access; ++row_id)
            warp_sum[row_id] = WarpReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);

#pragma unroll

        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            ComputeType* row_buf = buf[row_id];
#pragma unroll

            for (int i = 0; i < cols_per_thread; ++i)
                row_buf[i] = row_buf[i] / warp_sum[row_id];

#pragma unroll

            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * thread_group_width + lane_id) * pack_size;

                if (!padding || col < cols)
                    store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
            }
        }
    }
}


template<typename T, int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access, bool padding>
void launch_warp_softmax_padding_impl(T* in, T* out, int rows, int cols)
{
    // 整个block的大小固定为128
    constexpr int block_size = 128;
    constexpr int waves = 32;
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    // 处理cols的，能够处理多少个cols * rows_per_access的
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    // 按照行 load来计算grid 的大小 rows_per_access = 2的时候 相应的load次数也少? 这里也有个疑问  可以用更大的row_per_access 来增加单个kernel的计算量吗？
    const int64_t num_blocks = (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    CHECK_CALL_ERROR(GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x));
    using ComputeType = typename DefaultComputeType<T>::type;
    auto load = DirectLoad<T, ComputeType>(in, cols);
    auto store = DirectStore<ComputeType, T>(out, cols);
    printf("called kernel\n");
    warp_sofrmax_padding_impl<T, pack_size, cols_per_thread, thread_group_width, rows_per_access, padding>
    <<< grid_dim_x, block_dim>>>
    (load, store, rows, cols);
}

// template<typename T>
// __gloabl
template<typename T, int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access>
void dispatch_warp_softmax_padding_impl(T* in, T* out, int rows, int cols)
{
    // 不需要进行padding的case
    if (cols == cols_per_thread * thread_group_width)
        launch_warp_softmax_padding_impl<T, pack_size, cols_per_thread, thread_group_width, rows_per_access, false>(in, out, rows, cols);
    else
        launch_warp_softmax_padding_impl<T, pack_size, cols_per_thread, thread_group_width, rows_per_access, true>(in, out, rows, cols);
}


// cols 是奇数的case
template<typename T, int pack_size>
typename enable_if<pack_size == 1, void>::type dispatch_warp_softmax_impl(T* in, T* out, int rows, int cols)
{
    if (cols <= 1) {}

    // cols 1-32 现成的thread_group_width  1/32 1/16 1/8 1/4 1/2 1/1大小的warp
#define DEFINE_ONE_ELIF(thread_group_width)                                                                          \
    else if (cols <= (thread_group_width)*pack_size) {                                                               \
        if (rows % 2 == 0) {                                                                                         \
            dispatch_warp_softmax_padding_impl<T, pack_size, pack_size, thread_group_width, 2>(in, out, rows, cols); \
        } else {                                                                                                     \
            dispatch_warp_softmax_padding_impl<T, pack_size, pack_size, thread_group_width, 1>(in, out, rows, cols); \
        }                                                                                                            \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    // cols 64 - 1024
#define DEFINE_ONE_ELIF(col)                                                                        \
    else if (cols <= (col)*WARP_SIZE) {                                                              \
        dispatch_warp_softmax_padding_impl<T, pack_size, col, WARP_SIZE, 1>(in, out, rows, cols);   \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
}

template<typename T, int pack_size>
typename enable_if<pack_size == 2, void>::type dispatch_warp_softmax_impl(T* in, T* out, int rows, int cols)
{
    if (cols <= 1) {}

    // cols 1-32 现成的thread_group_width  1/32 1/16 1/8 1/4 1/2 1/1大小的warp
#define DEFINE_ONE_ELIF(thread_group_width)                                                                          \
    else if (cols <= (thread_group_width)*pack_size) {                                                               \
        if (rows % 2 == 0) {                                                                                         \
            dispatch_warp_softmax_padding_impl<T, pack_size, pack_size, thread_group_width, 2>(in, out, rows, cols); \
        } else {                                                                                                     \
            dispatch_warp_softmax_padding_impl<T, pack_size, pack_size, thread_group_width, 1>(in, out, rows, cols); \
        }                                                                                                            \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    // cols 64 - 1024
#define DEFINE_ONE_ELIF(col)                                                                        \
    else if (cols <= (col)*WARP_SIZE) {                                                              \
        dispatch_warp_softmax_padding_impl<T, pack_size, col, WARP_SIZE, 1>(in, out, rows, cols);   \
    }
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
}

template<typename T>
void dispatch_warp_softmax(T* in, T* out, int rows, int cols)
{
    if (cols % 2 == 0)
        dispatch_warp_softmax_impl<T, 2>(in, out, rows, cols);
    else
        dispatch_warp_softmax_impl<T, 1>(in, out, rows, cols);
}


template<typename T>
void softmax_v0_warpper(T* in, T* out, int rows, int cols)
{
    if (cols < 1024)
        dispatch_warp_softmax<T>(in, out, rows, cols);
    else {
        // block reduce softmax
        // unshared reduce softmax
    }
}


int main(int argc, char const* argv[])
{
    /* code */
    constexpr int MAX_ROWS = 32 * 12 * 128, MAX_COLS = 4096;
    REGISTER_BUFF(A, bytes_of<float>(MAX_ROWS * MAX_COLS));
    REGISTER_BUFF(B, bytes_of<float>(MAX_ROWS * MAX_COLS));
    genOrLoad<float>(string("test_float32.bin"), h_A, (size_t)0);
    CHECK_CALL_ERROR(cudaMemcpy(d_A, h_A, bytes_of<float>(MAX_ROWS * MAX_COLS), cudaMemcpyHostToDevice));
    softmax_v0_warpper((float*)d_A, (float*)d_B, MAX_ROWS, 1021);
    CHECK_CUDA_ERROR();
    CHECK_CALL_ERROR(cudaMemcpy(h_B, d_B, bytes_of<float>(MAX_ROWS * 1021), cudaMemcpyDeviceToHost));
    dump_to_file("softmax_out_1021.bin", h_B, bytes_of<float>(MAX_ROWS * 1021));
    softmax_v0_warpper((float*)d_A, (float*)d_B, MAX_ROWS, 1022);
    CHECK_CUDA_ERROR();
    CHECK_CALL_ERROR(cudaMemcpy(h_B, d_B, bytes_of<float>(MAX_ROWS * 1022), cudaMemcpyDeviceToHost));
    dump_to_file("softmax_out_1022.bin", h_B, bytes_of<float>(MAX_ROWS * 1022));
    return 0;
}
