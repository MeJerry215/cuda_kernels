#include "common.h"

/*
1. Thread Divergence: Instructions are executed in warps, which are groups of 32 threads. Optimal instruction
throughput is achieved if all 32 threads of a warp execute the same instruction. The chosen
launch configuration, early thread completion, and divergent flow control can significantly
lower the number of active threads in a warp per cycle. This kernel achieves an average of 31.7 threads
being active per cycle. This is further reduced to 17.8 threads per warp due to predication.
The compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled,
but a per-thread condition code or predicate controls which threads execute the instructions.
Try to avoid different execution paths within a warp when possible.
2. Barrier Stalls: On average, each warp of this kernel spends 13.5 cycles being stalled waiting for sibling warps
at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code
paths before a barrier. This causes some warps to wait a long time until other warps reach the
synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads.
If the block size is 512 threads or greater, consider splitting it into smaller groups.
This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy
limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code
executed before that synchronization point first. This stall type represents about 34.5% of the
total average of 39.0 cycles between issuing two instructions.
3. The difference between calculated theoretical (100.0%) and measured achieved occupancy (87.3%)
can be the result of warp scheduling overheads or workload imbalances during the kernel execution.
Load imbalances can occur between warps within a block as well as across blocks of the same kernel.
See the  CUDA Best Practices Guide for more details on optimizing occupancy.

*/

#define BLOCK_THREADS 256
#define N_ELEMS 2

template<typename T, int THREADS_PER_BLOCK = 256, int ELEMS_PER_THREAD = 1>
__global__ void reudce_v0(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[THREADS_PER_BLOCK];
    int bx = blockIdx.x, tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    sdata[tx] = tid < n_elem ? x[tid] : 0;
    __syncthreads();
    // if (bx == 0 && tx == 0) {
    //     for(int i = 0; i < THREADS_PER_BLOCK; i++) {
    //         printf("%f ", sdata[i]);
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    for(int i = 1; i < THREADS_PER_BLOCK; i *= 2) {
        if (tx % (i * 2) == 0)
            sdata[tx] += sdata[tx + i];

        __syncthreads();
    }

    // printf("write y[%d]=%f tid %d\n", bx, sdata[0], tx);
    if(tx == 0) y[bx] = sdata[tx];
}


/*
1. This kernel has uncoalesced shared accesses resulting in a total of 430080 excessive wavefronts
(70% of the total 614400 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary
source locations. The  CUDA Best Practices Guide has an example on optimizing shared memory accesses.
2. The memory access pattern for shared loads might not be optimal and causes on average a 3.8 -
way bank conflict across all 102400 shared load requests.This results in 286727 bank conflicts,
which represent 72.87% of the overall 393480 wavefronts for shared loads. Check the  Source Counters
section for uncoalesced shared loads.
3. The memory access pattern for shared stores might not be optimal and causes on average a 2.8 -
way bank conflict across all 81920 shared store requests.This results in 143674 bank conflicts,
which represent 63.69% of the overall 225592 wavefronts for shared stores. Check the  Source Counters
section for uncoalesced shared stores.
*/
template<typename T, int THREADS_PER_BLOCK = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v1(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[THREADS_PER_BLOCK];
    int bx = blockIdx.x, tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;

    if (tid < n_elem) sdata[tx] = x[tid]; //tid < n_elem ? x[tid] : 0;

    __syncthreads();

    for(int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tx;

        if(index < blockDim.x)
            sdata[index] += sdata[index + s];

        __syncthreads();
    }

    if(tx == 0) y[bx] = sdata[tx];
}



template<typename T, int THREADS_PER_BLOCK = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v2(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[THREADS_PER_BLOCK];
    int bx = blockIdx.x, tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;

    if (tid < n_elem) sdata[tx] = x[tid]; //tid < n_elem ? x[tid] : 0;

    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tx < s)
            sdata[tx] += sdata[tx + s];

        __syncthreads();
    }

    if(tx == 0) y[bx] = sdata[tx];
}

template<typename T, int THREADS_PER_BLOCK = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v3(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[THREADS_PER_BLOCK];
    int bx = blockIdx.x, tx = threadIdx.x;
    int tid = bx * blockDim.x * ELEMS_PER_THREAD + tx;
    // sdata[tx] =  (tid < n_elem ? x[tid] : 0)  + (tid + blockDim.x < n_elem ? x[tid] : 0);
    sdata[tx] = (tid < n_elem ? x[tid] : 0) +  (tid + blockDim.x < n_elem ? x[tid + blockDim.x] : 0); //x[tid + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tx < s)
            sdata[tx] += sdata[tx + s];

        __syncthreads();
    }

    if(tx == 0) y[bx] = sdata[tx];
}


/* 这里必须要使用 volatile， 否则编译器会优化导致结果不对 */

template<int BLOCK_SIZE = 256>
__device__ void warpReduce(volatile float* cache, int tid)
{
    if (BLOCK_SIZE >= 64) cache[tid] += cache[tid + 32];

    if (BLOCK_SIZE >= 32) cache[tid] += cache[tid + 16];

    if (BLOCK_SIZE >= 16) cache[tid] += cache[tid + 8];

    if (BLOCK_SIZE >= 8) cache[tid] += cache[tid + 4];

    if (BLOCK_SIZE >= 4) cache[tid] += cache[tid + 2];

    if (BLOCK_SIZE >= 2) cache[tid] += cache[tid + 1];
}
/*
Loop unroll
*/
template<typename T, int BLOCK_SIZE = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v4(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[BLOCK_SIZE];
    int bx = blockIdx.x, tx = threadIdx.x;
    int tid = bx * blockDim.x * ELEMS_PER_THREAD + tx;
    sdata[tx] = (tid < n_elem ? x[tid] : 0) +  (tid + blockDim.x < n_elem ? x[tid + blockDim.x] : 0); //x[tid + blockDim.x];
    __syncthreads();

    if(tx < 128) sdata[tx] += sdata[tx + 128];

    __syncthreads();

    if(tx < 64) sdata[tx] += sdata[tx + 64];

    __syncthreads();

    if(tx < 32) warpReduce<BLOCK_SIZE>(sdata, tx);

    if(tx == 0) y[bx] = sdata[tx];
}


template<typename T, int BLOCK_SIZE = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v5(T* x, T* y, int32_t n_elem)
{
    __shared__  T sdata[BLOCK_SIZE];
    int bx = blockIdx.x, tx = threadIdx.x;
    sdata[tx] = 0;
    int idx = bx * BLOCK_SIZE * ELEMS_PER_THREAD + tx;

    if (ELEMS_PER_THREAD > 1) {
        int n_iter = max(1, ELEMS_PER_THREAD / 2);
        int32_t stride = BLOCK_SIZE * 2;

        while(n_iter--) {
            sdata[tx] += (idx < n_elem ? x[idx] : 0) + ((idx + BLOCK_SIZE) < n_elem ? x[idx + BLOCK_SIZE] : 0);
            idx += stride;
        }
    } else
        sdata[tx] = (idx < n_elem ? x[idx] : 0);

    __syncthreads();

    if (BLOCK_SIZE >= 1024) {
        if (tx < 512) sdata[tx] += sdata[tx + 512];

        __syncthreads();
    }

    if (BLOCK_SIZE >= 512) {
        if (tx < 256) sdata[tx] += sdata[tx + 256];

        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tx < 128) sdata[tx] += sdata[tx + 128];

        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tx < 64) sdata[tx] += sdata[tx + 64];

        __syncthreads();
    }

    if (tx < 32) warpReduce<BLOCK_SIZE>(sdata, tx);

    if (tx == 0) y[bx] = sdata[tx];
}

template <unsigned int BLOCK_SIZE = 256>
__device__ __forceinline__ float warpReduceSum(float sum)
{
    if(BLOCK_SIZE >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16);

    if(BLOCK_SIZE >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);

    if(BLOCK_SIZE >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);

    if(BLOCK_SIZE >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);

    if(BLOCK_SIZE >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);

    return sum;
}


template<typename T, int BLOCK_SIZE = 256, int ELEMS_PER_THREAD = 1>
__global__ void reduce_v6(T* x, T* y, int32_t n_elem)
{
    float sum = 0.0f;
    int bx = blockIdx.x, tx = threadIdx.x;
    int idx = bx * BLOCK_SIZE * ELEMS_PER_THREAD + tx;

    if (ELEMS_PER_THREAD > 1) {
        int n_iter = max(1, ELEMS_PER_THREAD / 2);
        int32_t stride = BLOCK_SIZE * 2;

        while(n_iter--) {
            sum += (idx < n_elem ? x[idx] : 0) + ((idx + BLOCK_SIZE) < n_elem ? x[idx + BLOCK_SIZE] : 0);
            idx += stride;
        }
    } else
        sum = (idx < n_elem ? x[idx] : 0);

    __syncthreads();
    __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = tx % WARP_SIZE;
    const int warpId = tx / WARP_SIZE;
    sum = warpReduceSum<BLOCK_SIZE>(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;

    __syncthreads();
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

    if(warpId == 0)sum = warpReduceSum < BLOCK_SIZE / WARP_SIZE > (sum);

    if (tx == 0) y[bx] = sum;
}

template<int BLOCK_SIZE = 256, int ELEMS_PER_THREAD = 1>
pair<size_t, vector<int>> calculate_grid_block(int32_t n_elem)
{
    int block_elems = BLOCK_SIZE * ELEMS_PER_THREAD;
    int grid_size = (n_elem + block_elems - 1) / block_elems;
    return {n_elem, {grid_size, BLOCK_SIZE, ELEMS_PER_THREAD}};
}

vector<float*> gen_exp_cpu(float* x, vector<pair<size_t, vector<int>>>& testcases)
{
    vector<float*> results;

    for (auto testcase : testcases) {
        int stride = testcase.second[1] * testcase.second[2];
        size_t n_elem = testcase.first;
        float* result = (float*)malloc(bytes_of<float>(testcase.second[0]));
        float sum = 0.0f;

        for(int i = 0; i < n_elem; i++) {
            sum += x[i];

            if ((i + 1) % stride == 0 || (i + 1) == n_elem) {
                result[i / stride] = sum;
                sum = 0.0f;
            }
        }

        results.push_back(result);
    }

    return results;
}

#define reduce_kernel reduce_v5

template<typename T, int BLOCK_SIZE = 256, int ELEMS_PER_THREAD = 1>
void reduce_wrapper(T* x, T* y, int32_t n_elem, dim3 grid, dim3 block)
{
    reduce_kernel<T, BLOCK_SIZE, ELEMS_PER_THREAD> <<< grid, block>>>(x, y, n_elem);
}


using KernelFunction = void (*)(float*, float*, int32_t, dim3, dim3);

unordered_map<string, KernelFunction> kernel_map;

void register_kernel(const char* name, KernelFunction func)
{
    assert(func != nullptr);
    kernel_map[string(name)] = func;
}

KernelFunction query_kernel(int block_size, int elems_per_thread)
{
    string name = to_string(block_size) + "_" + to_string(elems_per_thread) + "_reduce";
    auto func = kernel_map[name];

    if (func == nullptr) {
        cout << "block size " << block_size << " elems_per_thread " << elems_per_thread << " kernel not found." << endl;
        assert(false);
    }

    return func;
}

#define REGISTER_KERNEL(dtype, BLOCK_SIZE, ELEMS_PER_THREAD) \
    register_kernel(#BLOCK_SIZE "_" #ELEMS_PER_THREAD "_reduce", reduce_wrapper<dtype, BLOCK_SIZE, ELEMS_PER_THREAD>);


#define REGISTER_FLOAT_KERNEL(BLOCK_SIZE, ELEMS_PER_THREAD) REGISTER_KERNEL(float, BLOCK_SIZE, ELEMS_PER_THREAD)

#define REGISTER_FLOAT_KERNEL_ALL(BLOCK_SIZE) \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 128)   \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 64)   \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 32)   \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 16)   \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 8)    \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 4)    \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 2)    \
    REGISTER_FLOAT_KERNEL(BLOCK_SIZE, 1)


void register_kernels()
{
    // kernel_map = unordered_map<string, KernelFunction>();
    REGISTER_FLOAT_KERNEL_ALL(1024)
    REGISTER_FLOAT_KERNEL_ALL(512)
    REGISTER_FLOAT_KERNEL_ALL(256)
    REGISTER_FLOAT_KERNEL_ALL(128)
    REGISTER_FLOAT_KERNEL_ALL(64)
    REGISTER_FLOAT_KERNEL_ALL(32)
}

int main(int argc, char** argv)
{
    register_kernels();
    // const size_t max_elem_cnts = 512 * 1024 * 1024;
    const size_t max_elem_cnts = 536870912;
    srand(SEED);

    // v6
    vector<pair<size_t, vector<int>>> testcases = {
        calculate_grid_block<32, 1>(2048),  //  这个数据量的情况下, 要开到最够多的grid 提高并行度 就只能使用小的warp size
        calculate_grid_block<64, 2>(4096),
        calculate_grid_block<32, 2>(4096),   // 性能最优case 一次算两个数，且grid size 64 调度一轮
        calculate_grid_block<32, 16>(32768),
        calculate_grid_block<64, 8>(32768),
        calculate_grid_block<128, 8>(32768),
        calculate_grid_block<32, 8>(32768),
        calculate_grid_block<32, 8>(524288),
        calculate_grid_block<128, 8>(524288),
        calculate_grid_block<64, 32>(524288),
        calculate_grid_block<128, 4>(524288),
        calculate_grid_block<256, 4>(33554432),
        calculate_grid_block<128, 4>(33554432),
        calculate_grid_block<512, 4>(536870912),
        calculate_grid_block<256, 4>(536870912),
    };

    // v5
    // vector<pair<size_t, vector<int>>> testcases = {
    //     calculate_grid_block<32, 1>(2048),
    //     calculate_grid_block<32, 2>(4096),
    //     calculate_grid_block<32, 8>(32768),
    //     calculate_grid_block<64, 4>(32768),
    //     calculate_grid_block<32, 16>(32768),
    //     calculate_grid_block<128, 8>(32768),
    //     calculate_grid_block<64, 4>(524288),
    //     calculate_grid_block<32, 8>(524288),
    //     calculate_grid_block<64, 16>(524288),
    //     calculate_grid_block<32, 16>(524288),
    //     calculate_grid_block<256, 4>(33554432),
    //     calculate_grid_block<128, 4>(33554432),
    //     calculate_grid_block<256, 4>(536870912),
    //     calculate_grid_block<512, 4>(536870912),
    // }

    REGISTER_BUFF(x, bytes_of<float>(max_elem_cnts));
    REGISTER_BUFF(y, bytes_of<float>(CEIL_DIV(max_elem_cnts, BLOCK_THREADS)));
    gen_random<float>((float*)h_x, max_elem_cnts);

    for(int i = 0; i < 256; i++)
        ((float*)h_x)[i] = 0.01 * i;

    auto expects = gen_exp_cpu((float*)h_x, testcases);
    CHECK_CALL_ERROR(cudaMemcpy(d_x, h_x, bytes_of<float>(max_elem_cnts), cudaMemcpyHostToDevice));
    int cnt = 0;

    for (auto it = testcases.begin(); it != testcases.end(); ++it) {
        int elem_cnt = it->first;
        dim3 block_size(it->second[1]);
        dim3 grid_size(it->second[0]);
        cout << "elem_cnt " << elem_cnt << " grid size " << grid_size.x <<  " block size " << block_size.x  << " elems " << it->second[2]
            <<
            endl;
        auto kernel = query_kernel(it->second[1], it->second[2]);
        kernel((float*)d_x, (float*)d_y, elem_cnt, grid_size, block_size);
        CHECK_CALL_ERROR(cudaMemcpy(h_y, d_y, bytes_of<float>(grid_size.x), cudaMemcpyDeviceToHost));
        // printf("%f %f %f %f\n", ((float*)h_y)[0], expects[cnt][0], ((float*)h_y)[1], expects[cnt][1]);
        cout << count_large_diff((float*)h_y, expects[cnt], grid_size.x, 1e-3, 1e-3) << "/" << grid_size.x << " differs." << endl;
        cnt ++;
        MEMSET_BUF(y, bytes_of<float>(grid_size.x));
    }

    FREE_BUFF(x);
    FREE_BUFF(y);
}