#include "common.h"

__global__ void relu_kernel_v1(void* x_ptr, void* y_ptr, int32_t n_elem)
{
    float *x = (float*)x_ptr;
    float *y = (float*)y_ptr;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elem) y[idx] = x[idx] < 0 ? 0 : x[idx];
}

const int mpCnt = A10_SM_CNT;
const int spCnt = A10_SP_CNT;
const int min_elem_cnt = A10_SM_CNT * A10_SP_CNT;

template<int n_per_thread = 1>
__global__ void relu_kernel_v2(void* x_ptr, void* y_ptr, int32_t n_elem)
{
    float *x = (float*)x_ptr;
    float *y = (float*)y_ptr;
    int n_elem_per_thread = n_per_thread;
    if (n_elem < min_elem_cnt) n_elem_per_thread = 1;
    int32_t idx = blockIdx.x * blockDim.x * n_elem_per_thread + threadIdx.x;
    for (int i = 0; i < n_elem_per_thread; i++) {
        if (idx < n_elem) y[idx] = x[idx] < 0 ? 0 : x[idx];
        idx += blockDim.x;
    }
}

__global__ void relu_kernel_v3(void* x_ptr, void *y_ptr, int32_t n_elem)
{
    float2 *x = (float2*)y_ptr;
    float2 *y = (float2*)y_ptr;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx * 2 + 1) < n_elem) {
        float2 tmp = x[idx];
        tmp.x = tmp.x < 0 ? 0 : tmp.x;
        tmp.y = tmp.y < 0 ? 0 : tmp.y;
        y[idx] = tmp;
    }
}

__global__ void relu_kernel_v4(void* x_ptr, void *y_ptr, int32_t n_elem)
{
    float4 *x = (float4*)y_ptr;
    float4 *y = (float4*)y_ptr;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx * 4 + 3) < n_elem) {
        float4 tmp = x[idx];
        tmp.x = tmp.x < 0 ? 0 : tmp.x;
        tmp.y = tmp.y < 0 ? 0 : tmp.y;
        tmp.z = tmp.z < 0 ? 0 : tmp.z;
        tmp.w = tmp.w < 0 ? 0 : tmp.w;
        y[idx] = tmp;
    }
}


__global__ void relu_kernel_v5(void* x_ptr, void *y_ptr, int32_t n_elem)
{
    float2 *x = (float2*)y_ptr;
    float2 *y = (float2*)y_ptr;
    for(int i = 0; i < 2; i++) {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx * 2 + 1) < n_elem) {
            float2 tmp = x[idx];
            tmp.x = tmp.x < 0 ? 0 : tmp.x;
            tmp.y = tmp.y < 0 ? 0 : tmp.y;
            y[idx] = tmp;
        }
    }
}

pair<size_t, pair<int, int>> calculate_grid_block_v1(size_t n_elem)
{
    int block_size = min(size_t(max_block_threads), n_elem);
    int grid_size = (n_elem + block_size - 1) / block_size;
    return {n_elem, {grid_size, block_size}};
}

template<int n_per_thread = 1>
pair<size_t, pair<int, int>> calculate_grid_block_v2(size_t n_elem)
{
    assert(mpCnt > 0 && spCnt > 0);
    int best_of_grid = 0;
    int best_of_block = 0;
    int n_elem_per_thread = n_per_thread;
    int n = min(size_t(n_elem + warp_size - 1), size_t(max_block_threads)) / warp_size;
    int block_size = warp_size;
    // if (n_elem <= min_elem_cnt) n_elem_per_thread = 1;
    int block_elem = block_size * n_elem_per_thread;
    do {
        best_of_block = block_size;
        best_of_grid = (n_elem + block_elem - 1) / block_elem;
        block_size += warp_size;
        block_elem = block_size * n_elem_per_thread;
    } while((n_elem + block_elem - 1) / block_elem >= mpCnt && block_size <= max_block_threads);

    return {n_elem, {best_of_grid, best_of_block}};
}

#define relu_kernel          relu_kernel_v5
#define calculate_grid_block calculate_grid_block_v2<4>

/*
1. nv卡上提高并行度，使用更多的grid size，A10上 72 个MP，对于2048这样的数据，可以采用64 grid 32 block这样的case， warp最小的调度尺寸是32，所以block size 为32的倍数
2. vector 访问 float2 和float4 的效果相近，但是比float好，warp32 * 4 * 2 = 256 float2 
*/

int main(int argc, char** argv)
{
    // auto mp_sp = query_mp_sp(0);
    // mpCnt = mp_sp.first;
    // spCnt = mp_sp.first;
    const size_t max_elem_cnts = 1024 * 1024 * 32;
    vector<pair<size_t, pair<int, int>>> test_cases = {
        calculate_grid_block(2048),
        calculate_grid_block(4096),
        calculate_grid_block(8192),
        calculate_grid_block(16384),
        calculate_grid_block(1024 * 512),
        calculate_grid_block(1024 * 1024),
        calculate_grid_block(1024 * 1024 * 32)
    };
    REGISTER_BUFF(x, bytes_of<float>(max_elem_cnts));
    REGISTER_BUFF(y, bytes_of<float>(max_elem_cnts));
    gen_random<float>((float*)h_x, max_elem_cnts);
    CHECK_CALL_ERROR(cudaMemcpy(d_x, h_x, max_elem_cnts, cudaMemcpyHostToDevice));

    for (auto it = test_cases.begin(); it != test_cases.end(); ++it) {
        int elem_cnt = it->first;
        dim3 blockSize(it->second.second);
        dim3 gridSize(it->second.first);
        cout << "elem_cnt " << elem_cnt << " block size " << blockSize.x << " grid size " << gridSize.x << endl;
        relu_kernel <<< gridSize, blockSize>>>(d_x, d_y, elem_cnt);
    }

    FREE_BUFF(x);
    FREE_BUFF(y);
    // 72 * 128
}

