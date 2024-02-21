#include "common.h"


/**
 * @brief T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
 * 以width 分组，每个分组内的线程从srcLane 获取var，mask 无效
 */
__global__ void shfl_sync_test(float* y)
{
    int bx = blockIdx.x, tx = threadIdx.x;
    float val = bx + 0.01 * tx;
    val = __shfl_sync(0xffffffff, val, 1, 8);
    y[tx] = val;
}

/**
 * @brief T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
 *
 */
__global__  void shfl_up_test(float* y)
{
    int bx = blockIdx.x, tx = threadIdx.x;
    float val = bx + 0.01 * tx;
    val = __shfl_up_sync(0x00000000, val, 16, 32);
    y[tx] = val;
}

/**
 * @brief T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
 *
 */
__global__  void shfl_down_test(float* y)
{
    int bx = blockIdx.x, tx = threadIdx.x;
    float val = bx + 0.01 * tx;
    val = __shfl_down_sync(0x00000000, val, 16, 32);
    y[tx] = val;
}

// /**
//  * @brief T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
//  * 
//  */
// __global__  void shfl_xor_test(float* y) {
//     int bx = blockIdx.x, tx = threadIdx.x;
// }


int main(int argc, char** argv)
{
    REGISTER_BUFF(y, bytes_of<float>(32));
    dim3 grid(1);
    dim3 block(32);
    shfl_sync_test <<< grid, block>>>((float*)d_y);
    CHECK_CALL_ERROR(cudaMemcpy(h_y, d_y, bytes_of<float>(32), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 32; i++)
        cout << fixed << setprecision(2) <<  ((float*)h_y)[i] << " ";

    cout << endl;
    shfl_up_test <<< grid, block>>>((float*)d_y);
    CHECK_CALL_ERROR(cudaMemcpy(h_y, d_y, bytes_of<float>(32), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 32; i++)
        cout << fixed << setprecision(2) <<  ((float*)h_y)[i] << " ";
    cout << endl;

    shfl_down_test <<< grid, block>>>((float*)d_y);
    CHECK_CALL_ERROR(cudaMemcpy(h_y, d_y, bytes_of<float>(32), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 32; i++)
        cout << fixed << setprecision(2) <<  ((float*)h_y)[i] << " ";
    cout << endl;

    FREE_BUFF(y);
}