#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void shfl_xor_example()
{
    int tid = threadIdx.x;
    int value = tid;  // Some value to be shuffled
    // Each thread shuffles its value with the neighbor two positions to the right
    int result = __shfl_xor_sync(0xFFFFFFFF, value, 4);
    printf("Thread %d: Original value = %d, Shuffled value = %d\n", tid, value, result);
}

int main()
{
    shfl_xor_example <<< 1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}