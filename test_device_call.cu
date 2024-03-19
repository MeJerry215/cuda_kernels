#include "common.h"


template<typename T>
struct VectorizeLoad {};

template<>
struct VectorizeLoad<float> {
    VectorizeLoad(void* src, int64_t row_size): src(src), row_size(row_size) {}
    __device__ void load1(void* dst, int64_t row, int64_t col) const
    {
        *((float*)dst) = *((float*)((float*)src + row * row_size + col));
    }
    __device__ void load2(void* dst, int64_t row, int64_t col) const
    {
        *((float2* )dst) = *((float2*)((float*)src + row * row_size + col));
    }
    __device__ void load4(void* dst, int64_t row, int64_t col) const
    {
        *((float4*)dst) = *((float4*)((float*)src + row * row_size + col));
    }
    int64_t row_size;
    void* src;
};

__global__ void test_load(VectorizeLoad<float> load, float* dst)
{
    // 不能这么搞
    __device__ void* load_now_func = load.load4;
    __device__ void* load_rem_func = load.load2;
    load_now_func(dst, 0, 0);
    load_rem_func(dst + 4, 0, 4);
}



int main(int argc, char const* argv[])
{
    REGISTER_BUFF(A, bytes_of<float>(6));
    REGISTER_BUFF(B, bytes_of<float>(6));

    for(int i = 0; i < 6; i++)
        ((float*)h_A)[i] = i;

    CHECK_CALL_ERROR(cudaMemcpy(d_A, h_A, bytes_of<float>(6), cudaMemcpyHostToDevice));
    VectorizeLoad<float> load = VectorizeLoad<float>(h_A, 1);
    test_load<float> <<< 1,  1>>>(load, d_B);
    CHECK_CALL_ERROR(cudaMemcpy(h_B, d_B, bytes_of<float>(6), cudaMemcpyDeviceToHost));
    return 0;
}
