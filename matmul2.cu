#include "common.h"


#define V0
#ifdef V0
    #define matmul_nn matmul_v0_nn
#elif defined(V1)

#elif defined(V2)

#endif


#define REGISTER_KERNEL(BM, BN, BK, WM, WN, WK, TM, TN) \
    register_kernel({BM, BN, BK, WM, WN, WK, TM, TN}, matmul_wrapper<BM, BN, BK, WM, WN, WK, TM, TN>);

using KernelFunction = void (*)(half*, half*, half*, int, int, int, dim3, dim3);
unordered_map<vector<int>, KernelFunction> kernel_map;

template<int BM, int BN, int BK, int WM, int WN, int WK, int TM, int TN>
void matmul_wrapper(half* A, half* B, half* C, int M, int N, int K, dim3 grid, dim3 block);



void register_kernel(vector<int> config, KernelFunction func)
{
    assert(func != nullptr);
    kernel_map[config] = func;
}

template<int BM, int BN, int BK, int WM, int WN, int WK, int TM, int TN>
__global__ void matmul_v0_nn(half* A, half* B, half* C, int M, int N, int K)
{
    // 一个线程算一个点 线程数为m * n
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * BM + tx, y = by * BN + ty;

    if (x >= M || y >= N) return;

    half sum = zero<half>();

    // int tn = tx / N;
    for (int i = 0; i < K; i++)
        sum = fmaf(ELEME_OF(A, x, i, K), ELEME_OF(B, i, y, N), sum);
}


void register_kernels()
{
#ifdef V0
#endif
}

KernelFunction query_kernel(vector<int> config)
{
    auto func = kernel_map[config];

    if (func == nullptr)
        assert(false);

    return func;
}


template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
void matmul_wrapper(T* A, T* B, T* C, int M, int N, int K, dim3 grid, dim3 block, Major major)
{
    matmul_nn<T, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N> <<< grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char const* argv[])
{
    register_kernels();
    const int MAX_M = 4096, MAX_N = 4096, MAX_K = 4096;
    srand(SEED);
    //  {BM, BN, BK, WM, WN, WK, TM, TN} {M, N, K}
    vector<pair<vector<int>, vector<int>>> testcases = {
        {{}},
    };
    REGISTER_BUFF(A, bytes_of<half>(MAX_M * MAX_K));
    REGISTER_BUFF(B, bytes_of<half>(MAX_N * MAX_K));
    REGISTER_BUFF(C, bytes_of<half>(MAX_M * MAX_N));
    genOrLoad<half>(string("bin/matmul_A_4096_4096_float16.bin"), h_A, MAX_M * MAX_K);
    genOrLoad<half>(string("bin/matmul_B_4096_4096_float16.bin"), h_B, MAX_M * MAX_K);
    CHECK_CALL_ERROR(cudaMemcpy(d_A, h_A, bytes_of<half>(MAX_M * MAX_K), cudaMemcpyHostToDevice));
    CHECK_CALL_ERROR(cudaMemcpy(d_B, h_B, bytes_of<half>(MAX_K * MAX_N), cudaMemcpyHostToDevice));

    for(auto testcase : testcases) {
        MEMSET_BUF(C, bytes_of<half>(MAX_M * MAX_N));
        int bm = testcase.first[0];
        int bn = testcase.first[1];
        int bk = testcase.first[2];
        int wm = testcase.first[3];
        int wn = testcase.first[4];
        int wk = testcase.first[5];
        int tm = testcase.first[6];
        int tn = testcase.first[7];
        int m = testcase.second[0];
        int n = testcase.second[1];
        int k = testcase.second[2];
#ifdef V0
        dim3 block(bm, bn);
        dim3 grid((m + bm - 1) / bm, (n + bn - 1) / bn);
#elif defined(CASE1)
#endif
        auto kernel_wrapper = query_kernel(testcase.first);
        kernel_wrapper((half*)d_A, (half*)d_B, (half*)d_C, m, n, k, grid, block);
    }

    return 0;
}
