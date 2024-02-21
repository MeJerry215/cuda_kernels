#include "common.h"


enum Major {
    RowMajor = 0,
    ColMajor = 0,
};

#define ELEME_OF(p, x, y, s) (p[(x) * (s) + (y)])
#define sgemm_nn sgemm_v0_nn
using KernelFunction = void (*)(float*, float*, float*, int, int, int, dim3, dim3, Major);

// https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_3.cu
// M, K @ K, N -> M, N  raw implementation
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_SIZE_M, int THREAD_SIZE_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void sgemm_v0_nn(T* __restrict__  A, T* __restrict__  B, T* __restrict__  C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * BLOCK_M + tx, y = by * BLOCK_N + ty;

    if (x >= M || y >= N) return;

    T sum = zero<T>();

    for(int i = 0; i < K; i++)
        sum = fmaf(ELEME_OF(A, x, i, K), ELEME_OF(B, i, y, N), sum);

    ELEME_OF(C, x, y, N) = sum;
}




// basic optimization method: blocking load
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_SIZE_M, int THREAD_SIZE_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void sgemm_v1_nn(T* A, T* B, T* C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
}

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_SIZE_M, int THREAD_SIZE_N>
void sgemm_wrapper(T* A, T* B, T* C, int M, int N, int K, dim3 grid, dim3 block, Major major)
{
    if (major == RowMajor)
        sgemm_nn<T, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_SIZE_M, THREAD_SIZE_N> <<< grid, block>>>(A, B, C, M, N, K);
    else if (major == ColMajor)
        assert(false);
}

vector<float*> gen_exp_cpu(float* x, float* y, vector<pair<vector<int>, vector<int>>>& testcases)
{
    vector<float*> results;
    unordered_map<vector<int>, float*> cache;

    for (auto testcase : testcases) {
        int M = testcase.second[0];
        int N = testcase.second[1];
        int K = testcase.second[2];
        vector<int> key = {M, N, K};
        auto it = cache.find(key);

        if (it != cache.end()) {
            cout << "find cache case M " << M << " N " << N << " K " << K << endl;
            results.push_back(cache[key]);
            continue;
        }

        cout << "running case M " << M << " N " << N << " K " << K << endl;
        float* z = (float*)malloc(bytes_of<float>(M * N));

        for (int i = 0; i < M; i ++ ) {
            for (int j = 0; j < N; j ++) {
                float val = 0;

                for(int k = 0; k < K; k++) {
                    // printf("x[%d * %d + %d] * y[%d * %d + %d] = %f * %f\n", i, K, k, k, N, j, x[i * K + k], y[k * N + j]);
                    val += x[i * K + k] * y[k * N + j];
                }

                z[i * N + j] = val;
            }
        }

        results.push_back(z);
        cache[key] = z;
    }

    return results;
}


unordered_map<vector<int>, KernelFunction> kernel_map;

void register_kernel(vector<int> config, KernelFunction func)
{
    assert(func != nullptr);
    kernel_map[config] = func;
}

KernelFunction query_kernel(vector<int> config)
{
    auto func = kernel_map[config];

    if (func == nullptr) {
        // cout << "kernel not found M = " << config[0] << " N = " << config[1] << " K = " << config[2] << endl;
        assert(false);
    }

    return func;
}

#define REGISTER_KERNEL(dtype, BM, BN, BK, WM, WN) \
    register_kernel({BM, BN, BK, WM, WN}, sgemm_wrapper<dtype, BM, BN, BK, WM, WN>);

#define REGISTER_FLOAT_KERNEL(BM, BN, BK, WM, WN) REGISTER_KERNEL(float, BM, BN, BK, WM, WN)

void register_kernels()
{
    // v0 case
    // 1024 case
    REGISTER_FLOAT_KERNEL(1, 1024, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 512, 0, 0, 0);
    // 512 case
    REGISTER_FLOAT_KERNEL(1, 512, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 256, 0, 0, 0);
    // 256 case
    REGISTER_FLOAT_KERNEL(1, 256, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 128, 0, 0, 0);
    // 128 case
    REGISTER_FLOAT_KERNEL(1, 128, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 64, 0, 0, 0);
    // 64
    REGISTER_FLOAT_KERNEL(1, 64, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 32, 0, 0, 0);
    // 32
    REGISTER_FLOAT_KERNEL(1, 32, 0, 0, 0);
    REGISTER_FLOAT_KERNEL(2, 16, 0, 0, 0);
}


void generate_testcase_v0(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K, int T)
{
    for(int i = 1; i <= 2; i *= 2) {
        testcases.push_back(
            make_pair<vector<int>, vector<int>>({i, T / i, 0, 0, 0}, {M, N, K})
        );
    }
}

#define generate_testcase generate_testcase_v0

int main(int argc, char** argv)
{
    register_kernels();
    const int MAX_M = 1024, MAX_N = 1024, MAX_K = 1024;
    vector<pair<vector<int>, vector<int>>> testcases = {
    };
    generate_testcase(testcases, 1024, 1024, 1024, 1024);
    generate_testcase(testcases, 1024, 1024, 1024, 512);
    generate_testcase(testcases, 1024, 1024, 1024, 256);
    generate_testcase(testcases, 1024, 1024, 1024, 128);
    generate_testcase(testcases, 1024, 1024, 1024, 64);
    generate_testcase(testcases, 1024, 1024, 1024, 32);
    REGISTER_BUFF(A, bytes_of<float>(MAX_M * MAX_K));
    REGISTER_BUFF(B, bytes_of<float>(MAX_N * MAX_K));
    REGISTER_BUFF(C, bytes_of<float>(MAX_M * MAX_N));
    gen_random<float>((float*)h_A, MAX_M * MAX_K);
    gen_random<float>((float*)h_B, MAX_N * MAX_K);
    auto expects = gen_exp_cpu((float*)h_A, (float*)h_B, testcases);
    CHECK_CALL_ERROR(cudaMemcpy(d_A, h_A, bytes_of<float>(MAX_M * MAX_K), cudaMemcpyHostToDevice));
    CHECK_CALL_ERROR(cudaMemcpy(d_B, h_B, bytes_of<float>(MAX_K * MAX_N), cudaMemcpyHostToDevice));

    for (int i = 0; i < testcases.size(); i++) {
        auto testcase = testcases[i];
        int M = testcase.second[0];
        int N = testcase.second[1];
        int K = testcase.second[2];
        MEMSET_BUF(C, bytes_of<float>(M * N));
        int block_x = testcase.first[0];
        int block_y = testcase.first[1];
        int grid_x = (M + block_x - 1) / block_x;
        int grid_y = (N + block_y - 1) / block_y;
        dim3 block(block_x, block_y);
        dim3 grid(grid_x, grid_y);
        cout << "running test M = " << M << " M = " << N << " K = " << K << " grid (" << grid.x << ", " << grid.y << ")"
            << " block (" << block.x << ", " << block.y << ")" << endl;
        auto kernel_wrapper = query_kernel(testcase.first);
        kernel_wrapper((float*)d_A, (float*)d_B, (float*)d_C, M, N, K, grid, block, RowMajor);
        CHECK_CALL_ERROR(cudaMemcpy(h_C, d_C, bytes_of<float>(M * N), cudaMemcpyDeviceToHost));
        cout << count_large_diff((float*)h_C, expects[i], M * N, 1e-3, 1e-3) << "/" << M* N << " differs." << endl;
    }

    FREE_BUFF(A);
    FREE_BUFF(B);
    FREE_BUFF(C);
}