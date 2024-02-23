#include "common.h"

#define V4

#ifdef V0
    #define matmul_nn matmul_v0_nn
    #define generate_testcase generate_testcase_v0
    #define CASE0
#elif defined(V1)
    #define matmul_nn matmul_v1_nn
    #define generate_testcase generate_testcase_v1
    #define CASE0
#elif defined(V2)
    #define matmul_nn matmul_v2_nn
    #define generate_testcase generate_testcase_v2
    #define CASE1
#elif defined(V3)
    #define matmul_nn matmul_v3_nn
    #define generate_testcase generate_testcase_v3
    #define CASE1
#elif defined(V4)
    #define matmul_nn matmul_v4_nn
    #define generate_testcase generate_testcase_v4
    #define CASE2
#endif

#define REGISTER_KERNEL(dtype, BM, BN, BK, WM, WN) \
    register_kernel({BM, BN, BK, WM, WN}, matmul_wrapper<dtype, BM, BN, BK, WM, WN>);

#define REGISTER_FLOAT_KERNEL(BM, BN, BK, WM, WN) REGISTER_KERNEL(float, BM, BN, BK, WM, WN)


using KernelFunction = void (*)(float*, float*, float*, int, int, int, dim3, dim3, Major);
unordered_map<vector<int>, KernelFunction> kernel_map;

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
void matmul_wrapper(T* A, T* B, T* C, int M, int N, int K, dim3 grid, dim3 block, Major major);

void register_kernel(vector<int> config, KernelFunction func)
{
    assert(func != nullptr);
    kernel_map[config] = func;
}

KernelFunction query_kernel(vector<int> config)
{
    auto func = kernel_map[config];

    if (func == nullptr)
        assert(false);

    return func;
}

void generate_testcase_v0(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K)
{
    for (int T = 1024; T >= 64; T /= 2 ) {
        for(int i = 1; i <= 2; i *= 2) {
            if (N % (T / i) != 0) continue;

            testcases.push_back(
                make_pair<vector<int>, vector<int>>({i, T / i, 0, 0, 0}, {M, N, K})
            );
        }
    }
}

void generate_testcase_v1(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K)
{
    testcases.push_back(make_pair<vector<int>, vector<int>>({8, 8, 8, 0, 0}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({16, 16, 16, 0, 0}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({32, 32, 32, 0, 0}, {M, N, K}));
}

void generate_testcase_v2(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K)
{
    testcases.push_back(make_pair<vector<int>, vector<int>>({16, 16, 16, 2, 2}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({32, 32, 32, 2, 2}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({32, 32, 32, 4, 4}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({64, 64, 64, 2, 2}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({64, 64, 64, 4, 4}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({64, 64, 64, 8, 8}, {M, N, K}));
}

void generate_testcase_v3(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K)
{
    testcases.push_back(make_pair<vector<int>, vector<int>>({16, 16, 16, 2, 2}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({32, 32, 32, 2, 2}, {M, N, K}));
    testcases.push_back(make_pair<vector<int>, vector<int>>({32, 32, 32, 4, 4}, {M, N, K}));
}

void generate_testcase_v4(vector<pair<vector<int>, vector<int>>>& testcases, int M, int N, int K)
{
    testcases.push_back(make_pair<vector<int>, vector<int>>({128, 128, 8, 8, 8}, {M, N, K}));
}


void register_kernels()
{
#ifdef V0
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
#elif defined(V1)
    REGISTER_FLOAT_KERNEL(8, 8, 8, 0, 0);
    REGISTER_FLOAT_KERNEL(16, 16, 16, 0, 0);
    REGISTER_FLOAT_KERNEL(32, 32, 32, 0, 0);
#elif defined(V2)
    REGISTER_FLOAT_KERNEL(16, 16, 16, 2, 2);
    REGISTER_FLOAT_KERNEL(32, 32, 32, 2, 2);
    REGISTER_FLOAT_KERNEL(32, 32, 32, 4, 4);
    REGISTER_FLOAT_KERNEL(64, 64, 64, 2, 2);
    REGISTER_FLOAT_KERNEL(64, 64, 64, 4, 4);
    REGISTER_FLOAT_KERNEL(64, 64, 64, 8, 8);
#elif defined(V3)
    REGISTER_FLOAT_KERNEL(16, 16, 16, 2, 2);
    REGISTER_FLOAT_KERNEL(32, 32, 32, 2, 2);
    REGISTER_FLOAT_KERNEL(32, 32, 32, 4, 4);
#elif defined(V4)
    REGISTER_FLOAT_KERNEL(128, 128, 8, 8, 8);
#endif
}


vector<float*> gen_exp_cpu(float* x, float* y, vector<pair<vector<int>, vector<int>>>& testcases)
{
    vector<float*> results;
    unordered_map<vector<int>, float*> cache;
    char filename[256];

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

        sprintf(filename, "bin/matmul_C_%d_%d_%d_float32.bin", M, N, K);

        if (fileExists(string(filename))) {
            float* z = (float*)malloc(bytes_of<float>(M * N));
            size_t data_size = 0;
            load_from_file(filename, data_size, z);
            results.push_back(z);
            cache[key] = z;
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

        dump_to_file(filename, z, bytes_of<float>(M * N));
        results.push_back(z);
        cache[key] = z;
    }

    return results;
}

// https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_3.cu
// M, K @ K, N -> M, N  raw implementation
/*
256 x 256 x 256
best of 42.24 us    (128,2) (2, 128)
768 x 768 x 768
best of 0.9 ms      (768, 12) (1, 64)
1024 x 1024 x 1024
best of 2.13 ms     (512, 16), (2, 64)  (512, 8), (2, 128)  (512, 32), (2, 32)
2048 x 2048 x 2048
best of 16.97 ms    (1024, 16), (2, 128)  (1024, 64), (2, 32)
*/
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void matmul_v0_nn(T* __restrict__  A, T* __restrict__  B, T* __restrict__  C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * BLOCK_M + tx, y = by * BLOCK_N + ty;

    if (x >= M || y >= N) return;

    T sum = zero<T>();

    for(int i = 0; i < K; i++)
        sum = fmaf(ELEME_OF(A, x, i, K), ELEME_OF(B, i, y, N), sum);

    ELEME_OF(C, x, y, N) = sum;
}


// basic optimization method: block load matrix
// BLOCK_M = BLOCK_N = BLOCK_K
/*
256 x 256 x 256
best of 34.66 us    (8, 8), (32, 32)
768 x 768 x 768
best of 703.49 us   (48, 48), (16, 16)
1024 x 1024 x 1024
best of 1.66 ms     (64, 64), (16, 16)
2048 x 2048 x 2048
best of 12.99 ms    (128, 128), (16, 16)
*/
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void matmul_v1_nn(T* A, T* B, T* C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    T* AG = A + bx * BLOCK_M * K;
    T* BG = B + by * BLOCK_N;
    float sum = zero<T>();
    __shared__ T AS[BLOCK_M][BLOCK_K];
    __shared__ T BS[BLOCK_K][BLOCK_N];

    for(T* a_ptr = AG, *b_ptr = BG; a_ptr < AG + K; a_ptr += BLOCK_K, b_ptr += BLOCK_K * N) {
        // AS[ty][tx] = a_ptr[ty * K + tx];
        AS[ty][tx] = ELEME_OF(a_ptr, ty, tx, K);
        // BS[ty][tx] = b_ptr[ty * N + tx];
        BS[ty][tx] = ELEME_OF(b_ptr, ty, tx, N);
        __syncthreads();
#pragma unroll

        for (int k = 0; k < BLOCK_K; k++)
            sum = fmaf(AS[ty][k], BS[k][tx], sum);

        __syncthreads();
    }

    int x = bx * BLOCK_M + ty, y = by * BLOCK_N + tx;
    ELEME_OF(C, x, y, N) = sum;
}

// basic optimization: block optimize commute and store + share memory align  内存对齐在这一步加速了10%左右
// BLOCK_M = BLOCK_N = BLOCK_K THREAD_M = THREAD_N
/*
256 x 256 x 256
best of 18.50 us    (8, 8), (16, 16)     (2, 2)
768 x 768 x 768
best of 151.01 us   (12, 12), (16, 16)   (4, 4)
1024 x 1024 x 1024
best of 0.37 ms     (16, 16), (16, 16)   (4, 4)
2048 x 2048 x 2048
best of 3.08 ms    (32, 32), (16, 16)    (4, 4)
*/
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void matmul_v2_nn(T* A, T* B, T* C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    T* AG = A + bx * BLOCK_M * K;
    T* BG = B + by * BLOCK_N;
    T* CG = C + bx * BLOCK_M * N + by * BLOCK_N;
    T sum[THREAD_M][THREAD_N] = {zero<T>()};
    __shared__  __align__(BLOCK_M * BLOCK_K * sizeof(T)) T AS[BLOCK_M][BLOCK_K];
    __shared__  __align__(BLOCK_M * BLOCK_K * sizeof(T)) T BS[BLOCK_K][BLOCK_N];

    for(T* a_ptr = AG, *b_ptr = BG; a_ptr < AG + K; a_ptr += BLOCK_K, b_ptr += BLOCK_K * N) {
        // 16 x 16
        // load block matrix A & B to share
#pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            for(int j = 0; j < THREAD_N; j++) {
                // AS[ty * THREAD_M + i][tx * THREAD_M + j] = a_ptr[(ty * THREAD_M + i) * K + tx * THREAD_M + j];
                AS[ty * THREAD_M + i][tx * THREAD_M + j] = ELEME_OF(a_ptr, (ty * THREAD_M + i),  tx * THREAD_M + j, K);
                // BS[ty * THREAD_N + i][tx * THREAD_N + j] = b_ptr[(ty * THREAD_N + i) * N + tx * THREAD_N + j];
                BS[ty * THREAD_N + i][tx * THREAD_N + j] = ELEME_OF(b_ptr, (ty * THREAD_N + i), tx * THREAD_N + j, N);
            }
        }

        __syncthreads();
        // compute block THREAD_M * THREAD_N into register
#pragma unroll

        for (int i = 0; i < THREAD_M; i++) {
            for(int j = 0; j < THREAD_N; j++) {
                for (int k = 0; k < BLOCK_K; k++)
                    sum[i][j] = fmaf(AS[ty * THREAD_M + i][k], BS[k][tx * THREAD_N + j], sum[i][j]);
            }
        }

        __syncthreads();
    }

    for(int i = 0; i < THREAD_M; i++) {
        for(int j = 0; j < THREAD_N; j++)
            // CG[(ty * THREAD_M + i) * N + tx * THREAD_M + j] = sum[i][j];
            ELEME_OF(CG, (ty * THREAD_M + i), tx * THREAD_M + j, N) = sum[i][j];
    }
}

// __global__ void static_shared_alloc(float* x, float* y)
// {
//     __shared__ float test[12 * 1024]; // 静态申请的共享内存只能这么大 48K
//     int bx = blockIdx.x, tx = threadIdx.x;
//     test[tx * blockDim.x + bx] = x[bx * blockDim.x + tx] * 2;
//     y[bx * blockDim.x + tx] = test[bx * blockDim.x + tx];
// }

// using double buffer to optimize the kernel ? 只是增大了share memory，降低了并行度，反而引起了性能劣化
/*
256 x 256 x 256
best of 17.38 us    (8, 8), (16, 16)  (2, 2)
768 x 768 x 768
best of 245.47 us   (24, 24), (16, 16)  (2, 2)
1024 x 1024 x 1024
best of 0.59 ms     (32, 32), (16, 16) (2, 2)
2048 x 2048 x 2048
best of 4.53 ms    (64, 64), (16, 16)  (2, 2)
*/
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void matmul_v3_nn(T* A, T* B, T* C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    T* AG = A + bx * BLOCK_M * K;
    T* BG = B + by * BLOCK_N;
    T* CG = C + bx * BLOCK_M * N + by * BLOCK_N;
    T sum[THREAD_M][THREAD_N] = {zero<T>()};
    __shared__ __align__(BLOCK_M * BLOCK_K * sizeof(T)) float AS[BLOCK_M][BLOCK_K * 2];
    __shared__ __align__(BLOCK_M * BLOCK_K * sizeof(T)) float BS[2 * BLOCK_K][BLOCK_N];

    for(T* a_ptr = AG, *b_ptr = BG; a_ptr < AG + K; a_ptr += 2 * BLOCK_K, b_ptr += 2  * BLOCK_K * N) {
        for (int i = 0; i < THREAD_M; i++) {
            for(int j = 0; j < THREAD_N; j++) {
                AS[ty * THREAD_M + i][tx * THREAD_M + j] = ELEME_OF(a_ptr, (ty * THREAD_M + i),  tx * THREAD_M + j, K);
                AS[ty * THREAD_M + i][tx * THREAD_M + j + BLOCK_K] = ELEME_OF(a_ptr, (ty * THREAD_M + i),  tx * THREAD_M + j + BLOCK_K, K);
                BS[ty * THREAD_N + i][tx * THREAD_N + j] = ELEME_OF(b_ptr, (ty * THREAD_N + i), tx * THREAD_N + j, N);
                BS[ty * THREAD_N + i + BLOCK_K][tx * THREAD_N + j] = ELEME_OF(b_ptr, (ty * THREAD_N + i + BLOCK_K), tx * THREAD_N + j, N);
            }
        }

        __syncthreads();

        for (int i = 0; i < THREAD_M; i++) {
            for (int j = 0; j < THREAD_N; j++) {
                for (int k = 0; k < 2 * BLOCK_K; k++)
                    sum[i][j] = fmaf(AS[ty * THREAD_M + i][k], BS[k][tx * THREAD_N + j], sum[i][j]);
            }
        }

        __syncthreads();
    }

#pragma unroll

    for(int i = 0; i < THREAD_M; i++) {
        for(int j = 0; j < THREAD_N; j++)
            ELEME_OF(CG, (ty * THREAD_M + i), tx * THREAD_M + j, N) = sum[i][j];
    }
}

// gmem -> smem 使用interleave的方式加载 1个warp内32 threads 访问连续索引的32个float 总共访问128bytes，可以合并为一个transacation
/*
256 x 256 x 256
best of 109.54 us   (2, 2), (256)
768 x 768 x 768
best of 299.62 us   (6, 6), (256)
1024 x 1024 x 1024
best of 0.40 ms     (8, 8), (256)
2048 x 2048 x 2048
best of 2.25 ms     (16, 16), (256)
*/
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, const bool ENABLE_DOUBLE_BUFFER = false>
__global__ void matmul_v4_nn(T* A, T* B, T* C, int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
    // 从A 加载 128, 8 到AS的是 8, 128
    __shared__ T __align__(BLOCK_M * BLOCK_K * sizeof(T)) ashare[BLOCK_K * BLOCK_M];
    // 从B 加载 8, 128 到BS的是 8, 128
    __shared__ T __align__(BLOCK_N * BLOCK_K * sizeof(T)) bshare[BLOCK_K * BLOCK_N];
    float sum[8][8] = {0};
    float panelA[8] = {0}, panelB[8] = {0};
    int from_a = (by * BLOCK_M + tx / BLOCK_K * 4) * K + tx % BLOCK_K;
    // load b 最小使用了一个warp size, 保证可以合并访存 32 * 4 = 128bytes  1个transaction可以加载完。
    int from_b = (tx / WARP_SIZE) * N + bx * BLOCK_N + tx % WARP_SIZE;
    int load_a_cnt = BLOCK_M * BLOCK_K / blockDim.x;
    int load_b_cnt = BLOCK_N * BLOCK_K / blockDim.x;

    for (int k = 0; k < K; k += BLOCK_K) {
        // part1: gmem to smem
        int to_a = (tx % BLOCK_K) * BLOCK_M + (tx / BLOCK_K) * 4;  // and doing transpose store

        // 这里需要循环展开，循环展开能有20% 性能提升
        for (int i = 0; i < load_a_cnt; ++i)  // 这里存在8-way bank 冲突load_a_cnt = 8, 0-7 8-15 16-23 24-31 都写在一个bank 里
            ashare[to_a + i] = A[from_a + i * K];


        int to_b = (tx / WARP_SIZE) * BLOCK_N + (tx % WARP_SIZE);

        // 这里需要循环展开，循环展开能有20% 性能提升
        for (int i = 0; i < load_b_cnt; ++i) // 这里没有bank 冲突
            bshare[to_b + i * WARP_SIZE] = B[from_b + i * WARP_SIZE]; // 32 thread 合并访问。 thread i 访问  [i, i+32, i+64, i+96]

        __syncthreads();
        from_a += BLOCK_K;
        from_b += BLOCK_K * N;
        // part2: calculation
        // 计算 2x2 个 4x4  8x128 AS 8x128 BS compute
        int aidx0 = (tx / 16) * 4;
        int bidx0 = (tx % 16) * 4;

        for (int subk = 0; subk < BLOCK_K; ++subk) {
            float* ptrA = ashare + aidx0 + subk * BLOCK_M;

            for (int i = 0; i < 4; ++i) {
                panelA[i] = ptrA[i];
                panelA[i + 4] = ptrA[i + 64];
            }

            const float* ptrB = bshare + bidx0 + subk * BLOCK_N;

            for (int i = 0; i < 4; ++i) {
                panelB[i] = ptrB[i];
                panelB[i + 4] = ptrB[i + 64];
            }

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j)
                    sum[i][j] += panelA[i] * panelB[j];
            }
        }

        __syncthreads();
    }

    // part3: save to C
    int write_offset = (by * 128 + (tx / 16) * 4) * N + bx * 128 + (tx % 16) * 4;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[write_offset + i * N + j] = sum[i][j];
            C[write_offset + i * N + j + 64] = sum[i][j + 4];
            C[write_offset + (i + 64) * N + j] = sum[i + 4][j];
            C[write_offset + (i + 64) * N + j + 64] = sum[i + 4][j + 4];
        }
    }
}


template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
void matmul_wrapper(T* A, T* B, T* C, int M, int N, int K, dim3 grid, dim3 block, Major major)
{
    if (major == RowMajor)
        matmul_nn<T, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N> <<< grid, block>>>(A, B, C, M, N, K);
    else if (major == ColMajor)
        assert(false);
}


int main(int argc, char** argv)
{
    register_kernels();
    srand(SEED);
    const int MAX_M = 4096, MAX_N = 4096, MAX_K = 4096;
    vector<pair<vector<int>, vector<int>>> testcases = {
    };
    generate_testcase(testcases, 256, 256, 256);
    generate_testcase(testcases, 768, 768, 768);
    generate_testcase(testcases, 1024, 1024, 1024);
    generate_testcase(testcases, 2048, 2048, 2048);
    REGISTER_BUFF(A, bytes_of<float>(MAX_M * MAX_K));
    REGISTER_BUFF(B, bytes_of<float>(MAX_N * MAX_K));
    REGISTER_BUFF(C, bytes_of<float>(MAX_M * MAX_N));
    genOrLoad<float>(string("bin/matmul_A_4096_4096_float32.bin"), h_A, MAX_M * MAX_K);
    genOrLoad<float>(string("bin/matmul_B_4096_4096_float32.bin"), h_B, MAX_M * MAX_K);
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
        int block_k = testcase.first[2];
        int thread_x = testcase.first[3];
        int thread_y = testcase.first[4];
        int grid_x = (M + block_x - 1) / block_x;
        int grid_y = (N + block_y - 1) / block_y;
#ifdef CASE0
        dim3 block(block_x, block_y);
#elif defined(CASE1)
        dim3 block(block_x / thread_x, block_y / thread_y);
#elif defined(CASE2)
        dim3 block(256);
#endif
        dim3 grid(grid_x, grid_y);
        cout << "running test M = " << M << " M = " << N << " K = " << K << " grid (" << grid.x << ", " << grid.y << ")"
            << " block (" << block.x << ", " << block.y << ")" << endl;
        auto kernel_wrapper = query_kernel(testcase.first);
        // static_shared_alloc<<<grid, block>>>((float*)d_A, (float*)d_B);
        kernel_wrapper((float*)d_A, (float*)d_B, (float*)d_C, M, N, K, grid, block, RowMajor);
        CHECK_CALL_ERROR(cudaMemcpy(h_C, d_C, bytes_of<float>(M * N), cudaMemcpyDeviceToHost));
        cout << count_large_diff((float*)h_C, expects[i], M * N, 1e-3, 1e-3) << "/" << M* N << " differs." << endl;
    }

    FREE_BUFF(A);
    FREE_BUFF(B);
    FREE_BUFF(C);
}