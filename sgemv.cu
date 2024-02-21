#include "common.h"

enum Major {
    RowMajor = 0,
    ColMajor = 0,
};


pair<vector<int>, vector<int>> calculate_grid_block(int m, int n, int k)
{
    int grid = 1;
    int block = 32;
    return {{m, n, k}, {grid, block}};
}


template<typename T>
void sgemv_wrapper(T* x, T* y, T* z, int M, int N, int K, dim3 grid, dim3 block, Major major)
{
    if (major == RowMajor) {

    } else if (major == ColMajor) {

    }
}

using KernelFunction = void (*)(float*, float*, float*, int, int, int, dim3, dim3);

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
        cout << "kernel not found M = " << config[0] << " N = " << config[1] << " K = " << config[2] << endl;
        assert(false);
    }

    return func;
}

vector<float*> gen_exp_cpu(float* x, float* y, vector<pair<vector<int>, vector<int>>>& testcases)
{
    vector<float*> results;

    for (auto testcase : testcases) {
        int M = testcase.first[0];
        int N = testcase.first[1];
        int K = testcase.first[2];
        cout << "running case M " << M << " N " << N << " K " << K << endl;
        float* z = (float*)malloc(bytes_of<float>(M * N));

        for (int i = 0; i < M; i ++ ) {
            for (int j = 0; j < N; j ++) {
                float val = 0;

                for(int k = 0; k < K; k++) {
                    printf("x[%d * %d + %d] * y[%d * %d + %d] = %f * %f\n", i, K, k, k, N, j, x[i * K + k], y[k * N + j]);
                    val += x[i * K + k] * y[k * N + j];
                }

                z[i * N + j] = val;
            }
        }

        results.push_back(z);
    }

    return results;
}



void register_kernels()
{
}

/*
0.0 0.1     0.0 0.1   = 0.02 0.03
0.2 0.3     0.2 0.3     0.06 0.11
0.4 0.5                 0.1  0.19
*/


int main(int argc, char** argv)
{
    register_kernels();
    const int MAX_M = 1, MAX_N = 2, MAX_K = 2;
    vector<pair<vector<int>, vector<int>>> testcases = {
        calculate_grid_block(1, 2, 2),
    };
    REGISTER_BUFF(x, bytes_of<float>(MAX_M * MAX_K));
    REGISTER_BUFF(y, bytes_of<float>(MAX_N * MAX_K));
    REGISTER_BUFF(z, bytes_of<float>(MAX_M * MAX_N));
    gen_random<float>((float*)h_x, MAX_M * MAX_K);
    gen_random<float>((float*)h_y, MAX_N * MAX_K);
    // for (int i = 0; i < MAX_M * MAX_K; i++)
    //     ((float*)h_x)[i] = 0.1 * i;
    // for (int i = 0; i < MAX_K * MAX_N; i++)
    //     ((float*)h_y)[i] = 0.1 * i;
    auto expects = gen_exp_cpu((float*)h_x, (float*)h_y, testcases);

    for (int i = 0; i < testcases.size(); i++) {
        auto testcase = testcases[i];
        int M = testcase.first[0];
        int N = testcase.first[1];
        int K = testcase.first[2];
        dim3 grid(testcase.second[0]);
        dim3 block(testcase.second[1]);
        cout << "running test M = " << M << " M = " << N << " K = " << K << " grid " << grid.x << " block " << block.x << endl;
        auto kernel = query_kernel(testcase.first);

        cout << count_large_diff((float*)h_z, expects[i], M * N, 1e-3, 1e-3) << "/" << M * N << " differs." << endl;
    }

    FREE_BUFF(x);
    FREE_BUFF(y);
    FREE_BUFF(z);
}