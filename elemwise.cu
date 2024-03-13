#include "common.h"


constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);

        if (err != cudaSuccess)  return err;
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

        if (err != cudaSuccess)  return err;
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

        if (err != cudaSuccess)  return err;
    }

    int maxBlocksPerSM;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, dev);

    std::cout << "Maximum blocks per SM: " << maxBlocksPerSM << " tpm: " << tpm << " sm_count "  << sm_count << std::endl;
    cout << sm_count << " " << tpm << endl;
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}




// void gen_elemwise_expect_cpu(float* x, float* y, )
// {
// }


int main(int argc, char** argv)
{
    const size_t max_elem_cnts = 1024 * 1024 * 32;
    // vector<pair<size_t, pair<int, int>>> test_cases = {
    //     calculate_grid_block(2048),
    //     calculate_grid_block(4096),
    //     calculate_grid_block(8192),
    //     calculate_grid_block(16384),
    //     calculate_grid_block(1024 * 512),
    //     calculate_grid_block(1024 * 1024),
    //     calculate_grid_block(1024 * 1024 * 32)
    // };
    int n = 1024, k = 1;
    CHECK_CALL_ERROR(GetNumBlocks(n, &k));
    cout << n << " " << k << endl;
    exit(0);
    // void* data = malloc(bytes_of<float>(max_elem_cnts));
    // void* exp = malloc(bytes_of<float>(max_elem_cnts));
    // gen_random<float>((float*)h_data, max_elem_cnts);
    // REGISTER_BUFF(x, bytes_of<float>(max_elem_cnts));
    // REGISTER_BUFF(y, bytes_of<float>(max_elem_cnts));
    // CHECK_CALL_ERROR(cudaMemcpy(d_x, h_x, max_elem_cnts, cudaMemcpyHostToDevice));
    // FREE_BUFF(x);
    // FREE_BUFF(y);
}