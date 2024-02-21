#include "common.h"

void benchmark_host_device_mem_bandwidth()
{
    // alloc 10g for testing
    const size_t max_bytes = size_t(1024) * 1024 * 1024 * 8;
    const size_t benchmark_bytes[] = {
        1024,
        1024 * 512,
        1024 * 1024,
        1024 * 1024 * 512,
        1024 * 1024 * 1024,
        size_t(1024) * 1024 * 1024 * 2,
        size_t(1024) * 1024 * 1024 * 4,
        size_t(1024) * 1024 * 1024 * 8
    };
    char *h_data = new char[max_bytes];
    char *d_data = nullptr;
    cudaMalloc((void**)&d_data, max_bytes);
    std::cout << "=============================> benchmark host device copy bandwidth <=============================" << std::endl;
    for (size_t benchmark_byte: benchmark_bytes) {
        double h2dtransferRate = 0.0f;
        double d2htransferRate = 0.0f;
        {
            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d_data, h_data, benchmark_byte, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            h2dtransferRate = static_cast<double>(benchmark_byte) / (duration.count() * 1024 * 1024); // MB/s
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(h_data, d_data, benchmark_byte, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            d2htransferRate = static_cast<double>(benchmark_byte) / (duration.count() * 1024 * 1024); // MB/s
        }
        std::cout << "Copy " << (double)benchmark_byte / 1024 / 1024 << " MB Host to Device Transfer Rate: " << h2dtransferRate << " MB/s" << std::endl;
        std::cout << "Copy " << (double)benchmark_byte / 1024 / 1024 << " MB Device to Host Transfer Rate: " << d2htransferRate << " MB/s" << std::endl;
    }
    std::endl;
    delete[] h_data;
    cudaFree(d_data);
}
/*
mixbench (v0.04-9-g858bbdd)
------------------------ Device specifications ------------------------
Device:              NVIDIA A10
CUDA driver version: 12.20
GPU clock rate:      1695 MHz
Memory clock rate:   3125 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       6144 KB
Total global mem:    22515 MB
ECC enabled:         Yes
Compute Capability:  8.6
Total SPs:           9216 (72 MPs x 128 SPs/MP)
Compute throughput:  31242.24 GFlops (theoretical single precision FMAs)
Memory bandwidth:    600.10 GB/sec
-----------------------------------------------------------------------
Total GPU memory 23609475072, free 23359193088
Buffer size:          256MB
Trade-off type:       compute with global memory (block strided)
Elements per thread:  8
Thread fusion degree: 4
----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------
Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,, 
Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec
            0,      0.250,    0.28,  119.59, 478.36,      0.125,    0.54,   62.53, 500.27,      0.500,    0.27,  244.54, 489.07,     0.250,    0.27,  123.19, 492.75
            1,      0.750,    0.27,  366.81, 489.07,      0.375,    0.54,  187.25, 499.32,      1.500,    0.27,  736.36, 490.91,     0.750,    0.27,  369.56, 492.75
            2,      1.250,    0.27,  611.34, 489.07,      0.625,    0.54,  307.97, 492.75,      2.500,    0.27, 1222.69, 489.07,     1.250,    0.27,  613.63, 490.91
            3,      1.750,    0.27,  855.88, 489.07,      0.875,    0.61,  388.12, 443.56,      3.500,    0.27, 1724.63, 492.75,     1.750,    0.28,  852.70, 487.26
            4,      2.250,    0.27, 1104.54, 490.91,      1.125,    0.76,  394.80, 350.93,      4.500,    0.27, 2200.84, 489.07,     2.250,    0.27, 1104.54, 490.91
            5,      2.750,    0.28, 1339.96, 487.26,      1.375,    0.93,  396.10, 288.07,      5.500,    0.28, 2679.91, 487.26,     2.750,    0.27, 1355.07, 492.75
            6,      3.250,    0.28, 1583.58, 487.26,      1.625,    1.09,  398.86, 245.45,      6.500,    0.28, 3167.17, 487.26,     3.250,    0.27, 1607.49, 494.61
            7,      3.750,    0.28, 1827.21, 487.26,      1.875,    1.26,  399.61, 213.13,      7.500,    0.28, 3654.42, 487.26,     3.750,    0.27, 1861.82, 496.48
            8,      4.250,    0.28, 2063.17, 485.45,      2.125,    1.42,  401.34, 188.86,      8.500,    0.28, 4126.34, 485.45,     4.250,    0.27, 2102.10, 494.61
            9,      4.750,    0.28, 2305.90, 485.45,      2.375,    1.59,  402.19, 169.34,      9.500,    0.27, 4646.21, 489.07,     4.750,    0.27, 2349.40, 494.61
           10,      5.250,    0.27, 2577.26, 490.91,      2.625,    1.74,  403.83, 153.84,     10.500,    0.28, 5097.24, 485.45,     5.250,    0.27, 2596.71, 494.61
           11,      5.750,    0.27, 2822.71, 490.91,      2.875,    1.91,  403.68, 140.41,     11.500,    0.28, 5562.10, 483.66,     5.750,    0.27, 2844.01, 494.61
           12,      6.250,    0.28, 2957.40, 473.18,      3.125,    2.07,  404.74, 129.52,     12.500,    0.28, 6068.15, 485.45,     6.250,    0.27, 3103.03, 496.48
           13,      6.750,    0.27, 3301.25, 489.07,      3.375,    2.24,  404.73, 119.92,     13.500,    0.28, 6505.41, 481.88,     6.750,    0.27, 3326.08, 492.75
           14,      7.250,    0.27, 3559.49, 490.96,      3.625,    2.40,  405.41, 111.84,     14.500,    0.28, 6987.29, 481.88,     7.250,    0.27, 3585.93, 494.61
           15,      7.750,    0.27, 3790.33, 489.07,      3.875,    2.56,  405.67, 104.69,     15.500,    0.28, 7496.74, 483.66,     7.750,    0.27, 3833.24, 494.61
           16,      8.250,    0.27, 4034.87, 489.07,      4.125,    2.73,  405.91,  98.40,     16.500,    0.28, 7921.93, 480.12,     8.250,    0.27, 4065.20, 492.75
           17,      8.750,    0.28, 4263.49, 487.26,      4.375,    2.89,  405.83,  92.76,     17.500,    0.28, 8340.95, 476.63,     8.750,    0.27, 4311.58, 492.75
           18,      9.250,    0.28, 4490.43, 485.45,      4.625,    3.05,  406.44,  87.88,     18.500,    0.28, 8849.75, 478.36,     9.250,    0.27, 4557.96, 492.75
           20,     10.250,    0.28, 4994.38, 487.26,      5.125,    3.38,  406.75,  79.37,     20.500,    0.28, 9665.38, 471.48,    10.250,    0.27, 5013.01, 489.07
           22,     11.250,    0.28, 5481.64, 487.26,      5.625,    3.71,  407.00,  72.36,     22.500,    0.28,10646.64, 473.18,    11.250,    0.27, 5522.70, 490.91
           24,     12.250,    0.28, 5968.89, 487.26,      6.125,    4.04,  407.31,  66.50,     24.500,    0.28,11593.01, 473.18,    12.250,    0.28, 5946.79, 485.45
           28,     14.250,    0.28, 6917.69, 485.45,      7.125,    4.69,  407.54,  57.20,     28.500,    0.29,13153.35, 461.52,    14.250,    0.28, 6892.16, 483.66
           32,     16.250,    0.27, 8098.56, 498.37,      8.125,    5.35,  407.95,  50.21,     32.500,    0.29,14894.55, 458.29,    16.250,    0.28, 7773.43, 478.36
           40,     20.250,    0.28, 9686.89, 478.36,     10.125,    6.66,  408.28,  40.32,     40.500,    0.30,18368.22, 453.54,    20.250,    0.29, 9412.09, 464.79
           48,     24.250,    0.28,11516.29, 474.90,     12.125,    7.96,  408.76,  33.71,     48.500,    0.30,21476.32, 442.81,    24.250,    0.30,10922.67, 450.42
           56,     28.250,    0.28,13319.37, 471.48,     14.125,    9.27,  408.83,  28.94,     56.500,    0.31,24850.90, 439.84,    28.250,    0.32,11944.46, 422.81
           64,     32.250,    0.29,15096.69, 468.11,     16.125,   10.58,  409.12,  25.37,     64.500,    0.34,25387.82, 393.61,    32.250,    0.35,12396.11, 384.38
           80,     40.250,    0.29,18774.55, 466.45,     20.125,   13.20,  409.31,  20.34,     80.500,    0.40,26848.08, 333.52,    40.250,    0.43,12681.85, 315.08
           96,     48.250,    0.29,22190.26, 459.90,     24.125,   15.82,  409.44,  16.97,     96.500,    0.48,26911.59, 278.88,    48.250,    0.50,12828.04, 265.87
          128,     64.250,    0.35,24915.31, 387.79,     32.125,   21.06,  409.54,  12.75,    128.500,    0.63,27386.59, 213.13,    64.250,    0.66,13016.04, 202.58
          192,     96.250,    0.48,27130.50, 281.88,     48.125,   31.53,  409.75,   8.51,    192.500,    0.93,27726.77, 144.04,    96.250,    0.98,13182.53, 136.96
          256,    128.250,    0.62,27877.25, 217.37,     64.125,   41.99,  409.92,   6.39,    256.500,    1.19,28883.13, 112.60,   128.250,    1.30,13236.21, 103.21
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA A10"
  CUDA Driver Version / Runtime Version          12.2 / 11.8
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 22516 MBytes (23609475072 bytes)
  (072) Multiprocessors, (128) CUDA Cores/MP:    9216 CUDA Cores
  GPU Max Clock rate:                            1695 MHz (1.70 GHz)
  Memory Clock rate:                             6251 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 6291456 bytes  6M
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes  100KB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.2, CUDA Runtime Version = 11.8, NumDevs = 1

*/


int main()
{
    benchmark_host_device_mem_bandwidth();
    return 0;
}
