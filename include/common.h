#include <stdio.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <vector_functions.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <stdint.h>
#include <unordered_map>
#include <utility>
#include <map>
#include <cfloat>
#include <ctime>
#include <cmath>
#include <string>
#include <fstream>
#include <sys/stat.h>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;
using std::srand;
using std::rand;
using std::is_same;
using std::min;
using std::max;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::map;
using std::time;
using std::fabs;
using std::string;
using std::to_string;
using std::fixed;
using std::setprecision;
using std::ifstream;
using std::ofstream;
using std::enable_if;

const unsigned int SEED = 42;

const int max_block_threads = 1024;
const int A10_SM_CNT = 72;
const int A10_SP_CNT = 128;

enum Major {
    RowMajor = 0,
    ColMajor = 0,
};

#define WARP_SIZE 32
#define OF_DEVICE_FUNCTION __device__ __host__ __forceinline__
#define DEVICE_FUNCTION __device__ __forceinline__
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA_ERROR() \
    do { \
        cudaError_t result = cudaDeviceSynchronize(); \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CALL_ERROR(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define PRO_BEG   \
    {   \
        auto start = std::chrono::high_resolution_clock::now(); \

#define PRO_END \
    auto end = std::chrono::high_resolution_clock::now();\
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
    double elapsed = duration.count() / 1000.0;  \
    cout << "Elapsed time: " << fixed << setprecision(4) << elapsed << " milliseconds" << endl;  \
    }

#define EVT_BEG(repeat_time) \
    {   \
        cudaEvent_t start, stop;    \
        cudaEventCreate(&start);    \
        cudaEventCreate(&stop);     \
        cudaEventRecord(start, 0);      \
        for(int repeat_cnt = 0; repeat_cnt < repeat_time; repeat_cnt++) {

#define EVT_END(repeat_time) \
    }                           \
    cudaEventRecord(stop, 0);   \
    cudaEventSynchronize(stop); \
    float milliseconds = 0.0f;  \
    cudaEventElapsedTime(&milliseconds, start, stop);   \
    cout << "repeat " << repeat_time << " times, total time " << milliseconds   \
        << " avg time " << fixed << setprecision(4) << milliseconds / repeat_time << " milliseconds" << endl;  \
    }

#define REGISTER_BUFF(name, bytes)                  \
    void* h_##name = (void*)malloc(bytes);          \
    void* d_##name = nullptr;                       \
    CHECK_CALL_ERROR(cudaMalloc((void**)&d_##name, bytes))

#define FREE_BUFF(name)     \
    free(h_##name);         \
    CHECK_CALL_ERROR(cudaFree(d_##name));

#define MEMSET_BUF(name, bytes)                         \
    memset(h_##name, 0, bytes);                         \
    CHECK_CALL_ERROR(cudaMemset(d_##name, 0, bytes));

#define ELEME_OF(p, x, y, s) (p[(x) * (s) + (y)])

template<typename T>
size_t bytes_of()
{
    return sizeof(T);
}

template <typename T, typename... Args>
size_t bytes_of(size_t num, Args... args)
{
    return num * bytes_of<T>(args...);
}

size_t num_of_elems()
{
    return 1;
}

template<typename... Args>
size_t num_of_elems(size_t num, Args... args)
{
    return num *  bytes_of(args...);
}

enum DTYPE {
    DOUBLE = 0,
    FLOTAT64 = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    HALF = 2,
    INT64 = 3,
    INT32 = 4,
    INT8 = 5,
    UINT64 = 6,
    UINT32 = 7,
    UINT8 = 8,
    UNK_TYPE
};

template<typename T>
DTYPE checkType()
{
    if (is_same<T, double>::value)
        return DOUBLE;
    else if (is_same<T, float>::value)
        return FLOAT32;
    else if (is_same<T, half>::value)
        return HALF;
    else if (is_same<T, int64_t>::value)
        return INT64;
    else if (is_same<T, int32_t>::value)
        return INT32;
    else if (is_same<T, int8_t>::value)
        return INT8;
    else if (is_same<T, uint64_t>::value)
        return UINT64;
    else if (is_same<T, uint32_t>::value)
        return UINT32;
    else if (is_same<T, uint8_t>::value)
        return UINT8;
    else
        return UNK_TYPE;
}



template<typename T>
T random_value()
{
    DTYPE type = checkType<T>();

    switch (type) {
        case DOUBLE: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case FLOAT32: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case HALF: {
            T rand_val = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
            return rand_val;
        }

        case INT64: return 0;

        case INT32: return 0;

        case INT8: return 0;

        case UINT64: return 0;

        case UINT32: return 0;

        case UINT8: return 0;

        default:
            return 0;
    }
}

template<typename T>
void gen_random(T* data, size_t n_elems)
{
    assert(data != nullptr);

    while (n_elems--)
        data[n_elems] = random_value<T>();
}


inline pair<int, int> query_capability(int device = 0)
{
    CHECK_CALL_ERROR(cudaSetDevice(device));
    int major = 0;
    int minor = 0;
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return {major, minor};
}


inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
    int* num_blocks)
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
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

// inline pair<int, int> query_mp_sp(int device = 0)
// {
//     int smCount = 0;
//     auto capability = query_capability(device);
//     CHECK_CALL_ERROR(cudaSetDevice(device));
//     CHECK_CALL_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
//     int spPerSM = _ConvertSMVer2Cores(capability.first, capability.second);
//     return {smCount, spPerSM};
// }
/*
total_bandwidth_per_sm = memory_clock_rate * global_memory_bus_width / 8
max_blocks_per_sm = max_threads_per_multi_processor / max_threads_per_block
average_bandwidth_per_sm = total_bandwidth_per_sm / max_blocks_per_sm
total_bandwidth = average_bandwidth_per_sm * multi_processor_count
6251000 * 384 / 8  * 1024 / 1536 * 72 = 13.4 测试最大 h2d 12.3 d2h 13.2

compute capability:                    8.6
max_threads_per_block:                 1024
max_shared_memory_per_block:           49152
warp_size:                             32
max_registers_per_block:               65536
clock_rate:                            1695000
multi_processor_count:                 72
memory_clock_rate:                     6251000
global_memory_bus_width:               384
l2_cache_size:                         6291456
max_threads_per_multi_processor:       1536
max_shared_memory_per_multi_processor: 102400
max_registers_per_multi_processor:     65536
max_blocks_per_multi_processor:        16

 Device 0: NVIDIA A10
 Quick Mode
 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     12.3

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     13.2

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     471.0
*/
inline void query_device(int device = 0)
{
    CHECK_CALL_ERROR(cudaSetDevice(device));
    int major = 0;
    int minor = 0;
    int max_threads_per_block = 0;
    int max_shared_memory_per_block = 0;
    int warp_size = 0;
    int max_registers_per_block = 0;
    int clock_rate = 0;
    int multi_processor_count = 0;
    int memory_clock_rate = 0;
    int global_memory_bus_width = 0;
    int l2_cache_size = 0;
    int max_threads_per_multi_processor = 0;
    int max_shared_memory_per_multi_processor = 0;
    int max_registers_per_multi_processor = 0;
    int max_blocks_per_multi_processor = 0;
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_registers_per_block, cudaDevAttrMaxRegistersPerBlock, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&memory_clock_rate, cudaDevAttrMemoryClockRate, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&global_memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_threads_per_multi_processor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_shared_memory_per_multi_processor, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
            device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_registers_per_multi_processor, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&max_blocks_per_multi_processor, cudaDevAttrMaxBlocksPerMultiprocessor, device));
    cout << "compute capability:                    " << major << "." << minor << endl;
    cout << "max_threads_per_block:                 " << max_threads_per_block << endl;
    cout << "max_shared_memory_per_block:           " << max_shared_memory_per_block << endl;
    cout << "warp_size:                             " << warp_size << endl;
    cout << "max_registers_per_block:               " << max_registers_per_block << endl;
    cout << "clock_rate:                            " << clock_rate << endl;
    cout << "multi_processor_count:                 " << multi_processor_count << endl;
    cout << "memory_clock_rate:                     " << memory_clock_rate << endl;
    cout << "global_memory_bus_width:               " << global_memory_bus_width << endl;
    cout << "l2_cache_size:                         " << l2_cache_size << endl;
    cout << "max_threads_per_multi_processor:       " << max_threads_per_multi_processor << endl;
    cout << "max_shared_memory_per_multi_processor: " << max_shared_memory_per_multi_processor << endl;
    cout << "max_registers_per_multi_processor:     " << max_registers_per_multi_processor << endl;
    cout << "max_blocks_per_multi_processor:        " << max_blocks_per_multi_processor << endl;
}


template<typename T>
struct UnaryFunctor {
    virtual DEVICE_FUNCTION T operator()(T x) const = 0;
};

template<typename T>
struct SquareFunctor : public UnaryFunctor<T> {
    virtual DEVICE_FUNCTION T operator()(T x) const override
    {
        return x * x;
    }
};

template <>
DEVICE_FUNCTION half SquareFunctor<half>::operator()(half x) const
{
    return __hmul(x, x);
}

template <>
DEVICE_FUNCTION half2 SquareFunctor<half2>::operator()(half2 x) const
{
    return __hmul2_rn(x, x);
}

template<typename T>
struct BinarayFunctor {
    virtual DEVICE_FUNCTION T operator()(T x, T y) const = 0;
};

template<typename T>
struct TenarayFunctor {
    virtual DEVICE_FUNCTION T operator()(T x, T y, T z) const = 0;
};

size_t count_large_diff(const float* a, const float* b, size_t n, float atol, float rtol, int n_print = 8, int row_num = 8)
{
    size_t count = 0;

    for (size_t i = 0; i < n; ++i) {
        float diff = fabs(a[i] - b[i]);
        float abs_tol = fabs(atol);
        float rel_tol = fabs(rtol);

        // cout << a[i] << " vs " << b[i] << ", ";
        if (diff > abs_tol && diff > max(fabs(a[i]), fabs(b[i])) * rel_tol)
            count++;
    }

    for(int i = 0; i < n_print; i++) {
        printf("%8.2f", a[i]);

        if ((i + 1) % row_num == 0)
            printf("\n");
    }

    printf("\n");
    // cout << endl;
    return count;
}

bool fileExists(const string& filename)
{
    ifstream file(filename.c_str());
    return file.good();
}


void dump_to_file(const char* filename, const void* data, size_t size)
{
    // 打开二进制写文件
    ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // 将数据写入文件
    file.write(reinterpret_cast<const char*>(data), size);
    // 关闭文件
    file.close();

    if (file.fail())
        cerr << "Error writing to file: " << filename << endl;
    else
        cout << "Data dumped to file: " << filename << endl;
}

void* load_from_file(const char* filename, size_t& dataSize, void* outputPtr = nullptr)
{
    ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return nullptr;
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    dataSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 分配内存，如果输出指针为 nullptr
    if (outputPtr == nullptr)
        outputPtr = new char[dataSize];

    // 读取数据
    file.read(reinterpret_cast<char*>(outputPtr), dataSize);
    // 关闭文件
    file.close();

    if (file.fail()) {
        cerr << "Error reading from file: " << filename << endl;

        // 释放内存，如果输出指针为 nullptr
        if (outputPtr != nullptr) {
            delete[] reinterpret_cast<char*>(outputPtr);
            outputPtr = nullptr;
        }
    } else
        cout << "Data loaded from file: " << filename << endl;

    return outputPtr;
}

template<typename T>
void genOrLoad(const string& filename, void* ptr, size_t n_elems)
{
    if (fileExists(filename)) {
        size_t dataSize = 0;
        load_from_file(filename.c_str(), dataSize, ptr);
    } else
        gen_random<T>((T*)ptr, n_elems);
}

template <typename T>
DEVICE_FUNCTION T zero();

template <>
DEVICE_FUNCTION half2 zero<half2>()
{
    return make_half2(0.0f, 0.0f);
}

template <>
DEVICE_FUNCTION half zero<half>()
{
    return __float2half(0.0f);
}

template <>
DEVICE_FUNCTION float4 zero<float4>()
{
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template <>
DEVICE_FUNCTION float2 zero<float2>()
{
    return make_float2(0.0f, 0.0f);
}

template <>
DEVICE_FUNCTION float zero<float>()
{
    return 0.0f;
}

template <>
DEVICE_FUNCTION int zero<int>()
{
    return 0;
}

template <>
DEVICE_FUNCTION int8_t zero<int8_t>()
{
    return 0;
}

namespace std
{
template <>
struct hash<std::vector<int>> {
    size_t operator()(const std::vector<int>& v) const
    {
        size_t hash = 0;

        for (int i : v) {
            hash ^= std::hash<int> {}(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        return hash;
    }
};
}