#include <stdio.h>
#include <cuda_runtime.h>
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

const unsigned int SEED = 42;

const int WARP_SIZE = 32;
const int max_block_threads = 1024;
const int A10_SM_CNT = 72;
const int A10_SP_CNT = 128;

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

inline int _ConvertSMVer2Cores(int major, int minor)
{
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}
    };
    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[index].Cores;

        index++;
    }

    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
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

inline pair<int, int> query_mp_sp(int device = 0)
{
    int smCount = 0;
    auto capability = query_capability(device);
    CHECK_CALL_ERROR(cudaSetDevice(device));
    CHECK_CALL_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    int spPerSM = _ConvertSMVer2Cores(capability.first, capability.second);
    return {smCount, spPerSM};
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

size_t count_large_diff(const float* a, const float* b, size_t n, float atol, float rtol)
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

    // cout << endl;
    return count;
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