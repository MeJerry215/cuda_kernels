# Optimize


## Basic

测试加载以interleave, 同时混合使用vectorize的性能。测试代码 `cache_hit.cu`,
函数`load1_interleave`、`load2_interleave`、`load4_interleave`、`load8_interleave`、`load16_interleave`

搬运 256MB u8的数据从 global->register->global

load1_interleave:   2.15ms  u8
load2_interleave:   1.30ms  half
load4_interleave:   1.10ms  float
load8_interleave:   1.09ms  float2
load16_interleave:  1.09ms  float4

基础case load1_interleave u8 interleave方式的加载，一个线程加载一个1byte数据并写回1byte的数据，warp加载也就是32bytes数据。而night-compute 给出的的优化建议是
L2 Load Access Pattern && L2 Store Access Pattern : The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only accesses an average of 1.7 sectors out of the possible 4 sectors per cache line. Check the  Source Counters section for uncoalesced loads and try to minimize how many cache lines need to be accessed per memory request.


大概得意思是每个L2 request访问是128bytes cache line，也就是4个连续的32bytes sectors = 128bytes，而实际上值访存只占用了1.7个 sectors.

而load4_interleave 是没有这个优化建议的，所以理想情况下一个线程一个访问到至少一个float，这个warp最好能够访问连续的32个float数据值，也就是所说的连续的128bytes sectors。

而load8 load16 并没有多大的提升 因为已经是最优的访问方式。



### Memory
### global -> register -> global

1. 搬运的指令在底层只和数据类型的字节数有关 uint8(LDG.E.U8 STG.E.U8) half(LDG.E.U16 STG.E.U16) float(LDG.E STG.E) float2(LDG.E.64 STG.E.64) float4(LDG.E.128 STG.E.128)
2. 1个线程单次至少搬运1byte，warp搬运 128bytes才能充分利用cache line
3. 算术强度，memory-bound和compute-bound引入

算术强度：FLOPS/Bytes 完成算法所需要运算量和内存读写量之间的比值
vecor-add load a、load b、store c 3次读写 计算1次 float 所以为1/(3 * sizeof(float))

这个时候就要根据硬件信息算是否达到理论算力，A100 40G 带宽1555GB/s 算力19500 GFLOPS，在跑满带宽的情况下， 结果是远远小于理论算力的，因此是memory bound

matmul, (n, n) * (n, n)的矩阵, 计算量为 n次的乘法 n - 1次的加法 n * n * (2n - 1)，而A和B需要加载一次 2 *  n * n 写回一次 n * n, 数据类型为float 的情况下为n / 6 正常得到n > 12 就已经超过理论带宽。

