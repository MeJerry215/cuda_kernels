import numpy as np
import math
np.random.seed(42)

FLT_MIN = np.finfo(np.float32).min
input = np.random.randn(1, 32, 1, 73).astype("float32")
np.set_printoptions(precision=4, suppress=True)
print("input", input)


def safe_softmax(input):
    exp_arr = np.exp(input - np.max(input, axis=-1, keepdims=True))
    print(exp_arr)
    return exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)


def three_pass_softmax(input):
    origin_shape = input.shape
    input = input.reshape(-1, origin_shape[-1])  # 32, 79
    output = np.zeros_like(input)
    # print(input)
    M, N = input.shape
    for m in range(M):
        max_N = FLT_MIN
        for n in range(N):
            # m_i = max(m_{i - 1}, x_i)
            max_N = max(max_N, input[m, n])
        sum_N = 0
        for n in range(N):
            # d_i = d_{i - 1} + e^{x_i - m_N}
            sum_N += math.exp(input[m, n] - max_N)
        print("sum_N", sum_N)
        for n in range(N):
            # a_i = (e^{x_i - m_N}) / d_N
            output[m, n] = (math.exp(input[m, n] - max_N)) / sum_N
        # print(max_N)
    return output.reshape(origin_shape)


def online_softmax(input):
    # create another helper sequence d_i^'
    origin_shape = input.shape
    input = input.reshape(-1, origin_shape[-1])  # 32, 79
    output = np.zeros_like(input)

    M, N = input.shape
    for m in range(M):
        sum_N = 0
        max_N = FLT_MIN
        for n in range(N):
            pre_max = max_N
            max_N = max(max_N, input[m, n])
            # online softmax 的基础
            sum_N = sum_N * math.exp((pre_max - max_N)) + \
                math.exp(input[m, n] - max_N)
        for n in range(N):
            output[m, n] = (math.exp(input[m, n] - max_N)) / sum_N

    return output.reshape(origin_shape)


s0 = safe_softmax(input)
s1 = three_pass_softmax(input)
s2 = online_softmax(input)  # 结果不对暂时不知道为啥错
print(np.allclose(s1, s2, atol=1e-3, rtol=1e-3))
# print(s1, s2)


arr = [-1.4831,		-0.0534, 	-0.7940, 	0.2750, 	-1.4693,		0.9238,		-0.2683,	-0.0470,
       -0.7228,		0.3967, 	1.8994,		-0.0136, 	1.4331, 		0.6369, 	0.3386,		-0.3063,
       ]
arr = [
    0.5802, 		-0.1653, 	-0.1108, 	-0.1566,	-1.9679,		-1.0373, 	0.9431,
]

arr = [
    -1.4831,		-0.0534, 	-0.7940, 	0.2750, 	-1.4693,		0.9238,		-0.2683,	-0.0470,
    -0.7228,		0.3967, 	1.8994,		-0.0136, 	1.4331, 		0.6369, 	0.3386,		-0.3063,
    0.5802, 		-0.1653, 	-0.1108, 	-0.1566,	-1.9679,		-1.0373, 	0.9431,
]
# 4.896761290240578 3.782361715750391 1.8994 2.8997246288885252 0.9431

arr = [-0.75773627, 1.670679]

input = np.array(arr, dtype="float32").reshape(1, -1)
print(safe_softmax(arr))


# arr = np.array([0.081, 0.919])
# brr = np.array(
#     [[
#         0.0000, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0060, 0.0070,
#         0.0080, 0.0090, 0.0100, 0.0110, 0.0120, 0.0130, 0.0140, 0.0150,
#         0.0160, 0.0170, 0.0180, 0.0190, 0.0200, 0.0210, 0.0220, 0.0230,
#         0.0240, 0.0250, 0.0260, 0.0270, 0.0280, 0.0290, 0.0300, 0.0310,
#         0.0320, 0.0330, 0.0340, 0.0350, 0.0360, 0.0370, 0.0380, 0.0390,
#         0.0400, 0.0410, 0.0420, 0.0430, 0.0440, 0.0450, 0.0460, 0.0470,
#         0.0480, 0.0490, 0.0500, 0.0510, 0.0520, 0.0530, 0.0540, 0.0550,
#         0.0560, 0.0570, 0.0580, 0.0590, 0.0600, 0.0610, 0.0620, 0.0630,
#         0.0640, 0.0650, 0.0660, 0.0670, 0.0680, 0.0690, 0.0700, 0.0710,
#         0.0720, 0.0730, 0.0740, 0.0750, 0.0760, 0.0770, 0.0780, 0.0790,
#         0.0800, 0.0810, 0.0820, 0.0830, 0.0840, 0.0850, 0.0860, 0.0870,
#         0.0880, 0.0890, 0.0900, 0.0910, 0.0920, 0.0930, 0.0940, 0.0950,
#         0.0960, 0.0970, 0.0980, 0.0990, 0.1000, 0.1010, 0.1020, 0.1030,
#         0.1040, 0.1050, 0.1060, 0.1070, 0.1080, 0.1090, 0.1100, 0.1110,
#         0.1120, 0.1130, 0.1140, 0.1150, 0.1160, 0.1170, 0.1180, 0.1190,
#         0.1200, 0.1210, 0.1220, 0.1230, 0.1240, 0.1250, 0.1260, 0.1270
#     ],
#         [
#         0.1280, 0.1290, 0.1300, 0.1310, 0.1320, 0.1330, 0.1340, 0.1350,
#         0.1360, 0.1370, 0.1380, 0.1390, 0.1400, 0.1410, 0.1420, 0.1430,
#         0.1440, 0.1450, 0.1460, 0.1470, 0.1480, 0.1490, 0.1500, 0.1510,
#         0.1520, 0.1530, 0.1540, 0.1550, 0.1560, 0.1570, 0.1580, 0.1590,
#         0.1600, 0.1610, 0.1620, 0.1630, 0.1640, 0.1650, 0.1660, 0.1670,
#         0.1680, 0.1690, 0.1700, 0.1710, 0.1720, 0.1730, 0.1740, 0.1750,
#         0.1760, 0.1770, 0.1780, 0.1790, 0.1800, 0.1810, 0.1820, 0.1830,
#         0.1840, 0.1850, 0.1860, 0.1870, 0.1880, 0.1890, 0.1900, 0.1910,
#         0.1920, 0.1930, 0.1940, 0.1950, 0.1960, 0.1970, 0.1980, 0.1990,
#         0.2000, 0.2010, 0.2020, 0.2030, 0.2040, 0.2050, 0.2060, 0.2070,
#         0.2080, 0.2090, 0.2100, 0.2110, 0.2120, 0.2130, 0.2140, 0.2150,
#         0.2160, 0.2170, 0.2180, 0.2190, 0.2200, 0.2210, 0.2220, 0.2230,
#         0.2240, 0.2250, 0.2260, 0.2270, 0.2280, 0.2290, 0.2300, 0.2310,
#         0.2320, 0.2330, 0.2340, 0.2350, 0.2360, 0.2370, 0.2380, 0.2390,
#         0.2400, 0.2410, 0.2420, 0.2430, 0.2440, 0.2450, 0.2460, 0.2470,
#         0.2480, 0.2490, 0.2500, 0.2510, 0.2520, 0.2530, 0.2540, 0.2550
#     ]]
# )


# print(arr @ brr)
# 0.4967 -0.1383  0.6477
# 1 , 1 , 1
# 0.4967 -0.1383  0.6477
# c1 = 1.9649
# c2 = c1 * math.exp(1.523 - 1.5792) + 1.7703
# c3 = c2 * math.exp(1.5792 + 0.4695) + 1
# c3 *= math.exp(-0.4695-1.5792)
# print(c1, c2)
# print(c1, c2, c3)
