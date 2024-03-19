import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

data = np.fromfile("test_float32.bin", dtype="float32")

x1 = data[:32 * 12 * 128 * 1021].reshape(-1, 1021)
x2 = data[:32 * 12 * 128 * 1022].reshape(-1, 1022)

o1 = softmax(x1)
o2 = softmax(x2)

e1 = np.fromfile("softmax_out_1021.bin", dtype="float32")
e2 = np.fromfile("softmax_out_1022.bin", dtype="float32")

print(np.allclose(o1, e1))
print(np.allclose(o2, e2))
