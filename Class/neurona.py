import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
target = np.array([0, 0, 0, 1])

e = np.zeros(4, dtype = float)
hE = np.zeros(20, dtype = float)

alpha = 0.2
w1 = np.random.rand(1)
w2 = np.random.rand(1)
w0 = np.random.rand(1)

t = np.random.permutation(4)

for j in range(0, 20):
    t = np.random.permutation(4)

    for i in range(0, 4):
        yn = x1[t[i]]*w1 + x2[t[i]]*w2 + w0
        if yn < 0: y = 0
        else: y = 1

        e[i] = target[t[i]] - y

        w0 = w0 + alpha * e[i] * 1
        w1 = w1 + alpha * e[i] * x1[t[i]]
        w2 = w2 + alpha * e[i] * x2[t[i]]

    hE[j] = np.average(np.absolute(e))
