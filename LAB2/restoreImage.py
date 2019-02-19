import numpy as np
import matplotlib.pyplot as plt
import cv2

imgOriginal = cv2.imread("Daytona1.jpg", 0)
imgRepair = cv2.imread("Daytona2.png", 0)

kOB = imgOriginal
kRB = imgRepair

x1 = np.array(kRB.ravel())
target = np.array(kOB.ravel())
w, h = imgRepair.shape
s = len(x1)

x1_1 = x1 / 255.0
target_1 = target / 255.0
'''
k = 20
e = np.zeros(s, dtype = float)
hE = np.zeros(k, dtype = float)

alpha = 0.01
w0 = np.random.rand(1)
w1 = np.random.rand(1)

for j in range(0, k):
    t = np.random.permutation(s)

    for i in range(0, s):
        yn = x1_1[t[i]] * w1 + w0
        y = yn
        e[i] = target_1[t[i]] - y

        w0 = w0 + alpha * e[i] * 1
        w1 = w1 + alpha * e[i] * x1_1[t[i]]

    hE[j] = np.average(np.absolute(e))
    print("Epoca: ", j + 1)

plt.plot(hE)
plt.show()
'''
w1 = 1.016493778927428160e+01
w0 = -9.186616326833849655e+00

yn = x1_1 * w1 + w0
out = yn.reshape(w, h)
out = np.uint8(out * 255)
print("Done")

plt.subplot(131), plt.imshow(kOB, cmap=plt.gray()), plt.title('Original')
plt.subplot(132), plt.imshow(kRB, cmap=plt.gray()), plt.title('Repair')
plt.subplot(133), plt.imshow(out, cmap=plt.gray()), plt.title('Output')
plt.show()

wT = np.array([w0, w1]).T
print(wT)
np.savetxt('test.out', wT)