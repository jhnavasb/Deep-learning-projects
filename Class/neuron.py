import numpy as np
import matplotlib.pyplot as plt
import cv2

def activation(x):
    #return 1 / (1 + np.exp(-x)) #sigmoid
    #return np.tanh(x)
    #return np.where(x < 0, 0, 1)
    return np.uint8(x * 255) / 255


def d(x):
    #return x * (1 - x) #sigmoid
    #return 1.0 - np.tanh(x) ** 2
    return 0.0000001


def train(xS, wS, target, k):
    hE = np.zeros(k, dtype=float)

    for i in range(k):
        yn = activation(np.dot(xS, wS))
        error = target - yn
        fix = np.dot(xS.T, error * d(yn))
        wS += fix
        #wS = wS / np.max(wS)

        hE[i] = np.average(np.absolute(error))
        print("Epoca: ", i + 1)

    plt.plot(hE)
    plt.show()

    return wS


def graph(out, original, repair):
    plt.subplot(131), plt.imshow(original, cmap=plt.gray()), plt.title('Original')
    plt.subplot(132), plt.imshow(repair, cmap=plt.gray()), plt.title('Repair')
    plt.subplot(133), plt.imshow(out, cmap=plt.gray()), plt.title('Output')
    plt.show()


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main():
    imgOriginal = cv2.imread("Daytona1.jpg", 0)#255 * np.ones((2, 2))#
    imgRepair = cv2.imread("DaytonaX1.png", 0)#3 * 255 * np.ones((2, 2))#

    ini, n = imgRepair.ravel(), 2
    w, h = imgRepair.shape

    input = (np.array([ini]) / 255.0).T
    xS_1 = np.ones((len(input), n))
    xS_1[:, :-1] = input
    target = (np.array([imgOriginal.ravel()]) / 255.0).T

    #xS_1 = input
    xS, target = xS_1, target#shuffle(xS_1, target)

    wS = np.random.random((n, 1))
    #wS = 2 * np.random.random((n, 1)) - 1  # sigmoid

    wT = train(xS, wS, target, 500000)
    #wT = np.array([4158491.363849572372, 2079245.781615501968, 1386164.124931287952, -6683231.396833237261]).T
    #wT = wT / 255

    y = np.dot(xS_1, wT)
    out = y.reshape(w, h)
    out = np.uint8(out * 255)

    graph(out, imgOriginal, imgRepair)
    print("Done!")

    np.savetxt('test.out', wT)
if __name__ == "__main__":
   main()

