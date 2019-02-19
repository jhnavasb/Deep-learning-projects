import cv2
import numpy as np
from matplotlib import pyplot as plt


def ploting(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(img1), plt.title('Original')
    plt.subplot(122), plt.imshow(img2), plt.title('Modificada')
    plt.show()


def segmentation(img, select):
    img_bu = img.copy()
    if select == 0:
        img_bu = (((img > 15) & (img < 100)) * 255).astype("uint8")
    else:
        img_bu[(img_bu > 15) & (img_bu < 100)] = 255

    ploting(img, img_bu)


def gamma_law(img):

    gamma = 5
    c = 0.8
    img_1 = img / 255.0
    img_1 = c * cv2.pow(img_1, gamma)
    img_g = np.uint8(img_1 * 255)

    ploting(img, img_g)


def main():
    img = cv2.imread("Test4.tif")
    #img = cv2.imread("gr.jpg")
    #gamma_law(img)
    segmentation(img, 1)


if __name__ == "__main__":
   main()