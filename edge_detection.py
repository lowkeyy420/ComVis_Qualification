import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./assets/cat.jpg")
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def showResult(nrow=None, ncol=None, res_stack=None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(lbl)
        plt.axis("off")

    plt.show()


sobel_x = cv2.Sobel(igray, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(igray, cv2.CV_32F, 0, 1, ksize=3)

sobel_labels = ["sobel x", "sobel y"]
sobel_images = [sobel_x, sobel_y]

merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255 / merged_sobel.max()

showResult(1, 1, zip(["Edge Detection With Sobel"], [merged_sobel]))
