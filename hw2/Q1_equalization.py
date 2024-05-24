import cv2
import numpy as np

img = cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)

height, width = img.shape[0], img.shape[1]

n_pixels = height * width

pdf = np.zeros(256)
for i in range(height):
    for j in range(width):
        pdf[img[i][j]] += 1
pdf /= n_pixels

cdf = np.zeros(256)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + pdf[i]
cdf *= 255

for i in range(height):
    for j in range(width):
        img[i][j] = round(cdf[img[i][j]])

cv2.imwrite("Q1_equalization.jpg", img)
