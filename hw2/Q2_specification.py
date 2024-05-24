import cv2
import numpy as np

tar_img = cv2.imread('Q2_source.jpg', cv2.IMREAD_GRAYSCALE)
ref_img = cv2.imread('Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)

tar_height, tar_width = tar_img.shape[0], tar_img.shape[1]
ref_height, ref_width = ref_img.shape[0], ref_img.shape[1]

tar_n_pixels = tar_height * tar_width
ref_n_pixels = ref_height * ref_width

tar_pdf = np.zeros(256)
for i in range(tar_height):
    for j in range(tar_width):
        tar_pdf[tar_img[i][j]] += 1
tar_pdf /= tar_n_pixels

ref_pdf = np.zeros(256)
for i in range(ref_height):
    for j in range(ref_width):
        ref_pdf[ref_img[i][j]] += 1
ref_pdf /= ref_n_pixels

tar_cdf, ref_cdf = np.zeros(256), np.zeros(256)
tar_cdf[0], ref_cdf[0] = tar_pdf[0], ref_pdf[0]
for i in range(1, 256):
    tar_cdf[i] = tar_cdf[i-1] + tar_pdf[i]
    ref_cdf[i] = ref_cdf[i-1] + ref_pdf[i]

map = np.zeros(256)
for i in range(256):
    diff = np.abs(tar_cdf[i] - ref_cdf)
    idx = diff.argmin()
    map[i] = idx

for i in range(tar_height):
    for j in range(tar_width):
        tar_img[i][j] = map[tar_img[i][j]]

cv2.imwrite("Q2_specification.jpg", tar_img)
