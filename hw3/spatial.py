import numpy as np
import cv2


def laplacian_kernel():
    
    # kernel 1
    kernel = np.array(
            [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]])
    
    # # kernel 2
    # kernel = np.array(
    #         [[-1, -1, -1],
    #         [-1, 9, -1],
    #         [-1, -1, -1]])

    return kernel


def convolution(img, kernel):
    
    n = kernel.shape[0]
    padding = n // 2
    padded_img = np.pad(img, padding, mode='constant')
    output = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            convolved = np.sum(padded_img[i:i+n, j:j+n] * kernel)
            output[i, j] = np.clip(convolved, 0, 1)

    return output


img = cv2.imread('images/dog.png', cv2.IMREAD_GRAYSCALE)
img = img / 255

kernel = laplacian_kernel()
sharpened_img = convolution(img, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image (Spatial Domain)', sharpened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
