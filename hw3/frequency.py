import numpy as np
import cv2


def laplacian_filter(img):
    
    P, Q = img.shape
    H = np.zeros((P, Q))
    for u in range(P):
        for v in range(Q):
            H[u, v] = -4 * (np.pi**2) * ((u - P/2)**2 + (v - Q/2)**2)

    return H


img = cv2.imread('images/dog.png', cv2.IMREAD_GRAYSCALE)
img = img / 255

# Get the Spectrum using Fourier Transform
spectrum = np.fft.fft2(img)
F = np.fft.fftshift(spectrum)
H = laplacian_filter(img)

# Apply filter on Spectrum
laplacian = H * F

# Convert the new Spectrum to spatial domain using Inverse Fourier Transform
laplacian = np.fft.ifftshift(laplacian)
laplacian = np.fft.ifft2(laplacian).real

# Normalization
range = np.max(laplacian) - np.min(laplacian)
scaled = ((laplacian - np.min(laplacian)) * 2 / range) - 1

sharpened_img = img - scaled
sharpened_img = np.clip(sharpened_img, 0, 1)

cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image (Frequency Domain)', sharpened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
