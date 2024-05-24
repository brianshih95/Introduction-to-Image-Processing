import numpy as np
import cv2
import math

def bicubic_formula(p0, p1, p2, p3, x):
    return (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * x**3 + (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * x**2 + (-0.5 * p0 + 0.5 * p2) * x + p1

img = cv2.imread('building.jpg').astype(np.float32)
height, width, channels = img.shape

magnified_height, magnified_width = height * 2, width * 2
center_x, center_y = width / 2, height / 2
cos, sin = math.cos(math.radians(30)), math.sin(math.radians(30))
rotated_width = int(abs(width * cos) + abs(height * sin))
rotated_height = int(abs(height * cos) + abs(width * sin))

# rotate 30 degree (nearest neighbor)
new_img = np.zeros([height, width, 3], dtype=np.uint8)
new_img[:, :, :] = [0, 0, 0]

for i in range(rotated_height):
    for j in range(rotated_width):
        new_x = int(center_x - rotated_width / 2) + j
        new_y = int(center_y - rotated_height / 2) + i
        
        x = int(round((j - rotated_width / 2) * cos + (i - rotated_height / 2) * sin + center_x))
        y = int(round(-(j - rotated_width / 2) * sin + (i - rotated_height / 2) * cos + center_y))
        
        if x >= 0 and x < width and y >= 0 and y < height and new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
            new_img[new_y, new_x] = img[y, x]

cv2.imwrite('rotation (nearest neighbor).jpg', new_img)


# rotate 30 degree (bilinear)
new_img = np.zeros([height, width, 3], dtype=np.uint8)
new_img[:, :, :] = [0, 0, 0]

for i in range(rotated_height):
    for j in range(rotated_width):
        new_x = int(center_x - rotated_width / 2) + j
        new_y = int(center_y - rotated_height / 2) + i
        
        x = (j - rotated_width / 2) * cos + (i - rotated_height / 2) * sin + center_x
        y = -(j - rotated_width / 2) * sin + (i - rotated_height / 2) * cos + center_y
        
        if x >= 0 and x < width and y >= 0 and y < height and new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
            dx, dy = x - x1, y - y1

            top = dx * img[y1, x2] + (1 - dx) * img[y1, x1]
            bottom = dx * img[y2, x2] + (1 - dx) * img[y2, x1]            
            new_img[new_y, new_x] = dy * bottom + (1 - dy) * top

cv2.imwrite('rotation (bilinear).jpg', new_img)


# rotate 30 degree (bicubic)
new_img = np.zeros([height, width, 3], dtype=np.int32)
new_img[:, :, :] = [0, 0, 0]

for i in range(rotated_height):
    for j in range(rotated_width):
        new_x = int(center_x - rotated_width / 2) + j
        new_y = int(center_y - rotated_height / 2) + i
        
        x = (j - rotated_width / 2) * cos + (i - rotated_height / 2) * sin + center_x
        y = -(j - rotated_width / 2) * sin + (i - rotated_height / 2) * cos + center_y
        
        if x >= 0 and x < width and y >= 0 and y < height and new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
            x1, y1 = int(x), int(y)
            dx, dy = x - x1, y - y1
            
            tmp_x = []
            for k1 in range(-1, 3):
                xk = max(x1 + k1, 0)
                xk = min(xk, width - 1)

                tmp_y = []
                for k2 in range(-1, 3):
                    yk = max(y1 + k2, 0)
                    yk = min(yk, height - 1)
                    tmp_y.append(img[yk, xk])

                tmp_x.append(bicubic_formula(tmp_y[0], tmp_y[1], tmp_y[2], tmp_y[3], dy))
            
            new_img[new_y, new_x] = bicubic_formula(tmp_x[0], tmp_x[1], tmp_x[2], tmp_x[3], dx)

new_img = np.clip(new_img, 0, 255)
cv2.imwrite('rotation (bicubic).jpg', new_img)


# magnify 2x (nearest neighbor)
new_img = np.zeros([magnified_height, magnified_width, 3], dtype=np.uint8)

for i in range(magnified_height):
    for j in range(magnified_width):
        x, y = min(int(round(j / 2)), width - 1), min(int(round(i / 2)), height - 1)
        new_img[i, j] = img[y, x]

cv2.imwrite('magnification (nearest neighbor).jpg', new_img)


# magnify 2x (bilinear)
new_img = np.zeros([magnified_height, magnified_width, 3], dtype=np.uint8)

for i in range(magnified_height):
    for j in range(magnified_width):
        x, y = j / 2, i / 2
        x1, y1 = j // 2, i // 2
        x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
        dx, dy = x - x1, y - y1
        
        top = dx * img[y1, x2] + (1 - dx) * img[y1, x1]
        bottom = dx * img[y2, x2] + (1 - dx) * img[y2, x1]
        new_img[i, j] = dy * bottom + (1 - dy) * top

cv2.imwrite('magnification (bilinear).jpg', new_img)


# magnify 2x (bicubic)
new_img = np.zeros([magnified_height, magnified_width, 3], dtype=np.int32)

for i in range(magnified_height):
    for j in range(magnified_width):
        x, y = j / 2, i / 2
        x1, y1 = j // 2, i // 2
        dx, dy = x - x1, y - y1
        
        tmp_x = []
        for k1 in range(-1, 3):
            xk = max(x1 + k1, 0)
            xk = min(xk, width - 1)
            
            tmp_y = []
            for k2 in range(-1, 3):
                yk = max(y1 + k2, 0)
                yk = min(yk, height - 1)
                tmp_y.append(img[yk, xk])
            
            tmp_x.append(bicubic_formula(tmp_y[0], tmp_y[1], tmp_y[2], tmp_y[3], dy))
        
        new_img[i, j] = bicubic_formula(tmp_x[0], tmp_x[1], tmp_x[2], tmp_x[3], dx)

new_img = np.clip(new_img, 0, 255)
cv2.imwrite('magnification (bicubic).jpg', new_img)
