import cv2
import os


def laplacian_sharpen(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(img - laplacian)
    return sharp


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        sharpened_img = laplacian_sharpen(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, sharpened_img)

        print(f"Processed {img_name}")


input_folder = "filtered_results/None"
output_folder = "filtered_results/laplacian_sharpen"

process_folder(input_folder, output_folder)
