import cv2
import os


def gaussian_blur(img, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(img, kernel_size, sigma)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        denoised_img = gaussian_blur(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, denoised_img)

        print(f"Processed {img_name}")


input_folder = "filtered_results/None"
output_folder = "filtered_results/gaussian_blur"

process_folder(input_folder, output_folder)
