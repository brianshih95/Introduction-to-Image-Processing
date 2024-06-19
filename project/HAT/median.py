import cv2
import os


def median_blur(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        denoised_img = median_blur(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, denoised_img)

        print(f"Processed {img_name}")


input_folder = "filtered_results/None"
output_folder = "filtered_results/median_blur"

process_folder(input_folder, output_folder)
