import cv2
import os


def unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        sharpened_img = unsharp_mask(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, sharpened_img)

        print(f"Processed {img_name}")


input_folder = "results/2xhat/visualization/custom"
output_folder = "improvement/filter"

process_folder(input_folder, output_folder)
