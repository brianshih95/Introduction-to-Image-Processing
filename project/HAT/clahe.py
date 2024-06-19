import cv2
import os


def clahe_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        adjusted_img = clahe_equalization(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, adjusted_img)

        print(f"Processed {img_name}")


input_folder = "filtered_results/original"
output_folder = "filtered_results/clahe"

process_folder(input_folder, output_folder)
