import cv2
import os


def histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        adjusted_img = histogram_equalization(img)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, adjusted_img)

        print(f"Processed {img_name}")


input_folder = "filtered_results/None"
output_folder = "filtered_results/histogram_equalization"

process_folder(input_folder, output_folder)
