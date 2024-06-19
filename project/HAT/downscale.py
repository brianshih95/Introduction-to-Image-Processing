import os
from PIL import Image

input_folder = "improvement/filter"
output_folder = "improvement/filter + down"

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with Image.open(input_path) as img:
        img_resized = img.resize((256, 256), Image.LANCZOS)
        img_resized.save(output_path)
