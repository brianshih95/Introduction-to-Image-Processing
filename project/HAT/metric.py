from skimage import io, color
import pyiqa
from torchvision import transforms

total_niqe, total_brisque = 0.0, 0.0
for i in range(1, 16):
    image_path = f'new/{i}_Real_HAT_GAN_Sharper_Real_HAT_GAN_SRx4.png'

    image = io.imread(image_path)

    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0)

    print(f"Image {i}")

    # NIQE
    niqe_metric = pyiqa.create_metric('niqe')
    niqe_score = niqe_metric(image_tensor)
    print(f"NIQE Score:    {niqe_score}")
    total_niqe += niqe_score

    # BRISQUE
    brisque_metric = pyiqa.create_metric('brisque')
    brisque_score = brisque_metric(image_tensor).item()
    brisque_score = abs(brisque_score)
    print(f"BRISQUE Score: {brisque_score}")
    total_brisque += brisque_score

print(f"\nAverage NIQE Score:  {total_niqe / 15}")
print(f"Average BRISQUE Score: {total_brisque / 15}")
