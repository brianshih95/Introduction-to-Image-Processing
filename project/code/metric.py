from skimage import io
import pyiqa
from torchvision import transforms

for i in range(1, 2):
    image_path = f'results/Real_HAT_GAN_sharper/visualization/custom/{i}_Real_HAT_GAN_sharper.png'
    image = io.imread(image_path)

    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # 增加一個維度以匹配 batch 格式

    print(i)
    
    niqe_metric = pyiqa.create_metric('niqe')
    niqe_score = niqe_metric(image_tensor)
    print(f"NIQE Score: {niqe_score.item()}")

    brisque_metric = pyiqa.create_metric('brisque')
    brisque_score = brisque_metric(image_tensor)
    print(f"BRISQUE Score: {brisque_score.item()}")
