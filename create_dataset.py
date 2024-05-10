import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Veri setini oluşturma işlevi
def create_dataset(output_dir, num_samples=100):
    font = ImageFont.truetype("arial.ttf", 28)  # Yazı tipi ve boyutu ayarla

    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)  # Rakam dizinini oluşturun

        # Belirli sayıda örnek için bir döngü oluşturun
        for i in range(num_samples):
            print(f"Create {digit}: {i+1}/{num_samples}")

            # Rastgele bir rakam oluşturun ve bunu bir görüntüye çizin
            image = Image.new("L", (28, 28), color=255)  # Beyaz bir arka planla boş bir görüntü oluşturun
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), str(digit), fill=0, font=font)  # Rakamı görüntüye çizin

            # Görüntüyü kaydedin
            img_path = os.path.join(output_dir, str(digit), f"{digit}_{i}.png")
            image.save(img_path)

# Veri setini oluşturun
dataset_dir = "dataset"  # Veri setinin kaydedileceği dizin
create_dataset(dataset_dir, num_samples=100)  # Her bir rakam için 100 örnek oluşturun
