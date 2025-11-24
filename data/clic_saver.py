import os
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

# Folder where you want to save the subset images
output_dir = "data/clic_subset"
os.makedirs(output_dir, exist_ok=True)

# Load the 'train' split or any split you want
dataset = tfds.load('clic', split='train', shuffle_files=False)

# Limit number of images to save in subset, e.g. 50
max_images = 1000
count = 0

for example in dataset:
    img = example['image'].numpy()         # Decode image to numpy array (H x W x 3)
    img_pil = Image.fromarray(img)         # Convert numpy array to PIL Image

    filename = f"clic_img_{count+1:04d}.png"
    filepath = os.path.join(output_dir, filename)
    img_pil.save(filepath)                  # Save as PNG file

    count += 1
    if count >= max_images:
        break

print(f"{count} images saved to {output_dir}")
