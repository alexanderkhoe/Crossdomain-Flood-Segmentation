import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

DATA_PATH = './datasets/sen1floods11_v1.1'
IMAGE_DIR = 'data/S2L1CHand'
SPLITS_DIR = 'splits'
EXTENSION = '.tif'

def load_bolivia_samples():
    bolivia_file = os.path.join(DATA_PATH, SPLITS_DIR, 'flood_bolivia_data.txt')
    samples = []
    with open(bolivia_file, 'r') as f:
        for line in f:
            identifier = line.strip()
            if identifier:
                samples.append(identifier)
    return samples

def load_rgb(img_path):
    with rasterio.open(img_path) as src:
        img = src.read()
    
    # RGB bands: B4 (Red), B3 (Green), B2 (Blue) -> indices 3, 2, 1
    red = img[3, :, :]
    green = img[2, :, :]
    blue = img[1, :, :]
    
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    return rgb

def plot_bolivia_samples():
    samples = load_bolivia_samples()
    n_samples = len(samples)
    
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, identifier in enumerate(samples):
        img_path = os.path.join(DATA_PATH, IMAGE_DIR, f"{identifier}_S2Hand.tif")
        rgb = load_rgb(img_path)
        
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"{identifier}", fontsize=9)
        axes[idx].axis('off')
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('bolivia_rgb.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_bolivia_samples()