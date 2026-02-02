import os
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def load_bolivia_samples(data_path, splits_dir, split_file):
    bolivia_file = os.path.join(data_path, splits_dir, split_file)
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
    
    red = img[3, :, :]
    green = img[2, :, :]
    blue = img[1, :, :]
    
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    return rgb

def plot_bolivia_samples(args):
    samples = load_bolivia_samples(args.data_path, args.splits_dir, args.split_file)
    n_samples = len(samples)
    
    n_cols = args.n_cols
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.fig_width, args.fig_height * n_rows))
    axes = axes.flatten()
    
    for idx, identifier in enumerate(samples):
        img_path = os.path.join(args.data_path, args.image_dir, f"{identifier}_{args.suffix}{args.extension}")
        rgb = load_rgb(img_path)
        
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"{identifier}", fontsize=args.fontsize)
        axes[idx].axis('off')
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Bolivia flood samples from Sentinel data')
    parser.add_argument('--data-path', type=str, default='./datasets/sen1floods11_v1.1',
                        help='Path to dataset directory')
    parser.add_argument('--image-dir', type=str, default='data/S2L1CHand',
                        help='Image directory within data path')
    parser.add_argument('--splits-dir', type=str, default='splits',
                        help='Splits directory within data path')
    parser.add_argument('--split-file', type=str, default='flood_bolivia_data.txt',
                        help='Split file name')
    parser.add_argument('--extension', type=str, default='.tif',
                        help='Image file extension')
    parser.add_argument('--suffix', type=str, default='S2Hand',
                        help='Image file suffix')
    parser.add_argument('--n-cols', type=int, default=3,
                        help='Number of columns in plot grid')
    parser.add_argument('--fig-width', type=float, default=15,
                        help='Figure width')
    parser.add_argument('--fig-height', type=float, default=5,
                        help='Figure height per row')
    parser.add_argument('--fontsize', type=int, default=9,
                        help='Title font size')
    parser.add_argument('--output', type=str, default='./outputs/bolivia_rgb.png',
                        help='Output file name')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    
    args = parser.parse_args()
    plot_bolivia_samples(args)