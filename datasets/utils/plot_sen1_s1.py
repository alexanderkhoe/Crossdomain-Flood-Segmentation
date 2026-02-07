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


def load_s1_composite(img_path):
    """
    Load Sentinel-1 SAR data with VV and VH bands.
    Creates a false-color composite: VV (red), VH (green), VV/VH ratio (blue)
    """
    with rasterio.open(img_path) as src:
        img = src.read()
    
    # Assuming band 0 is VV and band 1 is VH
    vv = img[0, :, :].astype(np.float32)
    vh = img[1, :, :].astype(np.float32)
    
    # Avoid division by zero
    ratio = np.divide(vv, vh, out=np.zeros_like(vv), where=vh != 0)
    
    # Stack into composite
    composite = np.stack([vv, vh, ratio], axis=-1)
    
    # Normalize each channel independently
    for i in range(3):
        channel = composite[:, :, i]
        # Use percentile-based normalization for better visualization
        p2, p98 = np.percentile(channel[np.isfinite(channel)], [2, 98])
        if p98 > p2:
            composite[:, :, i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
        else:
            composite[:, :, i] = 0
    
    return composite


def plot_bolivia_samples(args):
    samples = load_bolivia_samples(args.data_path, args.splits_dir, args.split_file)
    n_samples = len(samples)
    n_cols = args.n_cols
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.fig_width, args.fig_height * n_rows))
    axes = axes.flatten()
    
    for idx, identifier in enumerate(samples):
        img_path = os.path.join(args.data_path, args.image_dir, f"{identifier}_{args.suffix}{args.extension}")
        
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
            axes[idx].text(0.5, 0.5, f"File not found\n{identifier}", 
                          ha='center', va='center', fontsize=args.fontsize)
            axes[idx].axis('off')
            continue
        
        s1_composite = load_s1_composite(img_path)
        axes[idx].imshow(s1_composite)
        axes[idx].set_title(f"{identifier}", fontsize=args.fontsize)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Bolivia flood samples from Sentinel-1 SAR data')
    parser.add_argument('--data-path', type=str, default='./datasets/sen1floods11_v1.1',
                        help='Path to dataset directory')
    parser.add_argument('--image-dir', type=str, default='data/S1GRDHand',
                        help='Image directory within data path')
    parser.add_argument('--splits-dir', type=str, default='splits',
                        help='Splits directory within data path')
    parser.add_argument('--split-file', type=str, default='flood_bolivia_data.txt',
                        help='Split file name')
    parser.add_argument('--extension', type=str, default='.tif',
                        help='Image file extension')
    parser.add_argument('--suffix', type=str, default='S1Hand',
                        help='Image file suffix')
    parser.add_argument('--n-cols', type=int, default=3,
                        help='Number of columns in plot grid')
    parser.add_argument('--fig-width', type=float, default=15,
                        help='Figure width')
    parser.add_argument('--fig-height', type=float, default=5,
                        help='Figure height per row')
    parser.add_argument('--fontsize', type=int, default=9,
                        help='Title font size')
    parser.add_argument('--output', type=str, default='./outputs/bolivia_s1.png',
                        help='Output file name')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    
    args = parser.parse_args()
    plot_bolivia_samples(args)