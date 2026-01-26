import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuration
DATA_PATH = './datasets/sen1floods11_v1.1'
IMAGE_DIR = 'data/S2L1CHand'
LABEL_DIR = 'data/LabelHand'
SPLITS_DIR = 'splits'

# RGB bands correspond to bands 1, 2, 3 (indices 0, 1, 2)
RGB_BANDS = [0, 1, 2]  # Red, Green, Blue bands

def load_bolivia_samples():
    """Load Bolivia sample identifiers from the split file."""
    bolivia_file = os.path.join(DATA_PATH, SPLITS_DIR, 'flood_bolivia_data.txt')
    
    samples = []
    with open(bolivia_file, 'r') as f:
        for line in f:
            identifier = line.strip()
            if identifier:
                samples.append(identifier)
    
    return samples

def load_and_normalize_rgb(img_path):
    """Load TIF image and extract RGB bands with normalization."""
    with rasterio.open(img_path) as src:
        img = src.read()
    
    # Extract RGB bands (bands 1, 2, 3)
    rgb = img[RGB_BANDS, :, :].astype(np.float32)
    
    # Handle NaN values
    rgb = np.nan_to_num(rgb)
    
    # Normalize to 0-1 range for visualization
    # Using percentile clipping for better contrast
    rgb_normalized = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[i]
        p2, p98 = np.percentile(band[band > 0], [2, 98])
        band_clipped = np.clip(band, p2, p98)
        if p98 > p2:
            rgb_normalized[i] = (band_clipped - p2) / (p98 - p2)
    
    # Transpose to (H, W, C) for matplotlib
    rgb_normalized = np.transpose(rgb_normalized, (1, 2, 0))
    
    return rgb_normalized

def load_mask(mask_path):
    """Load the flood mask."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    
    # Convert -1 to 255 (unknown/invalid)
    mask[mask == -1] = 255
    
    return mask

def plot_bolivia_samples(samples, max_samples=None, save_path=None):
    """Plot RGB images for Bolivia samples."""
    if max_samples:
        samples = samples[:max_samples]
    
    n_samples = len(samples)
    
    # Create figure with subplots
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, identifier in enumerate(samples):
        img_path = os.path.join(DATA_PATH, IMAGE_DIR, f"{identifier}_S2Hand.tif")
        mask_path = os.path.join(DATA_PATH, LABEL_DIR, f"{identifier}_LabelHand.tif")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load RGB image
        rgb = load_and_normalize_rgb(img_path)
        
        # Create subplot
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Display RGB image
        ax.imshow(rgb)
        ax.set_title(f"{identifier}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()


def main():
    """Main function to visualize Bolivia samples."""
    print("Loading Bolivia samples...")
    samples = load_bolivia_samples()
    print(f"Found {len(samples)} Bolivia samples")
    
    print("\nSample identifiers:")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample}")
    
    # Plot RGB only (all samples)
    print("\nPlotting RGB images...")
    plot_bolivia_samples(samples, max_samples=None, 
                        save_path='bolivia_rgb_samples.png')
    

if __name__ == '__main__':
    main()