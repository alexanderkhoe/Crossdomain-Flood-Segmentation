import os
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage import exposure

ROOT = 'datasets/WorldFloodsv2'
TEST_PATH_S2 = f'{ROOT}/train/S2/'
EXTENSION = '.tif'
OUTPUT_DIR = 'outputs'

TIMOR_LESTE_EVENTS = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

SATELLITE_RGB_BANDS = {
    'Sentinel-2': [4, 3, 2],
    'Pleiades-1A-1B': [4, 3, 2],
    'PlanetScope': [5, 4, 2]
}

def stretch_contrast(rgb, method='histogram', percentiles=(1, 99), gamma=1.2):
    rgb = np.clip(rgb, 0, 1)
    
    if method == 'percentile':
        p_low, p_high = np.percentile(rgb, percentiles, axis=(0, 1))
        stretched = np.zeros_like(rgb)
        for i in range(3):
            band = rgb[:, :, i]
            low, high = p_low[i], p_high[i]
            if high > low:
                band_stretched = np.clip((band - low) / (high - low), 0, 1)
                stretched[:, :, i] = band_stretched
            else:
                stretched[:, :, i] = band
        return stretched
    
    elif method == 'histogram':
        stretched = np.zeros_like(rgb)
        for i in range(3):
            band_uint8 = (rgb[:, :, i] * 255).astype(np.uint8)
            band_eq = exposure.equalize_hist(band_uint8)
            stretched[:, :, i] = band_eq
        return stretched
    
    elif method == 'gamma':
        return exposure.adjust_gamma(rgb, gamma=gamma)
    
    elif method == 'adaptive':
        stretched = np.zeros_like(rgb)
        for i in range(3):
            band_uint8 = (rgb[:, :, i] * 255).astype(np.uint8)
            band_adapteq = exposure.equalize_adapthist(band_uint8, clip_limit=0.03)
            stretched[:, :, i] = band_adapteq
        return stretched
    
    elif method == 'combined':
        rgb_percentile = stretch_contrast(rgb, method='percentile', percentiles=percentiles)
        rgb_gamma = exposure.adjust_gamma(rgb_percentile, gamma=gamma)
        return rgb_gamma
    
    else:
        return rgb

def load_and_enhance_rgb(img_path, satellite, enhance_method='combined'):
    rgb_bands = SATELLITE_RGB_BANDS[satellite]
    
    with rasterio.open(img_path) as src:
        img = src.read(rgb_bands)
    
    rgb = np.transpose(img, (1, 2, 0)).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    
    p1, p99 = np.percentile(rgb, [1, 99])
    rgb = np.clip(rgb, p1, p99)
    
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    if satellite == 'PlanetScope':
        scale_red = 0.28
        scale_green = 1.05
        scale_blue = 1.00

        rgb[:, :, 0] *= scale_red
        rgb[:, :, 1] *= scale_green
        rgb[:, :, 2] *= scale_blue

        rgb = np.clip(rgb, 0, 1)
    
    if enhance_method:
        rgb = stretch_contrast(rgb, method=enhance_method)
    
    return rgb

def save_individual_samples(enhance_method='combined', include_title=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Saving individual samples to '{OUTPUT_DIR}' directory...")
    print(f"Enhancement method: {enhance_method}")
    
    for event_id, satellite in TIMOR_LESTE_EVENTS.items():
        img_path = f"{TEST_PATH_S2}{event_id}{EXTENSION}"
         
        rgb = load_and_enhance_rgb(img_path, satellite, enhance_method)
        
        if include_title:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb)
            ax.set_title(f"{event_id}\n{satellite}", fontsize=12, pad=10)
            
            contrast = np.std(rgb)
            brightness = np.mean(rgb)
            ax.set_xlabel(f"Contrast: {contrast:.3f} | Brightness: {brightness:.3f}", 
                        fontsize=10)
            ax.axis('off')
        else:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        output_filename = f"{event_id}_{satellite.replace(' ', '_')}_{enhance_method}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1 if include_title else 0)
        plt.close()
        
        print(f"Saved: {output_filename}")
            
 
     
    print(f"All samples saved to '{OUTPUT_DIR}' directory")

def save_clean_samples(enhance_method='combined'):

    clean_dir = f"{OUTPUT_DIR}/clean"
    os.makedirs(clean_dir, exist_ok=True)
    
    print(f"Saving clean samples (no titles) to '{clean_dir}' directory...")
    print(f"Enhancement method: {enhance_method}")

    
    for event_id, satellite in TIMOR_LESTE_EVENTS.items():
        img_path = f"{TEST_PATH_S2}{event_id}{EXTENSION}"
 
        rgb = load_and_enhance_rgb(img_path, satellite, enhance_method)
        
        output_filename = f"{event_id}_{satellite.replace(' ', '_')}.png"
        output_path = os.path.join(clean_dir, output_filename)
        
        plt.imsave(output_path, rgb, dpi=150)
        
        print(f"Saved: {output_filename}")
        
 

def plot_timor_leste_samples(enhance_method='combined'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (event_id, satellite) in enumerate(TIMOR_LESTE_EVENTS.items()):
        img_path = f"{TEST_PATH_S2}{event_id}{EXTENSION}"
        

        rgb = load_and_enhance_rgb(img_path, satellite, enhance_method)
        
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"{event_id}\n{satellite}", fontsize=9)
        
        contrast = np.std(rgb)
        brightness = np.mean(rgb)
        axes[idx].set_xlabel(f"Contrast: {contrast:.3f}\nBrightness: {brightness:.3f}", 
                            fontsize=7)
        
 
        axes[idx].axis('off')
    
    axes[-1].axis('off')
    
    method_names = {
        'percentile': 'Percentile Stretching (2-98%)',
        'histogram': 'Histogram Equalization',
        'gamma': 'Gamma Correction (γ=1.2)',
        'adaptive': 'Adaptive Histogram Equalization',
        'combined': 'Combined (Percentile + Gamma)'
    }
    
    plt.suptitle(f"Timor-Leste Samples with {method_names[enhance_method]}", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'timor_leste_enhanced_{enhance_method}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process and save satellite imagery samples')
    
    parser.add_argument('--enhance-method', type=str, default='combined',
                        choices=['percentile', 'histogram', 'gamma', 'adaptive', 'combined'],
                        help='Enhancement method for contrast improvement')
    
    parser.add_argument('--include-title', action='store_true', default=True,
                        help='Include title and metadata on individual samples')
    
    parser.add_argument('--no-title', dest='include_title', action='store_false',
                        help='Exclude title and metadata on individual samples')
    
    parser.add_argument('--skip-individual', action='store_true', default=False,
                        help='Skip saving individual samples')
    
    parser.add_argument('--skip-clean', action='store_true', default=False,
                        help='Skip saving clean samples')
    
    parser.add_argument('--plot-grid', action='store_true', default=False,
                        help='Create the original grid view')
    
    args = parser.parse_args()
    
    if not args.skip_individual:
        save_individual_samples(enhance_method=args.enhance_method, 
                               include_title=args.include_title)
    
    if not args.skip_clean:
        save_clean_samples(enhance_method=args.enhance_method)
    
    if args.plot_grid:
        plot_timor_leste_samples(enhance_method=args.enhance_method)

if __name__ == '__main__':
    main()