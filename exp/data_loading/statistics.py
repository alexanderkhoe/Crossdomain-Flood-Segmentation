import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import rasterio

def getArrFlood(fname):
    return rasterio.open(fname).read()

LABEL_DIR = 'data/LabelHand'
IMAGE_DIR = 'data/S2L1CHand'
DATASET_DIR = 'splits'

def load_flood_identifiers(path, dataset_type):
    """Load the list of file identifiers"""
    fpath = os.path.join(path, DATASET_DIR, f'flood_{dataset_type}_data.txt')
    identifiers = []
    with open(fpath) as f:
        for line in f:
            identifier = line.strip()
            if identifier:
                identifiers.append(identifier)
    return identifiers

class WelfordCalculator:
    """Welford's online algorithm for computing mean and variance using CuPy"""
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.count = cp.zeros(n_channels, dtype=cp.int64)
        self.mean = cp.zeros(n_channels, dtype=cp.float64)
        self.M2 = cp.zeros(n_channels, dtype=cp.float64)  # Sum of squared differences from mean
    
    def update(self, values):
 
        # Process each channel
        for c in range(self.n_channels):
            channel_data = values[c].flatten()
            n = channel_data.size
            
            # Vectorized Welford update
            delta = channel_data - self.mean[c]
            self.count[c] += n
            self.mean[c] += cp.sum(delta) / self.count[c]
            delta2 = channel_data - self.mean[c]
            self.M2[c] += cp.sum(delta * delta2)
    
    def get_stats(self):
        """Return mean and std for each channel"""
        variance = cp.where(self.count > 1, self.M2 / self.count, 0)
        std = cp.sqrt(variance)
        # Convert back to numpy for printing
        return cp.asnumpy(self.mean), cp.asnumpy(std)

def compute_dataset_stats(data_path, used_bands, dataset_type='train'):
 
    print(f"Loading {dataset_type} identifiers...")
    identifiers = load_flood_identifiers(data_path, dataset_type)
    n_bands = len(used_bands)
    calculator = WelfordCalculator(n_bands)
    
    print(f"Processing {len(identifiers)} images on GPU...")
    for identifier in tqdm(identifiers):
        img_path = os.path.join(data_path, IMAGE_DIR, f"{identifier}_S2Hand.tif")
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
        
        # Load image and select bands (this part stays in NumPy/CPU)
        arr = np.nan_to_num(getArrFlood(img_path))
        arr_bands = arr[list(used_bands), :, :].astype(np.float32)
        arr_bands = arr_bands / 10000.0
        
        # Transfer to GPU
        arr_bands_gpu = cp.asarray(arr_bands)
        
        # Update statistics on GPU
        calculator.update(arr_bands_gpu)
        
        # Optional: clear GPU memory for this image
        del arr_bands_gpu
    
    means, stds = calculator.get_stats()
    return means, stds

if __name__ == "__main__":
    # Configuration
    DATA_PATH = './datasets/sen1floods11_v1.1'  # Adjust this to your data path
    USED_BANDS = (1, 2, 3, 8, 11, 12)
    # USED_BANDS = (1,2,3,4,5,6,7,8,9,10,11,12)
    
    # Check if CuPy is available
    try:
        print(f"Using GPU: {cp.cuda.Device()}")
    except Exception as e:
        print(f"Warning: Could not access GPU: {e}")
    
    # Compute statistics on training data
    means, stds = compute_dataset_stats(DATA_PATH, USED_BANDS, dataset_type='train')
    
    print("\n\nCopy these values to your code:")
    print(f"MEANS = {means.tolist()}")
    print(f"STDS = {stds.tolist()}")