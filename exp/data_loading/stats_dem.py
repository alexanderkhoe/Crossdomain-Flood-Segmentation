import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import rasterio

def getArrFlood(fname):
    return rasterio.open(fname).read()

IMAGE_DIR = 'data/CopernicusDEM'  
DATASET_DIR = 'splits'

def load_flood_identifiers(path, dataset_type):
    fpath = os.path.join(path, DATASET_DIR, f'flood_{dataset_type}_data.txt')
    identifiers = []
    with open(fpath) as f:
        for line in f:
            identifier = line.strip()
            if identifier:
                identifiers.append(identifier)
    return identifiers

class WelfordCalculator:
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.count = cp.zeros(n_channels, dtype=cp.int64)
        self.mean = cp.zeros(n_channels, dtype=cp.float64)
        self.M2 = cp.zeros(n_channels, dtype=cp.float64)
        # Add min and max trackers
        self.min_val = cp.full(n_channels, cp.inf, dtype=cp.float64)
        self.max_val = cp.full(n_channels, -cp.inf, dtype=cp.float64)
    
    def update(self, values):
        for c in range(self.n_channels):
            channel_data = values[c].flatten()
            n = channel_data.size
            
            # Update min and max
            channel_min = cp.min(channel_data)
            channel_max = cp.max(channel_data)
            self.min_val[c] = cp.minimum(self.min_val[c], channel_min)
            self.max_val[c] = cp.maximum(self.max_val[c], channel_max)
            
            # Update mean and variance (Welford's algorithm)
            delta = channel_data - self.mean[c]
            self.count[c] += n
            self.mean[c] += cp.sum(delta) / self.count[c]
            delta2 = channel_data - self.mean[c]
            self.M2[c] += cp.sum(delta * delta2)
    
    def get_stats(self):
        variance = cp.where(self.count > 1, self.M2 / self.count, 0)
        std = cp.sqrt(variance)
        return cp.asnumpy(self.mean), cp.asnumpy(std), cp.asnumpy(self.min_val), cp.asnumpy(self.max_val)

def compute_dataset_stats(data_path, used_bands, dataset_type='train'):
    print(f"Loading {dataset_type} identifiers...")
    identifiers = load_flood_identifiers(data_path, dataset_type)
     
    if isinstance(used_bands, int):
        n_bands = 1
        used_bands = [used_bands]   
    else:
        n_bands = len(used_bands)
    
    calculator = WelfordCalculator(n_bands)
    
    print(f"Processing {len(identifiers)} images on GPU...")
    for identifier in tqdm(identifiers):
        img_path = os.path.join(data_path, IMAGE_DIR, f"{identifier}_DEM.tif")
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
         
        arr = np.nan_to_num(getArrFlood(img_path))
         
        if n_bands == 1:
            arr_bands = arr[used_bands[0], :, :].astype(np.float32)
            arr_bands = arr_bands.reshape(1, *arr_bands.shape)   
        else:
            arr_bands = arr[list(used_bands), :, :].astype(np.float32)
        
 
        arr_bands = arr_bands / 10000.0   
        arr_bands_gpu = cp.asarray(arr_bands)
        calculator.update(arr_bands_gpu)
        del arr_bands_gpu
    
    means, stds, mins, maxs = calculator.get_stats()
    return means, stds, mins, maxs

if __name__ == "__main__":
    DATA_PATH = './datasets/sen1floods11_v1.1'  
    USED_BANDS = (0,)  
     
    try:
        print(f"Using GPU: {cp.cuda.Device()}")
    except Exception as e:
        print(f"Warning: Could not access GPU: {e}")
     
    means, stds, mins, maxs = compute_dataset_stats(DATA_PATH, USED_BANDS, dataset_type='train')
    
    
    print(f"\nChannel-wise Statistics:")
    for i in range(len(means)):
        print(f"\nChannel {i}:")
        print(f"  Min: {mins[i]:.6f}")
        print(f"  Max: {maxs[i]:.6f}")
        print(f"  Mean: {means[i]:.6f}")
        print(f"  Std: {stds[i]:.6f}")
     
    print("Copy these values to your code:")
 
    print(f"MEANS = {means.tolist()}")
    print(f"STDS = {stds.tolist()}")
    print(f"MINS = {mins.tolist()}")
    print(f"MAXS = {maxs.tolist()}")
 
    print("Actual DEM Elevation Range (if /10000.0 was applied):")
 
    print(f"Min Elevation (meters): {mins[0] * 10000.0:.2f}")
    print(f"Max Elevation (meters): {maxs[0] * 10000.0:.2f}")
    print(f"Mean Elevation (meters): {means[0] * 10000.0:.2f}")