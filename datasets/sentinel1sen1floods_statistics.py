import os
import rasterio
import cupy as cp
from tqdm import tqdm
DATASET_DIR = "./datasets/sen1floods11_v1.1"
IMAGE_DIR_S1 = "data/S1GRDHand"
INCLUDE = "./datasets/sen1floods11_v1.1/splits/flood_train_data.txt"
USED_BANDS = (1, 2)  # VV and VH bands
def get_file_paths(txt_path, dataset_dir, image_dir):
    """Load S1 file paths from txt."""
    paths = []
    with open(txt_path, 'r') as f:
        for line in f:
            identifier = line.strip()
            if identifier:
                path = os.path.join(dataset_dir, image_dir, f"{identifier}_S1Hand.tif")
                if os.path.exists(path):
                    paths.append(path)
    return paths
def calculate_s1_statistics(txt_path, dataset_dir, image_dir, used_bands=USED_BANDS):
    files = get_file_paths(txt_path, dataset_dir, image_dir)
    if not files:
        raise ValueError("No valid files found")
    print(f"Processing {len(files)} files...")
    num_bands = len(used_bands)
    all_pixels = []
    global_min = cp.full(num_bands, cp.inf)
    global_max = cp.full(num_bands, -cp.inf)
    for path in tqdm(files):
        try:
            with rasterio.open(path) as src:
                data = cp.array(src.read(list(used_bands)))
            pixels = data.reshape(num_bands, -1)
            valid = cp.all(cp.isfinite(pixels), axis=0)
            valid_pixels = pixels[:, valid]
            all_pixels.append(valid_pixels)
            global_min = cp.minimum(global_min, cp.min(valid_pixels, axis=1))
            global_max = cp.maximum(global_max, cp.max(valid_pixels, axis=1))
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"\nError: {path} - {e}")
    if not all_pixels:
        raise ValueError("No valid pixels found")
    # Vectorized computation on all data at once
    all_pixels = cp.concatenate(all_pixels, axis=1)
    mean = cp.mean(all_pixels, axis=1)
    std = cp.std(all_pixels, axis=1)
    # Min-max normalization
    mean_norm = (mean - global_min) / (global_max - global_min)
    std_norm = std / (global_max - global_min)
    mean, std = cp.asnumpy(mean), cp.asnumpy(std)
    mean_norm, std_norm = cp.asnumpy(mean_norm), cp.asnumpy(std_norm)
    print(f"\nVV - Mean: {mean[0]:.8f}, Std: {std[0]:.8f}")
    print(f"VH - Mean: {mean[1]:.8f}, Std: {std[1]:.8f}")
    print(f"\nMEANS = {mean_norm.tolist()}")
    print(f"STDS  = {std_norm.tolist()}")
    return mean_norm.tolist(), std_norm.tolist()
if __name__ == "__main__":
    calculate_s1_statistics(INCLUDE, DATASET_DIR, IMAGE_DIR_S1)