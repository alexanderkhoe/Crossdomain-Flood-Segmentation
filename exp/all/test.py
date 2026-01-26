import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
from data_loading.ml4floods import (
    timor_leste_events, test_path_s2, test_path_label, extension, 
    getArrFlood, USED_BANDS, MEANS, STDS, INPUT_SIZE, PATCH_SIZE, 
    processTimorLesteData, load_timor_leste_data_with_flood_masks
)
from torchvision import transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize model comparisons on Timor-Leste')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', 
                        help='Path to the data directory.')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing trained models (e.g., ./logs/comparison_xxx)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for inference (reduce if OOM)')
    parser.add_argument('--output_dir', type=str, default='./comparison_outputs',
                        help='Directory to save individual comparison figures')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of patches to visualize (default: all)')
    parser.add_argument('--samples_per_event', type=int, default=3,
                        help='Number of patches to sample from each event')
    parser.add_argument('--use_flood_masks', action='store_true',
                        help='Use flood masks from GeoJSON as ground truth')
    
    # Model parameters (should match training)
    parser.add_argument('--prithvi_out_channels', type=int, default=768)
    parser.add_argument('--unet_out_channels', type=int, default=768)
    parser.add_argument('--combine_func', type=str, default='concat')
    parser.add_argument('--random_dropout_prob', type=float, default=2/3)
    
    return parser.parse_args()

def load_model(model_name, args, device):
    """Load a trained model"""
    args.num_classes = 2
    args.in_channels = 6
    
    if model_name == 'unet':
        model = UNet(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            unet_encoder_size=args.unet_out_channels
        )
    elif model_name == 'prithvi':
        model = PritviSegmenter(
            weights_path='./prithvi/Prithvi_100M.pt',
            device=device,
            output_channels=args.num_classes,
            prithvi_encoder_size=args.prithvi_out_channels
        )
    elif model_name == 'prithvi_unet':
        model = PrithviUNet(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            weights_path='./prithvi/Prithvi_100M.pt',
            device=device,
            prithvi_encoder_size=args.prithvi_out_channels,
            unet_encoder_size=args.unet_out_channels,
            combine_method=args.combine_func,
            dropout_prob=args.random_dropout_prob
        )
    
    # Load weights
    model_path = os.path.join(args.models_dir, model_name, 'models', 'model_final.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def load_timor_leste_sampled_patches(samples_per_event=3, use_flood_masks=None):
    """
    Load a subset of patches from each Timor-Leste event.
    This avoids loading all patches into memory at once.
    
    Args:
        samples_per_event: Number of patches to sample from each event
        use_flood_masks: If True, use flood masks from GeoJSON as ground truth
    """
    all_samples = []
    
    if use_flood_masks:
        # Use the function that loads flood masks
        print("Loading data with flood masks from GeoJSON...")
        data_with_masks = load_timor_leste_data_with_flood_masks()
        
        # Process each event
        event_ids = list(timor_leste_events.keys())
        for idx, (event_id, satellite) in enumerate(timor_leste_events.items()):
            arr_x, arr_y = data_with_masks[idx]
            
            # Get all patches for this event
            patches = processTimorLesteData((arr_x, arr_y))
            
            # Sample evenly spaced patches
            total_patches = len(patches)
            if samples_per_event >= total_patches:
                selected_indices = range(total_patches)
            else:
                # Evenly space samples across all patches
                step = total_patches / samples_per_event
                selected_indices = [int(i * step) for i in range(samples_per_event)]
            
            # Create samples for selected patches
            for patch_idx in selected_indices:
                img, label = patches[patch_idx]
                all_samples.append({
                    'image': img,
                    'label': label,
                    'event_id': event_id,
                    'satellite': satellite,
                    'patch_idx': patch_idx,
                    'total_patches': total_patches
                })
    else:
        # Original method without flood masks
        print("Loading data with permanent water labels...")
        for event_id, satellite in timor_leste_events.items():
            img_path = f"{test_path_s2}{event_id}{extension}"
            label_path = f"{test_path_label}{event_id}{extension}"
            
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                raise ValueError(f"File not found: {img_path} or {label_path}")
            
            # Load raw arrays
            arr_x = np.nan_to_num(getArrFlood(img_path))
            arr_y = getArrFlood(label_path)
            
            # Convert ml4floods labels to binary
            arr_y_new = np.zeros_like(arr_y)
            arr_y_new[arr_y == 0] = 255  # invalid -> ignore
            arr_y_new[arr_y == 1] = 0    # land -> land
            arr_y_new[arr_y == 2] = 1    # water -> water
            arr_y_new[arr_y == 3] = 1    # permanent water -> water
            
            # Get all patches for this event
            patches = processTimorLesteData((arr_x, arr_y_new))
            
            # Sample evenly spaced patches
            total_patches = len(patches)
            if samples_per_event >= total_patches:
                selected_indices = range(total_patches)
            else:
                # Evenly space samples across all patches
                step = total_patches / samples_per_event
                selected_indices = [int(i * step) for i in range(samples_per_event)]
            
            # Create samples for selected patches
            for idx in selected_indices:
                img, label = patches[idx]
                all_samples.append({
                    'image': img,
                    'label': label,
                    'event_id': event_id,
                    'satellite': satellite,
                    'patch_idx': idx,
                    'total_patches': total_patches
                })
    
    return all_samples

def get_predictions_batch(models, images, device, batch_size=8):
    """
    Get predictions from all models in batches to avoid OOM.
    
    Args:
        models: Dictionary of {model_name: model}
        images: Tensor of images [N, C, H, W]
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of {model_name: predictions_array}
    """
    num_samples = images.shape[0]
    predictions = {name: [] for name in models.keys()}
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_imgs = images[i:batch_end].to(device)
        
        with torch.no_grad():
            for model_name, model in models.items():
                outputs = model(batch_imgs)
                pred = torch.argmax(outputs, dim=1)
                predictions[model_name].append(pred.cpu())
        
        # Clear GPU memory after each batch
        del batch_imgs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all batch predictions
    for model_name in predictions:
        predictions[model_name] = torch.cat(predictions[model_name], dim=0).numpy()
    
    return predictions

def save_individual_comparison(sample, predictions, model_names, output_path, use_flood_masks=False):
    """
    Create and save a single comparison figure for one patch
    """
    num_cols = 2 + len(model_names)
    fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4))
    
    img_np = sample['image'].numpy()
    mask_np = sample['label'].numpy()
    
    # Determine GT title based on data source
    gt_title = 'Ground Truth\n(Flood Mask)' if use_flood_masks else 'Ground Truth\n(Permanent Water)'
    
    col = 0
    
    # Column 0: RGB visualization
    rgb = img_np[:3, :, :].transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    axes[col].imshow(rgb)
    axes[col].set_title('Input RGB', fontsize=14, fontweight='bold')
    axes[col].axis('off')
    col += 1
    
    # Column 1: Ground Truth
    gt_vis = np.ma.masked_where(mask_np == 255, mask_np)
    axes[col].imshow(gt_vis, cmap='Blues', vmin=0, vmax=1)
    axes[col].set_title(gt_title, fontsize=14, fontweight='bold')
    axes[col].axis('off')
    col += 1
    
    # Remaining columns: Model predictions
    for model_name in model_names:
        axes[col].imshow(predictions[model_name], cmap='Blues', vmin=0, vmax=1)
        display_name = model_name.replace('_', '-').upper()
        if display_name == 'PRITHVI-UNET':
            display_name = 'U-Prithvi'
        axes[col].set_title(f'{display_name}', fontsize=14, fontweight='bold')
        axes[col].axis('off')
        col += 1
    
    # Add title with event and patch info
    event_id = sample['event_id']
    satellite = sample['satellite']
    patch_idx = sample['patch_idx']
    total_patches = sample['total_patches']
    
    data_source = 'Flood Masks' if use_flood_masks else 'Permanent Water'
    plt.suptitle(f'{event_id} ({satellite}) - Patch {patch_idx}/{total_patches} (GT: {data_source})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_individual_figures(models, samples, device, output_dir, batch_size=8, use_flood_masks=False):
    """
    Create individual comparison figures for each sample
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(samples)
    model_names = list(models.keys())
    
    # Prepare all images (keep on CPU initially)
    imgs = torch.stack([s['image'] for s in samples])
    
    # Get predictions in batches
    print(f"  Running inference in batches of {batch_size}...")
    predictions = get_predictions_batch(models, imgs, device, batch_size)
    
    # Save individual figures
    print(f"  Saving {num_samples} individual figures...")
    for i in range(num_samples):
        # Extract predictions for this sample
        sample_preds = {name: predictions[name][i] for name in model_names}
        
        # Create filename
        event_id = samples[i]['event_id']
        patch_idx = samples[i]['patch_idx']
        filename = f"{event_id}_patch{patch_idx:03d}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save figure
        save_individual_comparison(
            samples[i], 
            sample_preds, 
            model_names, 
            output_path,
            use_flood_masks
        )
        
        if (i + 1) % 10 == 0:
            print(f"    Saved {i + 1}/{num_samples} figures...")
    
    print(f"All figures saved to: {output_dir}")

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    print(f'Using device: {device}')
    
    # Load sampled Timor-Leste patches
    print(f"Loading {args.samples_per_event} patches per event from Timor-Leste...")
    samples = load_timor_leste_sampled_patches(
        samples_per_event=args.samples_per_event,
        use_flood_masks=args.use_flood_masks
    )
    print(f"Loaded {len(samples)} total patches from 5 events:")
    
    # Count patches per event
    from collections import Counter
    event_counts = Counter([s['event_id'] for s in samples])
    for event_id, count in event_counts.items():
        satellite = samples[[s['event_id'] for s in samples].index(event_id)]['satellite']
        total = samples[[s['event_id'] for s in samples].index(event_id)]['total_patches']
        print(f"  - {event_id} ({satellite}): {count}/{total} patches")
    
    # Optionally limit total number of samples
    if args.max_samples and len(samples) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples for visualization...")
        samples = samples[:args.max_samples]
    
    # Load all models
    print("\nLoading models...")
    models = {}
    for model_name in ['unet', 'prithvi', 'prithvi_unet']:
        try:
            print(f"  Loading {model_name}...")
            models[model_name] = load_model(model_name, args, device)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    if not models:
        raise ValueError("No models could be loaded. Check your models_dir path.")
    
    print(f"Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # Create individual comparison figures
    print(f"\nCreating individual comparison figures for {len(samples)} patches...")
    create_individual_figures(
        models, 
        samples,
        device, 
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_flood_masks=args.use_flood_masks
    )
    
    print("Done!")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)