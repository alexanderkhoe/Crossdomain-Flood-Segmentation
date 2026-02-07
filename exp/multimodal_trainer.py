import argparse
import torch
import torch.nn as nn
import torch.linalg as LA

from models.hydraunet.DSUnet import DSUNet          # Dual Modality Classical UNet
from models.hydraunet.DSUnetTP import DSUNet3P      # Dual Modality UNet3+

from models.hydraunet.HydraUnet import HydraUNet       # Triple Modality Classical UNet
from models.hydraunet.HydraUnetTP import HydraUNet3P  # Triple Modality UNet3+
 

from models.hydraunet.config import (
    Config_DSUnet, # Dual Stream | S1, S2 (Classic UNet),
    Config_DSUnet3P,    # Dual Stream | S1, S2 (UNet3+ UNet),
    Config_HydraUNet, # Triple Stream | S1, S2, DEM (Classic UNet)
    Config_HydraUnet3P # Triple Stream | S1, S2, DEM (UNet3+)
)

import os
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from utils.testing import computeIOU, computeAccuracy, computeMetrics

from utils.tversky import TverskyLoss
from utils.customloss import DiceLoss, DiceLoss2 
from segmentation_models_pytorch.losses import FocalLoss, LovaszLoss 

from lion_pytorch import Lion


from data_loading.sen1_multimodal import get_loader_MM
 
import json

torch.manual_seed(124)

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Multimodal')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', help='Path to the data directory.')
    parser.add_argument('--version', type=str, default='Multimodal', help='Experiment version')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every n epochs')
    parser.add_argument('--test_interval', type=int, default=1, help='Test the model every n epochs')
    parser.add_argument('--loss_func', type=str, default='bce', help='Loss function to use: bce, dice, dice2, focal, lovasz, tversky')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=1, help='Fine-tune ratio for Prithvi models')
    
    
    return parser.parse_args()

def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(model, loader, optimizer, criterion, epoch, device):
 
    model.train()
    running_loss = 0.0
    running_samples = 0
    running_accuracies = 0
    running_iou = 0
    
    for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"Training Epoch {epoch+1}"), 0):

        sar_imgs, optical_imgs, elevation_imgs, masks, water_occur = batch_data
        
        # Send to device
        sar_imgs = sar_imgs.to(device)
        optical_imgs = optical_imgs.to(device)
        elevation_imgs = elevation_imgs.to(device)
        masks = masks.to(device)
        
        # Pass different modalities to different streams
        outputs = model(sar_imgs, optical_imgs, elevation_imgs)  
        targets = masks.squeeze(1) if len(masks.shape) > 3 else masks

        loss = criterion(outputs, targets.long())
 
        loss.backward()
        
        iou = computeIOU(outputs, targets, device)
        accuracy = computeAccuracy(outputs, targets, device)
        
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()
        running_accuracies += accuracy
        running_iou += iou
    
    avg_loss = running_loss / (batch_idx + 1)
    avg_acc = running_accuracies / (batch_idx + 1)
    avg_iou = running_iou / (batch_idx + 1)
    
    return avg_loss, avg_acc, avg_iou

def test(model, loader, criterion, device):
    model.eval()
    metricss = {}
    index = 0
    
    with torch.no_grad():
        for batch_data in loader:

            sar_imgs, optical_imgs, elevation_imgs, masks, water_occur = batch_data
            
            # Send to device
            sar_imgs = sar_imgs.to(device)
            optical_imgs = optical_imgs.to(device)
            elevation_imgs = elevation_imgs.to(device)
            masks = masks.to(device)
            
            # Pass different modalities to different streams
            predictions = model(sar_imgs, optical_imgs, elevation_imgs)


            loss = criterion(predictions, masks.squeeze(1) if len(masks.shape) > 3 else masks)
            
            metrics = computeMetrics(predictions, masks, device, criterion)
            metricss = {k: metricss.get(k, 0) + v for k, v in metrics.items()}
            
            index += 1
    
    TP, FP, TN, FN, loss = metricss['TP'].item(), metricss['FP'].item(), metricss['TN'].item(), metricss['FN'].item(), metricss['loss']
    
    IOU_floods = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0
    IOU_non_floods = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0
    Avg_IOU = (IOU_floods + IOU_non_floods) / 2

    ACC_floods = TP / (TP + FN) if (TP + FN) > 0 else 0
    ACC_non_floods = TN / (TN + FP) if (TN + FP) > 0 else 0
    Avg_ACC = (ACC_floods + ACC_non_floods) / 2

    return {
        'IOU_floods': IOU_floods,
        'IOU_non_floods': IOU_non_floods,
        'Avg_IOU': Avg_IOU,
        'ACC_floods': ACC_floods,
        'ACC_non_floods': ACC_non_floods,
        'Avg_ACC': Avg_ACC,
        'Loss': loss / index
    }

def ph1_loop(model, train_loader, valid_loader, criterion, device, writer, scheduler, optimizer, args):
    

    num_params_phase_1 = get_number_of_trainable_parameters(model)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, train_iou = train_model(
            model, train_loader, optimizer, criterion, epoch, device
        )
        logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        
        scheduler.step()
        
        if (epoch + 1) % args.test_interval == 0:
            val_metrics = test(model, valid_loader, criterion, device)
            logger.info(f"Valid - Avg IOU: {val_metrics['Avg_IOU']:.4f}, Avg ACC: {val_metrics['Avg_ACC']:.4f}, Loss: {val_metrics['Loss']:.4f}")
            
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"{metric_name}/valid", metric_value, epoch)
    return num_params_phase_1

def ph2_loop(model, model_name, train_loader, valid_loader, criterion, device, writer, scheduler, optimizer, args):
        logger.info(f"\nFine-tuning {model_name}")

        num_params_phase_2 = get_number_of_trainable_parameters(model)
        
        finetune_epochs = int(args.epochs * args.prithvi_finetune_ratio)
        
        finetune_lr = args.learning_rate * 0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, finetune_epochs)
        
        for epoch in range(args.epochs, args.epochs + finetune_epochs):
            logger.info(f"\n{model_name} - Fine-tune Epoch {epoch+1}/{args.epochs + finetune_epochs}")
             
            train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device)
            logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}")
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("IoU/train", train_iou, epoch)
            
            scheduler.step()
            
            if (epoch + 1) % args.test_interval == 0:
                val_metrics = test(model, valid_loader, criterion, device)
                logger.info(f"Valid - Avg IOU: {val_metrics['Avg_IOU']:.4f}, Avg ACC: {val_metrics['Avg_ACC']:.4f}")
                
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"{metric_name}/valid", metric_value, epoch)

        return num_params_phase_2

def train(model, model_name, train_loader, valid_loader, test_loader, bolivia_loader, 
                                   args, device, base_log_dir):    
    model_log_dir = os.path.join(base_log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    model_dir = os.path.join(model_log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    num_params_phase_1 = "N/A"
    num_params_phase_2 = "N/A"

    writer = SummaryWriter(model_log_dir)
    
    num_params = get_number_of_trainable_parameters(model)
    num_params_total = get_total_parameters(model)
    logger.info(f"{model_name}| Total Params: {num_params_total}")

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = Lion(model.parameters(), lr=args.learning_rate)

    if args.loss_func == 'diceloss':
        criterion = DiceLoss(device=device)
    elif args.loss_func == 'dl2':
        criterion = DiceLoss2(device=device, epsilon=1e-7)
    elif args.loss_func == 'bce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3], device=device), ignore_index=255)
    elif args.loss_func == 'focal':
        criterion = FocalLoss(mode="multiclass", alpha=0.25, gamma=2, ignore_index=255, reduction='mean')
    elif args.loss_func == 'lovasz':
        criterion = LovaszLoss(mode='multiclass', per_image=False, from_logits=True, ignore_index=255)
    elif args.loss_func == 'tversky':
        criterion = TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, gamma=1.33, eps=1e-7, ignore_index=255, from_logits=True)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.epochs)
     
    # for epoch in range(args.epochs):
    #     logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
    #     train_loss, train_acc, train_iou = train_model(
    #         model, train_loader, optimizer, criterion, epoch, device
    #     )
    #     logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}")
        
    #     writer.add_scalar("Loss/train", train_loss, epoch)
    #     writer.add_scalar("Accuracy/train", train_acc, epoch)
    #     writer.add_scalar("IoU/train", train_iou, epoch)
        
    #     scheduler.step()
        
    #     if (epoch + 1) % args.test_interval == 0:
    #         val_metrics = test(model, valid_loader, criterion, device)
    #         logger.info(f"Valid - Avg IOU: {val_metrics['Avg_IOU']:.4f}, Avg ACC: {val_metrics['Avg_ACC']:.4f}, Loss: {val_metrics['Loss']:.4f}")
            
    #         for metric_name, metric_value in val_metrics.items():
    #             writer.add_scalar(f"{metric_name}/valid", metric_value, epoch)


    attention_based_model = ['DSUNet_Prithvi','DSUNet3P_Prithvi','HydraUNet_Prithvi','HydraUNet3P_Prithvi']
    if model_name in attention_based_model and args.prithvi_finetune_ratio is not None:
        model.change_prithvi_trainability(False)
        logger.info(f"Prithvi weights frozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")

    num_params_phase_1 = ph1_loop(model, train_loader, valid_loader, criterion, device, writer, scheduler, optimizer, args)


    if model_name in attention_based_model and args.prithvi_finetune_ratio is not None:
        model.change_prithvi_trainability(True)
        logger.info(f"Prithvi weights unfrozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")

        num_params_phase_2 = ph2_loop(model, model_name, train_loader, valid_loader, criterion, device, writer, scheduler, optimizer, args)
    
    logger.info(f"\nFinal Evaluation")
    
    test_metrics = test(model, test_loader, criterion, device)
    bolivia_metrics = test(model, bolivia_loader, criterion, device)
    
    logger.info(f"Test Set - Avg IOU: {test_metrics['Avg_IOU']:.4f}, Avg ACC: {test_metrics['Avg_ACC']:.4f}, Loss: {test_metrics['Loss']:.4f}")
    logger.info(f"Bolivia Set - Avg IOU: {bolivia_metrics['Avg_IOU']:.4f}, Avg ACC: {bolivia_metrics['Avg_ACC']:.4f}, Loss: {bolivia_metrics['Loss']:.4f}")
    
    writer.close()
    
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_final.pt"))
    
    return {
        'model_name': model_name,
        'num_trainable_params': num_params,
        'num_total_params': num_params_total,
        'params_phase_1': num_params_phase_1,
        'params_phase_2': num_params_phase_2,
        'test_metrics': test_metrics,
        'bolivia_metrics': bolivia_metrics
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')
    
    args.version = f"{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_log_dir = f'./logs/Multimodal_{args.epochs}E_{args.loss_func.upper()}'
    os.makedirs(base_log_dir, exist_ok=True)
    
    logger.info("Loading datasets...")
  
    train_loader = get_loader_MM(args.data_path, DatasetType.TRAIN.value, args)
    valid_loader = get_loader_MM(args.data_path, DatasetType.VALID.value, args)
    test_loader = get_loader_MM(args.data_path, DatasetType.TEST.value, args)
    bolivia_loader = get_loader_MM(args.data_path, DatasetType.BOLIVIA.value, args)
 
    models = {
        'DSUNet': DSUNet(
            cfg=Config_DSUnet,
            use_prithvi=False
        ),
        'DSUNet_Prithvi': DSUNet(
            cfg=Config_DSUnet,
            use_prithvi=True
        ),


        'DSUNet3P': DSUNet3P(
            cfg=Config_DSUnet3P,
            use_prithvi=False
        ),
        'DSUNet3P_Prithvi': DSUNet3P(
            cfg=Config_DSUnet3P,
            use_prithvi=True
        ),


        'HydraUNet': HydraUNet(
            cfg=Config_HydraUNet,
            use_prithvi=False
        ),
        'HydraUNet_Prithvi': HydraUNet(
            cfg=Config_HydraUNet,
            use_prithvi=True
        ),


        'HydraUNet3P': HydraUNet3P(
            cfg=Config_HydraUnet3P,
            use_prithvi=False
        ),
        'HydraUNet3P_Prithvi': HydraUNet3P(
            cfg=Config_HydraUnet3P,
            use_prithvi=True
        )
    }

     
    results  = []
    for model_name, model in models.items():
        model.to(device)
        result = train(
            model, model_name, train_loader, valid_loader, test_loader, bolivia_loader, 
            args, device, base_log_dir
        )
        results.append(result)
    del model
    torch.cuda.empty_cache()

    results_file = os.path.join(base_log_dir, f'multimodal_e{args.epochs}_{args.loss_func}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=float)
    logger.info(f"\nResults saved to: {results_file}")
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)