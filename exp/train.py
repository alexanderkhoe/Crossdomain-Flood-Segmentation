import argparse
import logging
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

# modules for U-Prithvi and Unet
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet 

import os
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from utils.testing import computeIOU, computeAccuracy, computeMetrics

from utils.customloss import DiceLoss, DiceLoss2 
from segmentation_models_pytorch.losses import FocalLoss, LovaszLoss, JaccardLoss, TverskyLoss
from torch.amp import autocast, GradScaler

from peft import LoraConfig, get_peft_model
from data_loading.sen1floods11 import get_loader
import json

torch.manual_seed(124)

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare Prithvi, UNet, and Prithvi-UNet models')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', help='Path to the data directory.')
    parser.add_argument('--version', type=str, default='comparison', help='Experiment version')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--loss_func', type=str, default='bce', help='Loss function to use: bce, dice, dice2, focal, lovasz, tversky')
    parser.add_argument('--prithvi_out_channels', type=int, default=768, help='Number of output channels from the Prithvi encoders')
    parser.add_argument('--unet_out_channels', type=int, default=768, help='Number of output channels from the UNet encoders')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=1, help='Fine-tune ratio for Prithvi models')
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every n epochs')
    parser.add_argument('--test_interval', type=int, default=1, help='Test the model every n epochs')
    parser.add_argument('--combine_func', type=str, default='concat', choices=['concat', 'mul', 'add'], help='Combination function for U-Prithvi')
    parser.add_argument('--random_dropout_prob', type=float, default=2/3, help='Dropout probability for U-Prithvi')

    
    return parser.parse_args()

def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(model, loader, optimizer, criterion, epoch, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_samples = 0
    running_accuracies = 0
    running_iou = 0
    
    for batch_idx, (imgs, masks) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch+1}"), 0):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Use autocast for mixed precision
        with autocast(device_type=str(device).split(':')[0], enabled=scaler is not None, dtype=torch.bfloat16):
            outputs = model(imgs)
            targets = masks.squeeze(1)
            loss = criterion(outputs, targets.long())
        
        # Use scaler for mixed precision training
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        iou = computeIOU(outputs, targets, device)
        accuracy = computeAccuracy(outputs, targets, device)
    
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
        for (imgs, masks) in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            predictions = model(imgs)
            
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

def train_single_model(model_name, model, train_loader, valid_loader, test_loader, bolivia_loader, args, device, base_log_dir):
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")
    
    model_log_dir = os.path.join(base_log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    model_dir = os.path.join(model_log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    num_params_phase_1 = "N/A"
    num_params_phase_2 = "N/A"
    
    writer = SummaryWriter(model_log_dir)
    
    num_params = get_number_of_trainable_parameters(model)
    num_params_total = get_total_parameters(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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
    scaler = GradScaler() if device.type == 'cuda' else None

    attention_based_model = ['prithvi', 'prithvi_unet', 'prithvi2.0_300M', 'prithvi2.0_unet_300M', 'prithvi2.0_600M', 'prithvi2.0_unet_600M']
    
    if model_name in attention_based_model and args.prithvi_finetune_ratio is not None:
        model.change_prithvi_trainability(False)
        logger.info(f"Prithvi weights frozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")
        num_params_phase_1 = get_number_of_trainable_parameters(model)
    else:
        pass
    
    # Phase 1
    for epoch in range(args.epochs):
        logger.info(f"\n{model_name} - Epoch {epoch+1}/{args.epochs}")
        num_params_phase_1 = get_number_of_trainable_parameters(model)
        
        if model_name in attention_based_model:
            train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device, scaler=scaler)
        else:
            train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device, scaler=None)
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
        


    # if finetune ratio is none then it will be end to end FT (Phase 2 will be disabled)

    # Phase 2
    if model_name in attention_based_model and args.prithvi_finetune_ratio is not None:
        logger.info(f"\nFine-tuning {model_name}")
        
        finetune_epochs = int(args.epochs * args.prithvi_finetune_ratio)
        model.change_prithvi_trainability(True)
        logger.info(f"Prithvi weights unfrozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")
        num_params_phase_2 = get_number_of_trainable_parameters(model)
        
        finetune_lr = args.learning_rate * 0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, finetune_epochs)
        
        for epoch in range(args.epochs, args.epochs + finetune_epochs):
            logger.info(f"\n{model_name} - Fine-tune Epoch {epoch+1}/{args.epochs + finetune_epochs}")
             
            train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device, scaler=scaler)
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
    
    logger.info(f"\n{model_name} - Final Evaluation")
    
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

def get_for_lora(model, model_name):
    if model_name in ['prithvi2.0_300M', 'prithvi2.0_unet_300M', 'prithvi2.0_600M', 'prithvi2.0_unet_600M']:
        lora_config = LoraConfig(
            r=32,   
            lora_alpha=64,  
            target_modules = [
                "attn.qkv",
                "attn.proj",
                "mlp.fc1", 
                "mlp.fc2",
                "decoder_embed",
                "decoder_pred",
            ],
            # lora_dropout=0.1,
            bias="none",
            use_dora=True, 
            task_type=None
        )
        logger.info(f"DoRA Applied to {model_name}")
        model = get_peft_model(model, lora_config)

        model.train()
        
        model.print_trainable_parameters()

    else:
        model = model

    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')
    
    args.version = f"{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_log_dir = f'./logs/bolivia_{args.epochs}E_{args.loss_func.upper()}_P20'
    os.makedirs(base_log_dir, exist_ok=True)
    
    args.num_classes = 2
    args.in_channels = 6
    
    logger.info("Loading datasets...")
    train_loader = get_loader(args.data_path, DatasetType.TRAIN.value, args)
    valid_loader = get_loader(args.data_path, DatasetType.VALID.value, args)
    test_loader = get_loader(args.data_path, DatasetType.TEST.value, args)
    bolivia_loader = get_loader(args.data_path, DatasetType.BOLIVIA.value, args)
    
    if args.prithvi_finetune_ratio == 0:
        args.prithvi_finetune_ratio = None

    models = {
        'unet': UNet(
            in_channels=args.in_channels, 
            out_channels=args.num_classes, 
            unet_encoder_size=args.unet_out_channels
        ),
        'prithvi': PritviSegmenter(
            weights_path='./prithvi/Prithvi_EO_V1_100M.pt', 
            device=device, 
            output_channels=args.num_classes, 
            prithvi_encoder_size=args.prithvi_out_channels,
            trainable=False,
        ),
        'prithvi_unet': PrithviUNet(
            in_channels=args.in_channels, 
            out_channels=args.num_classes, 
            weights_path='./prithvi/Prithvi_EO_V1_100M.pt', 
            device=device, 
            prithvi_encoder_size=args.prithvi_out_channels, 
            unet_encoder_size=args.unet_out_channels, 
            combine_method=args.combine_func, 
            dropout_prob=args.random_dropout_prob
        ),
        # 'prithvi2.0_300M': PritviSegmenter(
        #     weights_path='./prithvi20/Prithvi_EO_V2_300M.pt', 
        #     device=device, 
        #     output_channels=args.num_classes, 
        #     prithvi_encoder_size=args.prithvi_out_channels,
        #     trainable=True,
        # ),
        # 'prithvi2.0_unet_300M': PrithviUNet(
        #     in_channels=args.in_channels, 
        #     out_channels=args.num_classes, 
        #     weights_path='./prithvi20/Prithvi_EO_V2_300M.pt', 
        #     device=device, 
        #     prithvi_encoder_size=args.prithvi_out_channels, 
        #     unet_encoder_size=args.unet_out_channels, 
        #     combine_method=args.combine_func, 
        #     dropout_prob=args.random_dropout_prob
        # )
    }
    
    results = []
    for model_name, model in models.items():

        model = get_for_lora(model, model_name)
        # print(model)
        model = model.to(device)
        result = train_single_model(
            model_name, model, train_loader, valid_loader, test_loader, bolivia_loader, 
            args, device, base_log_dir
        )
        results.append(result)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results_file = os.path.join(base_log_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=float)
    logger.info(f"\nResults saved to: {results_file}")
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)