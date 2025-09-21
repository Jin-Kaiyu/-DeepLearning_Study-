"""
训练U-Net模型用于PASCAL VOC 2012语义分割任务
修复版本 - 添加IoU和Dice系数评估

作者: Kayu J
日期: 2025年9月16日
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from unet_model import SimpleUNet, UNet
from voc_seg_dataset2 import VOCSegmentationDataset

# ==================== 参数设置区域 ====================
DATA_DIR = "VOC2012"  # VOC数据集路径（修改为相对路径）
BATCH_SIZE = 2                # 批次大小
NUM_WORKERS = 0                 # 数据加载工作进程数

# 模型相关参数
N_CLASSES = 21                    # 类别数量

# 训练相关参数
EPOCHS = 120                      # 训练轮数（对照实验改为120）
LEARNING_RATE = 1e-4              # 学习率（提高初始学习率以突破瓶颈）
WEIGHT_DECAY = 1e-5               # 权重衰减
LOSS_TYPE = "combined"            # 损失函数类型（对照：0.5*CE+0.5*Dice）

# 优化器参数
OPTIMIZER_TYPE = "adam"           # 优化器
MODEL_TYPE = "unet"               # 模型类型："simple" 或 "unet"
SCHEDULER_TYPE = "multistep"  # 学习率调度："cosine" | "cosine_warm_restarts" | "onecycle" | "step" | "multistep" | "plateau"

# 设备设置
USE_CUDA = True                  # 是否使用GPU

# 日志和保存
LOG_DIR = "./runs"                # TensorBoard日志目录
SAVE_MODEL = True                 # 是否保存模型
SAVE_INTERVAL = 5                 # 保存间隔(epoch)

# 数据增强
USE_AUGMENTATION = True           # 是否使用数据增强

# 图像尺寸设置
TARGET_SIZE = (256, 256)

# 可选：限制每个epoch的批次数以便快速实验
MAX_TRAIN_BATCHES = None  # e.g., 30
MAX_VAL_BATCHES = None    # e.g., 30

# 学习率调度额外参数（用于 step/multistep/plateau）
STEP_SIZE = 20
GAMMA = 0.1
MILESTONES = [60, 120, 160]
PLATEAU_PATIENCE = 10
PLATEAU_FACTOR = 0.1
PLATEAU_MIN_LR = 1e-6

# ==================== 评估指标函数 ====================

def calculate_iou(pred, target, n_classes=21, ignore_index=255):
    """
    计算IoU（交并比） - 修复版本
    正确处理ignore_index
    """
    ious = []
    pred = torch.argmax(pred, dim=1)
    
    # 创建有效像素的mask
    valid_mask = (target != ignore_index)
    
    for cls in range(n_classes):
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # 没有该类别，返回NaN
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)  # 忽略NaN计算均值

def calculate_dice(pred, target, n_classes=21, ignore_index=255):
    """
    计算Dice系数 - 修复版本
    正确处理ignore_index
    """
    dices = []
    pred = torch.argmax(pred, dim=1)
    
    # 创建有效像素的mask
    valid_mask = (target != ignore_index)
    
    for cls in range(n_classes):
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask
        
        intersection = (pred_inds & target_inds).sum().float()
        total = pred_inds.sum().float() + target_inds.sum().float()
        
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2 * intersection / total).item())
    
    return np.nanmean(dices)

def calculate_pixel_accuracy(pred, target, ignore_index=255):
    """计算像素精度"""
    pred = torch.argmax(pred, dim=1)
    valid_pixels = (target != ignore_index)
    
    if valid_pixels.sum() == 0:
        return 0.0
    
    correct_pixels = (pred[valid_pixels] == target[valid_pixels]).sum().float()
    total_pixels = valid_pixels.sum().float()
    
    return (correct_pixels / total_pixels).item()

# ==================== 训练函数 ====================

def dice_loss(pred, target, smooth=1.0, ignore_index=255):
    """
    Dice损失函数 - 修复版本
    支持ignore_index，正确处理VOC数据集中的255值
    """
    # 应用softmax获得概率分布
    pred = torch.softmax(pred, dim=1)
    
    # 创建有效像素的mask（排除ignore_index）
    valid_mask = (target != ignore_index)
    
    # 将ignore_index的值临时设为0，避免one_hot编码出错
    target_masked = target.clone()
    target_masked[~valid_mask] = 0
    
    # 确保target_masked是长整型
    target_masked = target_masked.long()
    
    # 创建one-hot编码
    target_one_hot = torch.nn.functional.one_hot(target_masked, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
    
    # 将无效像素的one-hot编码设为0
    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(target_one_hot)
    target_one_hot = target_one_hot * valid_mask_expanded.float()
    
    # 同样将预测值中对应无效像素的部分设为0
    pred_masked = pred * valid_mask.unsqueeze(1).expand_as(pred).float()
    
    # 计算Dice系数
    intersection = torch.sum(pred_masked * target_one_hot, dim=(2, 3))
    union = torch.sum(pred_masked, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def create_loss_function(loss_type):
    """创建损失函数"""
    if loss_type == "dice":
        def dice_ce_loss(pred, target):
            # 纯 Dice 路线不混入 CE；为保持 general 脚本可选择，此处提供 dice 路线
            dice_val = dice_loss(pred, target, ignore_index=255)
            return dice_val
        return dice_ce_loss
    elif loss_type == "combined":
        def combined_loss(pred, target):
            ce_loss = nn.CrossEntropyLoss(ignore_index=255)(pred, target)
            dice_val = dice_loss(pred, target, ignore_index=255)
            return 0.5 * ce_loss + 0.5 * dice_val
        return combined_loss
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

def create_optimizer(model, optimizer_type, lr, weight_decay):
    """创建优化器"""
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("训练配置参数:")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"损失函数: {LOSS_TYPE}")
    print(f"优化器: {OPTIMIZER_TYPE}")
    print(f"模型类型: {MODEL_TYPE}")
    print(f"调度器: {SCHEDULER_TYPE}")
    print(f"使用GPU: {USE_CUDA and torch.cuda.is_available()}")
    print(f"数据增强: {USE_AUGMENTATION}")
    print(f"目标尺寸: {TARGET_SIZE}")
    print("=" * 60)

def train_model():
    """训练U-Net模型"""
    
    # 打印配置
    print_config()
    
     # 修复设备设置逻辑！
    USE_CUDA = True  # 确保这个变量是True
    if torch.cuda.is_available() and USE_CUDA:
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔢 CUDA版本: {torch.version.cuda}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device('cpu')
        print("⚠️  使用CPU训练 - 检查CUDA配置")
    
    print(f"最终设备: {device}")
    
    # 创建数据集和数据加载器
    print("正在加载数据集...")
    try:
        train_dataset = VOCSegmentationDataset(
            root_dir=DATA_DIR,
            split='train',
            apply_augmentation=USE_AUGMENTATION,
            target_size=TARGET_SIZE
        )
        
        val_dataset = VOCSegmentationDataset(
            root_dir=DATA_DIR,
            split='val',
            apply_augmentation=False,
            target_size=TARGET_SIZE
        )
        
        print(f"训练集: {len(train_dataset)} 图像")
        print(f"验证集: {len(val_dataset)} 图像")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    # 初始化模型
    print("正在初始化模型...")
    if MODEL_TYPE == "unet":
        model = UNet(n_channels=3, n_classes=N_CLASSES).to(device)
    else:
        model = SimpleUNet(n_channels=3, n_classes=N_CLASSES).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = create_loss_function(LOSS_TYPE)
    optimizer = create_optimizer(model, OPTIMIZER_TYPE, LEARNING_RATE, WEIGHT_DECAY)
    if SCHEDULER_TYPE == "cosine_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    elif SCHEDULER_TYPE == "onecycle":
        steps_per_epoch = max(1, len(train_loader))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
    elif SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    elif SCHEDULER_TYPE == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    elif SCHEDULER_TYPE == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    else:
        scheduler = None
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR, f"unet_voc_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    
    # 训练循环
    best_val_loss = float('inf')
    best_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    val_accuracies = []
    
    print("\n开始训练...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                # 梯度裁剪，提升训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # OneCycleLR按step推进
                if scheduler is not None and SCHEDULER_TYPE == "onecycle":
                    scheduler.step()
                
                epoch_train_loss += loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                # 限制训练批次数用于快速实验
                if MAX_TRAIN_BATCHES is not None and (batch_idx + 1) >= MAX_TRAIN_BATCHES:
                    break
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            epoch_val_loss = 0.0
            epoch_iou = 0.0
            epoch_dice = 0.0
            epoch_accuracy = 0.0
            
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(val_loader):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    
                    # 计算损失
                    loss = criterion(outputs, masks)
                    epoch_val_loss += loss.item()
                    
                    # 计算评估指标
                    epoch_iou += calculate_iou(outputs, masks, N_CLASSES)
                    epoch_dice += calculate_dice(outputs, masks, N_CLASSES)
                    epoch_accuracy += calculate_pixel_accuracy(outputs, masks)
                    # 限制验证批次数用于快速实验
                    if MAX_VAL_BATCHES is not None and (batch_idx + 1) >= MAX_VAL_BATCHES:
                        break
            
            # 计算平均指标
            avg_val_loss = epoch_val_loss / len(val_loader)
            avg_iou = epoch_iou / len(val_loader)
            avg_dice = epoch_dice / len(val_loader)
            avg_accuracy = epoch_accuracy / len(val_loader)
            
            val_losses.append(avg_val_loss)
            val_ious.append(avg_iou)
            val_dices.append(avg_dice)
            val_accuracies.append(avg_accuracy)
            
            # 学习率调整（非OneCycle按epoch推进）
            if scheduler is not None and SCHEDULER_TYPE != "onecycle":
                if SCHEDULER_TYPE == "cosine_warm_restarts":
                    scheduler.step(epoch + 1)
                elif SCHEDULER_TYPE == "plateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Metrics/IoU', avg_iou, epoch)
            writer.add_scalar('Metrics/Dice', avg_dice, epoch)
            writer.add_scalar('Metrics/Accuracy', avg_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss and SAVE_MODEL:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_iou': avg_iou,
                    'val_dice': avg_dice,
                    'val_accuracy': avg_accuracy,
                    'config': {
                        'batch_size': BATCH_SIZE,
                        'learning_rate': LEARNING_RATE,
                        'loss_type': LOSS_TYPE
                    }
                }, os.path.join(log_dir, 'best_model.pth'))
                print(f"✅ 保存最佳模型，验证损失: {best_val_loss:.4f}")
            
            if avg_iou > best_iou:
                best_iou = avg_iou
            
            # 定期保存
            if SAVE_MODEL and (epoch + 1) % SAVE_INTERVAL == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(log_dir, f'model_epoch_{epoch+1}.pth'))
            
            # 打印训练信息
            print(f'Epoch {epoch+1}/{EPOCHS}')
            print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            print(f'IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Accuracy: {avg_accuracy:.4f}')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            print("-" * 80)
            
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存最终模型
    if SAVE_MODEL:
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_ious': val_ious,
            'val_dices': val_dices,
            'val_accuracies': val_accuracies,
        }, os.path.join(log_dir, 'final_model.pth'))
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    writer.close()
    
    print("🎉 训练完成！")
    print(f"⏰ 总训练时间: {hours}h {minutes}m {seconds}s")
    print(f"📊 最佳验证损失: {best_val_loss:.4f}")
    print(f"🎯 最佳IoU: {best_iou:.4f}")
    print(f"💾 日志和模型保存在: {log_dir}")
    print(f"📈 启动TensorBoard: tensorboard --logdir={LOG_DIR}")
    
    # 保存训练结果到文本文件
    with open(os.path.join(log_dir, 'training_results.txt'), 'w') as f:
        f.write("训练结果总结:\n")
        f.write("=" * 50 + "\n")
        f.write(f"总训练时间: {hours}h {minutes}m {seconds}s\n")
        f.write(f"最佳验证损失: {best_val_loss:.4f}\n")
        f.write(f"最佳IoU: {best_iou:.4f}\n")
        f.write(f"最终学习率: {optimizer.param_groups[0]['lr']:.2e}\n")
        f.write("\n每个epoch的指标:\n")
        for i, (tl, vl, iou, dice, acc) in enumerate(zip(train_losses, val_losses, val_ious, val_dices, val_accuracies)):
            f.write(f"Epoch {i+1}: Loss={tl:.4f}/{vl:.4f}, IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}\n")

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train UNet on VOC2012 (loss: combined or dice)")
    parser.add_argument('--epochs', type=int, help='训练轮数覆盖值')
    parser.add_argument('--lr', type=float, help='学习率覆盖值')
    parser.add_argument('--batch-size', type=int, help='批次大小覆盖值')
    parser.add_argument('--data-dir', type=str, help='VOC2012 数据目录覆盖值')
    parser.add_argument('--scheduler', type=str, choices=['cosine','cosine_warm_restarts','onecycle','step','multistep','plateau'], help='学习率调度器')
    parser.add_argument('--model', type=str, choices=['unet','simple'], help='模型类型')
    parser.add_argument('--loss', type=str, choices=['combined','dice'], help='损失函数类型')
    parser.add_argument('--no-aug', action='store_true', help='禁用数据增强')
    parser.add_argument('--max-train-batches', type=int, help='每个epoch的训练批次数上限')
    parser.add_argument('--max-val-batches', type=int, help='每个epoch的验证批次数上限')
    # 调度器参数
    parser.add_argument('--step-size', type=int, help='StepLR 的 step_size')
    parser.add_argument('--gamma', type=float, help='StepLR/MultiStepLR 的 gamma 衰减因子')
    parser.add_argument('--milestones', type=str, help='MultiStepLR 的里程碑，例如 60,120,160')
    parser.add_argument('--plateau-patience', type=int, help='ReduceLROnPlateau 的 patience')
    parser.add_argument('--plateau-factor', type=float, help='ReduceLROnPlateau 的 factor')
    parser.add_argument('--plateau-min-lr', type=float, help='ReduceLROnPlateau 的 min_lr')
    args = parser.parse_args()

    # 覆盖全局配置（如提供）
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.lr is not None:
        LEARNING_RATE = args.lr
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    if args.scheduler is not None:
        SCHEDULER_TYPE = args.scheduler
    if args.model is not None:
        MODEL_TYPE = args.model
    if hasattr(args, 'loss') and args.loss is not None:
        LOSS_TYPE = args.loss
    if args.no_aug:
        USE_AUGMENTATION = False
    if args.max_train_batches is not None:
        MAX_TRAIN_BATCHES = args.max_train_batches
    if args.max_val_batches is not None:
        MAX_VAL_BATCHES = args.max_val_batches
    # 覆盖调度器参数
    if hasattr(args, 'step_size') and args.step_size is not None:
        STEP_SIZE = args.step_size
    if hasattr(args, 'gamma') and args.gamma is not None:
        GAMMA = args.gamma
    if hasattr(args, 'milestones') and args.milestones:
        try:
            MILESTONES = [int(x.strip()) for x in args.milestones.split(',') if x.strip()]
        except Exception:
            print("警告: 解析 --milestones 失败，保持默认值:", MILESTONES)
    if hasattr(args, 'plateau_patience') and args.plateau_patience is not None:
        PLATEAU_PATIENCE = args.plateau_patience
    if hasattr(args, 'plateau_factor') and args.plateau_factor is not None:
        PLATEAU_FACTOR = args.plateau_factor
    if hasattr(args, 'plateau_min_lr') and args.plateau_min_lr is not None:
        PLATEAU_MIN_LR = args.plateau_min_lr

    # 检查数据集路径
    if not os.path.exists(DATA_DIR):
        print("=" * 80)
        print("❌ 错误: 数据集路径不存在!")
        print("=" * 80)
        print(f"期望路径: {os.path.abspath(DATA_DIR)}")
        print()
        print("请按照以下步骤准备PASCAL VOC 2012数据集:")
        print("1. 从官方网站下载VOC2012数据集:")
        print("   http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
        print()
        print("2. 下载并解压以下文件:")
        print("   - VOCtrainval_11-May-2012.tar (训练和验证数据)")
        print()
        print("3. 将解压后的VOC2012文件夹放在当前目录下，确保具有以下结构:")
        print("   ./VOC2012/")
        print("   ├── JPEGImages/          # 原始图像")
        print("   ├── SegmentationClass/   # 分割标注")
        print("   └── ImageSets/")
        print("       └── Segmentation/    # 数据分割文件")
        print("           ├── train.txt")
        print("           ├── val.txt")
        print("           └── trainval.txt")
        print()
        print("4. 或者修改train_unet_voc.py中的DATA_DIR变量指向正确的数据集路径")
        print("=" * 80)
        print()
        print("💡 提示: 如果您想要快速测试代码，可以创建一个小型演示数据集")
        print("   或使用其他公开可用的语义分割数据集。")
        print("=" * 80)
    else:
        train_model()