"""
使用Faster R-CNN训练PASCAL VOC目标检测任务

作者: DeepLearning_Study
日期: 2025年9月
课程: 工业表面划痕检测研究 - 第5周
"""

import os
from collections import defaultdict
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
import torch.optim as optim
import time
import numpy as np
from datetime import datetime
from voc_detection_dataset import VOCDetectionDataset, create_voc_data_loaders

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"使用设备: {device}")


def create_faster_rcnn_model(num_classes: int, pretrained: bool = True):
    """
    创建Faster R-CNN模型
    
    Args:
        num_classes: 类别数量（包括背景）
        pretrained: 是否使用预训练权重
        
    Returns:
        Faster R-CNN模型
    """
    if pretrained:
        # 使用torchvision预训练的Faster R-CNN (修复deprecated警告)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # 替换分类头以适应VOC的类别数
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
    else:
        # 从头开始训练
        backbone = torchvision.models.mobilenet_v2(pretrained=False).features
        backbone.out_channels = 1280
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += losses.item()
        loss_classifier += loss_dict.get('loss_classifier', 0).item()
        loss_box_reg += loss_dict.get('loss_box_reg', 0).item()
        loss_objectness += loss_dict.get('loss_objectness', 0).item()
        loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 0).item()
        
        if i % print_freq == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] '
                  f'Loss: {losses.item():.4f} '
                  f'Time: {elapsed_time:.2f}s')
            start_time = time.time()
    
    # 计算平均损失
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_loss_classifier = loss_classifier / num_batches
    avg_loss_box_reg = loss_box_reg / num_batches
    avg_loss_objectness = loss_objectness / num_batches
    avg_loss_rpn_box_reg = loss_rpn_box_reg / num_batches
    
    return {
        'total_loss': avg_loss,
        'loss_classifier': avg_loss_classifier,
        'loss_box_reg': avg_loss_box_reg,
        'loss_objectness': avg_loss_objectness,
        'loss_rpn_box_reg': avg_loss_rpn_box_reg
    }



@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold: float = 0.5):
    """在验证集上评估模型并计算VOC风格的mAP (IoU=0.5)."""
    model.eval()
    all_detections = []
    all_ground_truths = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        outputs_cpu = [{k: v.detach().cpu() for k, v in output.items()} for output in outputs]
        targets_cpu = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()} for target in targets]

        all_detections.extend(outputs_cpu)
        all_ground_truths.extend(targets_cpu)

    class_names = getattr(data_loader.dataset, 'CLASSES', None)
    map_value, per_class_ap = calculate_map(all_detections, all_ground_truths, class_names=class_names, iou_threshold=iou_threshold)

    if per_class_ap and class_names:
        print("验证集AP (IoU=0.5):")
        for class_id, ap in sorted(per_class_ap.items()):
            if class_id == 0:
                continue
            name = class_names[class_id - 1] if 1 <= class_id <= len(class_names) else str(class_id)
            print(f"  {name:<15}: {ap:.4f}")

    return map_value


def calculate_map(detections, ground_truths, class_names=None, iou_threshold: float = 0.5):
    """根据检测结果和标注计算mAP及每类AP."""
    per_class_stats = defaultdict(lambda: {"scores": [], "matches": [], "num_gt": 0})

    for det, gt in zip(detections, ground_truths):
        gt_boxes = gt.get('boxes')
        gt_labels = gt.get('labels')
        if gt_boxes is None or gt_labels is None:
            continue

        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu()
        if torch.is_tensor(gt_labels):
            gt_labels = gt_labels.cpu()

        for label in gt_labels.tolist():
            if label == 0:
                continue
            per_class_stats[label]["num_gt"] += 1

        matched = defaultdict(set)

        if len(det.get('boxes', [])) == 0:
            continue

        pred_boxes = det['boxes']
        pred_labels = det['labels']
        pred_scores = det['scores']

        if torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.cpu()
        if torch.is_tensor(pred_labels):
            pred_labels = pred_labels.cpu()
        if torch.is_tensor(pred_scores):
            pred_scores = pred_scores.cpu()

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            label = int(label.item()) if hasattr(label, 'item') else int(label)
            if label == 0:
                continue
            score = float(score.item()) if hasattr(score, 'item') else float(score)
            per_class_stats[label]["scores"].append(score)

            candidate_indices = torch.where(gt_labels == label)[0]
            if len(candidate_indices) == 0:
                per_class_stats[label]["matches"].append(0)
                continue

            ious = box_iou(box.unsqueeze(0), gt_boxes[candidate_indices]).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            gt_index = candidate_indices[best_idx].item()

            if best_iou >= iou_threshold and gt_index not in matched[label]:
                per_class_stats[label]["matches"].append(1)
                matched[label].add(gt_index)
            else:
                per_class_stats[label]["matches"].append(0)

    ap_per_class = {}
    for label, stats in per_class_stats.items():
        num_gt = stats["num_gt"]
        if num_gt == 0:
            continue

        scores = np.array(stats["scores"], dtype=np.float32)
        matches = np.array(stats["matches"], dtype=np.float32)

        if scores.size == 0:
            ap_per_class[label] = 0.0
            continue

        order = np.argsort(-scores)
        matches = matches[order]

        tps = matches
        fps = 1.0 - matches

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)

        recalls = cum_tp / max(num_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

        ap = 0.0
        for recall_threshold in np.linspace(0.0, 1.0, 11):
            precisions_at_recall = precisions[recalls >= recall_threshold]
            precision = np.max(precisions_at_recall) if precisions_at_recall.size > 0 else 0.0
            ap += precision
        ap /= 11.0

        ap_per_class[label] = float(ap)

    if ap_per_class:
        map_value = float(np.mean(list(ap_per_class.values())))
    else:
        map_value = 0.0

    return map_value, ap_per_class



def main():
    """主训练函数"""
    # 超参数
    num_epochs = 10
    batch_size = 4  # 增大batch size提高训练效率
    learning_rate = 0.005
    num_classes = 21  # VOC的20个类别 + 背景
    
    # 数据路径
    root_dir = "/data/wf/Conference_paper_guidance/Codes/Kayu J/VOC2012"  # 根据你的实际路径修改
    
    print("🚀 开始训练Faster R-CNN on VOC...")
    print(f"超参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # 创建数据加载器
    train_loader, val_loader = create_voc_data_loaders(
        root_dir, batch_size=batch_size, num_workers=4)
    
    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    
    # 创建模型
    model = create_faster_rcnn_model(num_classes, pretrained=True)
    model.to(device)
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_map': [],
        'learning_rates': []
    }
    
    # 创建输出目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("🏋️ 开始训练...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 更新学习率
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 评估
        val_map = evaluate(model, val_loader, device)
        
        # 记录历史
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_map'].append(val_map)
        history['learning_rates'].append(current_lr)
        
        # 打印进度
        print(f"Epoch {epoch + 1}结果:")
        print(f"  训练损失: {train_metrics['total_loss']:.4f}")
        print(f"  验证mAP: {val_map:.4f}")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  分类损失: {train_metrics['loss_classifier']:.4f}")
        print(f"  回归损失: {train_metrics['loss_box_reg']:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'history': history
            }
            checkpoint_path = f'checkpoints/faster_rcnn_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = 'checkpoints/faster_rcnn_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ 训练完成! 最终模型已保存: {final_model_path}")
    
    # 打印最终结果
    print(f"\n🎯 最终结果:")
    print(f"最佳训练损失: {min(history['train_loss']):.4f}")
    print(f"最佳验证mAP: {max(history['val_map']):.4f}")
    
    return model, history


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        model, history = main()
        
        # 保存训练历史
        history_path = 'logs/training_history.npy'
        np.save(history_path, history)
        print(f"📊 训练历史已保存: {history_path}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()