"""
PASCAL VOC 目标检测数据集自定义Dataset类
实现图像和边界框标注的加载，用于Faster R-CNN训练

作者: DeepLearning_Study
日期: 2025年9月
课程: 工业表面划痕检测研究 - 第5周
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import List, Dict, Any, Optional


class VOCDetectionDataset(Dataset):
    """
    PASCAL VOC 目标检测数据集类
    
    数据集结构:
    VOC2012/
    ├── JPEGImages/          # 原始图像
    ├── Annotations/         # XML标注文件
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt    # 训练集图像名称列表
    │       └── val.txt      # 验证集图像名称列表
    """
    
    # PASCAL VOC 类别定义（20个物体类别）
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform: Optional[T.Compose] = None,
                 target_transform: Optional[Any] = None):
        """
        初始化VOC检测数据集
        
        Args:
            root_dir: VOC数据集根目录路径
            split: 数据集分割 ('train', 'val', 'trainval')
            transform: 应用于图像的变换
            target_transform: 应用于目标的变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 数据路径
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotations_dir = os.path.join(root_dir, 'Annotations')
        self.splits_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        
        # 检查路径是否存在
        self._check_paths()
        
        # 加载图像列表
        self.image_names = self._load_image_list()
        
        # 创建类别到索引的映射
        self.class_to_idx = {cls_name: idx + 1 for idx, cls_name in enumerate(self.CLASSES)}
        
        print(f"VOC Detection Dataset loaded: {len(self.image_names)} images in {split} split")
        print(f"Classes: {self.CLASSES}")
    
    def _check_paths(self):
        """检查必要的路径是否存在"""
        required_paths = [self.images_dir, self.annotations_dir, self.splits_dir]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")
    
    def _load_image_list(self) -> List[str]:
        """加载指定split的图像名称列表"""
        split_file = os.path.join(self.splits_dir, f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            # 处理可能包含标签信息的行（如 "2007_000027 -1"）
            lines = [line.strip().split()[0] for line in f.readlines()]
        
        return lines
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_names)
    
    def _parse_annotation(self, annotation_path: str) -> Dict[str, Any]:
        """解析XML标注文件"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        labels = []
        difficulties = []
        areas = []
        
        # 解析每个物体
        for obj in root.findall('object'):
            # 获取类别
            cls_name = obj.find('name').text
            if cls_name not in self.class_to_idx:
                continue  # 跳过未知类别
            
            # 获取边界框
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 检查边界框有效性
            if xmin >= xmax or ymin >= ymax:
                continue
            
            # 获取难度标志（如果有）
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            
            # 计算面积
            area = (xmax - xmin) * (ymax - ymin)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
            difficulties.append(difficult)
            areas.append(area)
        
        if not boxes:  # 如果没有有效的物体
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)
            difficulties = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            difficulties = torch.as_tensor(difficulties, dtype=torch.int64)
        
        # 构建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([0]),  # 占位符，会在collate_fn中替换
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'difficulties': difficulties
        }
        
        return target
    
    def __getitem__(self, idx: int):
        """
        获取指定索引的数据样本
        
        Returns:
            image: 图像张量 [C, H, W]
            target: 包含标注信息的字典
        """
        # 获取图像名称
        img_name = self.image_names[idx]
        
        # 构建文件路径
        img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
        annotation_path = os.path.join(self.annotations_dir, f'{img_name}.xml')
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 解析标注
        target = self._parse_annotation(annotation_path)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换：转换为张量
            image = TF.to_tensor(image)
        
        # 添加图像ID
        target['image_id'] = torch.tensor([idx])
        
        return image, target
    
    def collate_fn(self, batch):
        """
        自定义collate函数，用于处理不同数量的目标
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        return images, targets
    
    def get_class_name(self, class_id: int) -> str:
        """根据类别ID获取类别名称"""
        if 1 <= class_id <= len(self.CLASSES):
            return self.CLASSES[class_id - 1]
        return 'background'
    
    def visualize_sample(self, idx: int) -> Dict[str, Any]:
        """
        可视化指定样本（用于调试）
        """
        image, target = self.__getitem__(idx)
        
        return {
            'image': image,
            'target': target,
            'image_name': self.image_names[idx],
            'num_objects': len(target['boxes'])
        }


def create_voc_data_loaders(root_dir: str, 
                           batch_size: int = 2,
                           num_workers: int = 2):
    """
    创建VOC数据加载器
    
    Args:
        root_dir: VOC数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        
    Returns:
        train_loader, val_loader
    """
    # 数据变换
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    # 创建数据集
    train_dataset = VOCDetectionDataset(
        root_dir=root_dir,
        split='train',
        transform=transform
    )
    
    val_dataset = VOCDetectionDataset(
        root_dir=root_dir,
        split='val',
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据集
    root_dir = "VOC2012"  # 根据你的实际路径修改

    
    print("🧪 测试VOC检测数据集...")
    
    try:
        # 创建数据集实例
        dataset = VOCDetectionDataset(root_dir, split='train')
        
        print(f"数据集大小: {len(dataset)}")
        print(f"类别数: {len(dataset.CLASSES)}")
        print(f"类别映射: {dataset.class_to_idx}")
        
        # 测试数据加载
        image, target = dataset[0]
        print(f"图像形状: {image.shape}")
        print(f"目标键值: {list(target.keys())}")
        print(f"边界框数量: {len(target['boxes'])}")
        print(f"标签: {target['labels']}")
        
        print("✅ 数据集测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()