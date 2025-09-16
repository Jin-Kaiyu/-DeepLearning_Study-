"""
PASCAL VOC 2012 语义分割数据集自定义Dataset类
实现图像和分割mask的加载，支持数据增强且保持同步

作者: Kayu J
日期: 2025年9月16日
课程: 目标检测与语义分割
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
import random

# 尝试导入PyTorch，如果失败则使用简化版本
try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
    print("✅ PyTorch已导入")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch未安装，使用简化版本")
    
    # 定义简单的Dataset基类
    class Dataset:
        def __init__(self):
            pass
        
        def __len__(self):
            raise NotImplementedError
        
        def __getitem__(self, idx):
            raise NotImplementedError

class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 语义分割数据集
    
    数据集结构:
    VOC2012/
    ├── JPEGImages/          # 原始图像
    ├── SegmentationClass/   # 语义分割标签(每个像素标注类别)
    ├── SegmentationObject/  # 实例分割标签
    └── ImageSets/
        └── Segmentation/
            ├── train.txt    # 训练集图像名称列表
            ├── val.txt      # 验证集图像名称列表
            └── trainval.txt # 训练+验证集
    """
    
    # PASCAL VOC 2012 类别定义（21类：20个物体类别 + 1个背景）
    CLASSES = [
        'background',       # 0
        'aeroplane',        # 1
        'bicycle',          # 2
        'bird',             # 3
        'boat',             # 4
        'bottle',           # 5
        'bus',              # 6
        'car',              # 7
        'cat',              # 8
        'chair',            # 9
        'cow',              # 10
        'diningtable',      # 11
        'dog',              # 12
        'horse',            # 13
        'motorbike',        # 14
        'person',           # 15
        'pottedplant',      # 16
        'sheep',            # 17
        'sofa',             # 18
        'train',            # 19
        'tvmonitor'         # 20
    ]
    
    # 类别对应的可视化颜色(RGB)
    CLASS_COLORS = [
        (0, 0, 0),          # background - 黑色
        (128, 0, 0),        # aeroplane - 深红
        (0, 128, 0),        # bicycle - 绿色
        (128, 128, 0),      # bird - 橄榄色
        (0, 0, 128),        # boat - 蓝色
        (128, 0, 128),      # bottle - 紫色
        (0, 128, 128),      # bus - 青色
        (128, 128, 128),    # car - 灰色
        (64, 0, 0),         # cat - 暗红
        (192, 0, 0),        # chair - 亮红
        (64, 128, 0),       # cow - 黄绿
        (192, 128, 0),      # diningtable - 橙色
        (64, 0, 128),       # dog - 蓝紫
        (192, 0, 128),      # horse - 粉红
        (64, 128, 128),     # motorbike - 蓝绿
        (192, 128, 128),    # person - 浅灰
        (0, 64, 0),         # pottedplant - 深绿
        (128, 64, 0),       # sheep - 棕色
        (0, 192, 0),        # sofa - 亮绿
        (128, 192, 0),      # train - 黄绿2
        (0, 64, 128)        # tvmonitor - 深蓝
    ]
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform: Optional[object] = None,
                 target_transform: Optional[object] = None,
                 apply_augmentation: bool = True):
        """
        初始化VOC分割数据集
        
        Args:
            root_dir: VOC2012数据集根目录路径
            split: 数据集分割 ('train', 'val', 'trainval')
            transform: 应用于图像的变换
            target_transform: 应用于mask的变换
            apply_augmentation: 是否应用数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.apply_augmentation = apply_augmentation
        
        # 数据路径
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
        self.splits_dir = os.path.join(root_dir, 'ImageSets', 'Segmentation')
        
        # 检查路径是否存在
        self._check_paths()
        
        # 加载图像列表
        self.image_names = self._load_image_list()
        
        print(f"VOC Segmentation Dataset loaded: {len(self.image_names)} images in {split} split")
    
    def _check_paths(self):
        """检查必要的路径是否存在"""
        required_paths = [self.images_dir, self.masks_dir, self.splits_dir]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")
    
    def _load_image_list(self) -> List[str]:
        """加载指定split的图像名称列表"""
        split_file = os.path.join(self.splits_dir, f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        return image_names
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_names)
    
    def __getitem__(self, idx: int):
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            如果PyTorch可用: tuple: (image_tensor, mask_tensor)
            如果PyTorch不可用: dict: {'image': numpy_array, 'mask': numpy_array, 'name': str}
        """
        # 获取图像名称
        img_name = self.image_names[idx]
        
        # 构建文件路径
        img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_name}.png')
        
        # 加载图像和mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')  # P模式保持调色板
        
        # 应用同步的数据增强
        if self.apply_augmentation and self.split == 'train' and TORCH_AVAILABLE:
            image, mask = self._apply_synchronized_augmentation(image, mask)
        
        if TORCH_AVAILABLE:
            # PyTorch版本：返回张量
            # 应用变换
            if self.transform:
                image = self.transform(image)
            else:
                # 默认变换：转换为张量并归一化
                image = TF.to_tensor(image)
                image = TF.normalize(image, mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet标准化
            
            if self.target_transform:
                mask = self.target_transform(mask)
            else:
                # 默认变换：转换为张量，保持类别标签
                mask = torch.from_numpy(np.array(mask)).long()
            
            return image, mask
        else:
            # 简化版本：返回numpy数组和字典
            return {
                'name': img_name,
                'image': np.array(image),
                'mask': np.array(mask)
            }
    
    def _apply_synchronized_augmentation(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        应用同步的数据增强，确保图像和mask保持一致
        
        Args:
            image: PIL图像
            mask: PIL mask
            
        Returns:
            增强后的图像和mask
        """
        if not TORCH_AVAILABLE:
            # 如果PyTorch不可用，直接返回原图像
            return image, mask
            
        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 随机旋转 (-10度到10度)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=255)  # 255是ignore_index
        
        # 随机缩放和裁剪
        if random.random() > 0.5:
            # 原始尺寸
            w, h = image.size
            
            # 随机缩放比例 (0.8 - 1.2)
            scale = random.uniform(0.8, 1.2)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 缩放
            image = TF.resize(image, (new_h, new_w))
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)
            
            # 如果放大了，进行随机裁剪
            if scale > 1.0:
                # 随机裁剪回原始尺寸
                i = random.randint(0, new_h - h)
                j = random.randint(0, new_w - w)
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
            else:
                # 如果缩小了，padding到原始尺寸
                pad_h = h - new_h
                pad_w = w - new_w
                padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
                image = TF.pad(image, padding, fill=0)
                mask = TF.pad(mask, padding, fill=255)
        
        return image, mask
    
    def get_class_weights(self):
        """
        计算类别权重，用于处理类别不平衡问题
        
        Returns:
            如果PyTorch可用: torch.Tensor
            如果PyTorch不可用: numpy.ndarray
        """
        print("正在计算类别权重...")
        
        if TORCH_AVAILABLE:
            class_counts = torch.zeros(len(self.CLASSES))
        else:
            class_counts = np.zeros(len(self.CLASSES))
        
        for idx in range(len(self)):
            data = self.__getitem__(idx)
            
            if TORCH_AVAILABLE:
                _, mask = data
                unique, counts = torch.unique(mask, return_counts=True)
            else:
                mask = data['mask']
                unique, counts = np.unique(mask, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                if class_id < len(self.CLASSES):  # 忽略255等特殊值
                    class_counts[class_id] += count
        
        # 计算权重 (总像素数 / (类别数 * 每个类别的像素数))
        if TORCH_AVAILABLE:
            total_pixels = class_counts.sum()
            weights = total_pixels / (len(self.CLASSES) * class_counts)
            weights[class_counts == 0] = 0  # 避免除零
        else:
            total_pixels = class_counts.sum()
            weights = np.divide(total_pixels, (len(self.CLASSES) * class_counts), 
                               out=np.zeros_like(class_counts), where=class_counts!=0)
        
        return weights
    
    def visualize_sample(self, idx: int) -> Dict:
        """
        可视化指定样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含原始图像、mask和叠加图像的字典
        """
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_name}.png')
        
        # 加载原始图像和mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')
        
        # 转换mask为彩色
        mask_array = np.array(mask)
        colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.CLASS_COLORS):
            colored_mask[mask_array == class_id] = color
        
        return {
            'image_name': img_name,
            'image': np.array(image),
            'mask': mask_array,
            'colored_mask': colored_mask,
            'classes_present': np.unique(mask_array)
        }
    
    @staticmethod
    def get_class_name(class_id: int) -> str:
        """根据类别ID获取类别名称"""
        if 0 <= class_id < len(VOCSegmentationDataset.CLASSES):
            return VOCSegmentationDataset.CLASSES[class_id]
        return 'unknown'
    
    def get_dataset_statistics(self) -> Dict:
        """获取数据集统计信息"""
        print("正在计算数据集统计信息...")
        
        stats = {
            'total_images': len(self.image_names),
            'split': self.split,
            'num_classes': len(self.CLASSES),
            'classes': self.CLASSES,
            'image_sizes': [],
            'class_distribution': np.zeros(len(self.CLASSES)) if not TORCH_AVAILABLE else torch.zeros(len(self.CLASSES))
        }
        
        # 采样部分图像进行统计（避免太慢）
        sample_size = min(100, len(self.image_names))
        sample_indices = random.sample(range(len(self.image_names)), sample_size)
        
        for idx in sample_indices:
            img_name = self.image_names[idx]
            img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
            mask_path = os.path.join(self.masks_dir, f'{img_name}.png')
            
            # 图像尺寸
            with Image.open(img_path) as img:
                stats['image_sizes'].append(img.size)
            
            # 类别分布
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask)
                unique_classes = np.unique(mask_array)
                for class_id in unique_classes:
                    if class_id < len(self.CLASSES):
                        stats['class_distribution'][class_id] += 1
        
        return stats


def create_data_loaders(root_dir: str, 
                       batch_size: int = 4,
                       num_workers: int = 2,
                       pin_memory: bool = True):
    """
    创建训练和验证数据加载器（仅在PyTorch可用时）
    
    Args:
        root_dir: VOC2012数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存
        
    Returns:
        如果PyTorch可用: (train_loader, val_loader)
        如果PyTorch不可用: None
    """
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch不可用，无法创建DataLoader")
        return None
        
    # 创建数据集
    train_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        split='train',
        apply_augmentation=True
    )
    
    val_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        split='val',
        apply_augmentation=False
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据集
    root_dir = r"VOC2012"
    
    print(f"🧪 测试VOC分割数据集...")
    print(f"PyTorch可用: {TORCH_AVAILABLE}")
    
    try:
        # 创建数据集实例
        dataset = VOCSegmentationDataset(root_dir, split='train', apply_augmentation=False)
        
        print(f"数据集大小: {len(dataset)}")
        print(f"类别数: {len(dataset.CLASSES)}")
        print(f"类别: {dataset.CLASSES[:10]}...")  # 只显示前10个类别
        
        # 测试数据加载
        data = dataset[0]
        
        if TORCH_AVAILABLE:
            image, mask = data
            print(f"图像形状: {image.shape}")
            print(f"Mask形状: {mask.shape}")
            print(f"Mask中的类别: {torch.unique(mask)}")
        else:
            print(f"图像名称: {data['name']}")
            print(f"图像形状: {data['image'].shape}")
            print(f"Mask形状: {data['mask'].shape}")
            print(f"Mask中的类别: {np.unique(data['mask'])}")
        
        print("✅ 数据集测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查数据集路径是否正确")
