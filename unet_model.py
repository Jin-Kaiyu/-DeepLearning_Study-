"""
U-Net 语义分割模型实现 - 修复版本
解决通道数不匹配问题

作者: Kayu J
日期: 2025年9月20日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => BatchNorm => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样块：最大池化 + 双卷积"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块 - 修复版本"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # 如果是双线性上采样，减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 输入是CHW格式
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # 填充以确保尺寸匹配
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接跳跃连接的特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net 架构实现 - 修复版本"""
    
    def __init__(self, n_channels=3, n_classes=21, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器 (下采样)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # 根据是否使用双线性上采样决定最后的通道数
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器 (上采样)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器 - 保存跳跃连接的特征图
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024//factor, H/16, W/16]
        
        # 解码器 + 跳跃连接
        x = self.up1(x5, x4)  # [B, 512//factor, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256//factor, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128//factor, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        
        # 输出
        logits = self.outc(x) # [B, n_classes, H, W]
        return logits

# 简化版U-Net（如果完整版仍有问题）
class SimpleUNet(nn.Module):
    """简化版U-Net，避免复杂的跳跃连接问题"""
    
    def __init__(self, n_channels=3, n_classes=21):
        super(SimpleUNet, self).__init__()
        
        # 编码器
        self.enc1 = self._block(n_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # 瓶颈层
        self.bottleneck = self._block(512, 1024)
        
        # 解码器（不使用跳跃连接）
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 64)
        
        # 输出层
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)         # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        
        # 瓶颈
        bn = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]
        
        # 解码器（无跳跃连接）
        d4 = self.upconv4(bn)     # [B, 512, H/8, W/8]
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)     # [B, 256, H/4, W/4]
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)     # [B, 128, H/2, W/2]
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)     # [B, 64, H, W]
        d1 = self.dec1(d1)
        
        return self.final(d1)     # [B, n_classes, H, W]

def test_model():
    """测试模型输出形状"""
    print("测试U-Net模型...")
    
    # 测试完整版U-Net
    print("\n1. 测试完整版U-Net:")
    try:
        model = UNet(n_channels=3, n_classes=21)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"完整版U-Net测试失败: {e}")
    
    # 测试简化版U-Net
    print("\n2. 测试简化版U-Net:")
    try:
        model = SimpleUNet(n_channels=3, n_classes=21)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"简化版U-Net测试失败: {e}")

if __name__ == "__main__":
    test_model()