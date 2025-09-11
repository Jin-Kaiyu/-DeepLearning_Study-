# train_mnist.py
"""
基于PyTorch的MNIST手写数字分类器
实现一个完整的深度学习项目流程：数据加载、模型构建、训练和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 定义简单的CNN网络结构
class SimpleNet(nn.Module):
    def __init__(self):
        """
        初始化SimpleNet网络结构
        包含两个卷积层和两个全连接层
        """
        super(SimpleNet, self).__init__()
        # 第一个卷积层：输入通道1（灰度图），输出通道32，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # 最大池化层：2x2窗口，步长2，将28x28降采样为14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层：输入通道32，输出通道64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # 再次池化，将14x14降采样为7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层1：输入64*7*7=3136个特征，输出128个特征
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        # 全连接层2：输出10个类别（数字0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播过程
        Args:
            x: 输入图像张量 [batch_size, 1, 28, 28]
        Returns:
            输出分类结果 [batch_size, 10]
        """
        # 第一层卷积 → ReLU → 池化
        x = self.pool1(self.relu1(self.conv1(x)))
        # 第二层卷积 → ReLU → 池化
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平特征图为一维向量
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层 → ReLU
        x = self.relu3(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

def main():
    """主函数：执行完整的训练和评估流程"""
    
    # 设置随机种子以保证结果可重现
    torch.manual_seed(42)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.ToTensor(),          # 将PIL图像转换为Tensor，并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    print("正在下载和加载MNIST数据集...")
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,   # 训练时打乱数据
        num_workers=2   # 使用2个子进程加载数据
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,  # 测试时不需要打乱
        num_workers=2
    )
    
    # 初始化模型、损失函数和优化器
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("开始训练...")
    
    # 记录训练过程中的损失和准确率
    train_losses = []
    test_accuracies = []
    
    # 训练循环
    num_epochs = 5
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
            
            # 训练五步法
            optimizer.zero_grad()          # 1. 梯度清零
            outputs = model(images)        # 2. 前向传播
            loss = criterion(outputs, labels)  # 3. 计算损失
            loss.backward()                # 4. 反向传播
            optimizer.step()               # 5. 更新参数
            
            running_loss += loss.item()
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 评估阶段
        accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)
        
        # 打印epoch结果
        print(f'Epoch [{epoch+1}/{num_epochs}] 完成, '
              f'训练损失: {epoch_loss:.4f}, '
              f'测试准确率: {accuracy:.2%}')
        print('-' * 60)
    
    # 计算总训练时间
    training_time = time.time() - start_time
    print(f'训练完成! 总耗时: {training_time:.2f}秒')
    print(f'最终测试准确率: {test_accuracies[-1]:.2%}')
    
    # 可视化训练过程
    visualize_training(train_losses, test_accuracies)
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print("模型已保存为 'mnist_cnn_model.pth'")

def evaluate_model(model, test_loader, device):
    """
    评估模型在测试集上的准确率
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
    Returns:
        测试准确率
    """
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def visualize_training(train_losses, test_accuracies):
    """
    可视化训练过程中的损失和准确率变化
    Args:
        train_losses: 训练损失列表
        test_accuracies: 测试准确率列表
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()