import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.5):
        super().__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 减小通道数
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/4),
            
            # 第二个卷积块
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),  # 减小通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/4),
            
            # 第三个卷积块
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # 减小通道数
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate/4),
            
            # 第四个卷积块
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # 减小通道数
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate/4),
            
            # 第五个卷积块
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),  # 减小通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/4),
            
            # 自适应池化层
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # 分类器层 - 减小全连接层大小
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 6 * 6, 2048),  # 减小神经元数量
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),  # 减小神经元数量
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            
            nn.Linear(2048, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 测试模型
    model = AlexNet(num_classes=100)
    input = torch.randn(32, 3, 224, 224)  # 使用较小的批次大小
    output = model(input)
    print("输入形状:", input.shape)
    print("输出形状:", output.shape)
    print("输出类别数:", output.size(1))
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 估算显存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.cuda()
        input = input.cuda()
        output = model(input)
        print(f"峰值显存使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")