# 搭建神经网络
import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一个卷积层 - 调整参数以适应32x32输入
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 输出: 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16x64
            
            # 第二个卷积层
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 输出: 16x16x192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 8x8x192
            
            # 第三个卷积层
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # 输出: 8x8x384
            nn.ReLU(inplace=True),
            
            # 第四个卷积层
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 输出: 8x8x256
            nn.ReLU(inplace=True),
            
            # 第五个卷积层
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 输出: 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 4x4x256
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),  # 根据最终特征图大小调整
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 测试模型
    model = AlexNet()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print("输入形状:", input.shape)
    print("输出形状:", output.shape)
    
    # 打印每层的输出形状
    x = input
    for i, layer in enumerate(model.features):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"第{i+1}层 {layer.__class__.__name__} 输出形状: {x.shape}")
    
    # 测试分类器部分
    x = torch.flatten(x, 1)
    print(f"展平后形状: {x.shape}")
    for i, layer in enumerate(model.classifier):
        x = layer(x)
        if isinstance(layer, nn.Linear):
            print(f"全连接层 {i+1} 输出形状: {x.shape}")
