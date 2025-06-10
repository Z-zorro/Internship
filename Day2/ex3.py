import torch
from torch import nn

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=4),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 192, kernel_size=3),
            nn.Conv2d(192, 192, kernel_size=3),
            nn.Conv2d(192, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10)
        )
        
    def forward(self, x):
        x = self.model(x)

        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    alexnet = alex()
    y = alexnet(x)
    print(y.shape)