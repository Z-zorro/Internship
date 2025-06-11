# DAY3

## 常见激活函数及其特点

| 激活函数       | 数学表达式                                                   | 函数图像               | 优点                                                         | 缺点                                                    | 适用场景                  |
| -------------- | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------- |
| **Sigmoid**    | $f(x) = \frac{1}{1 + e^{-x}}$                                | S形曲线，值域(0,1)     | 1. 输出范围(0,1)，适合概率输出<br>2. 函数光滑可导            | 1. 容易梯度消失<br>2. 输出不以0为中心<br>3. 指数计算慢  | 二分类输出层              |
| **Tanh**       | $f(x) = \tanh(x)$                                            | S形曲线，值域(-1,1)    | 1. 输出以0为中心<br>2. 比Sigmoid梯度更强                     | 1. 仍然存在梯度消失问题<br>2. 指数计算慢                | RNN/LSTM隐藏层            |
| **ReLU**       | $f(x) = \max(0, x)$                                          | x<0时为0，x≥0时为x     | 1. 计算效率高<br>2. 缓解梯度消失问题<br>3. 加速收敛          | 1. "Dead ReLU"问题(负输入梯度为0)<br>2. 输出不以0为中心 | CNN/深度网络隐藏层        |
| **Leaky ReLU** | $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$ | 负区有小的斜率         | 1. 解决Dead ReLU问题<br>2. 保留ReLU优点                      | 1. 结果不一致(α需人工设定)<br>2. 负区梯度小             | 需要避免神经元死亡的场景  |
| **PReLU**      | 类似Leaky ReLU，但α可学习                                    | 负区斜率可调整         | 1. 自适应学习负区斜率<br>2. 性能通常优于ReLU                 | 1. 增加参数和计算量<br>2. 小数据集可能过拟合            | 大型卷积网络(如ResNet)    |
| **ELU**        | $f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}$ | 负区平滑饱和           | 1. 输出均值接近0<br>2. 负区有梯度<br>3. 对噪声鲁棒           | 1. 指数计算复杂<br>2. α需要选择                         | 需要更高鲁棒性的深度网络  |
| **Softmax**    | $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$                    | 多输出归一化为概率分布 | 1. 输出和为1，适合概率分布<br>2. 可导                        | 1. 数值不稳定(需log技巧)<br>2. 类别间竞争               | 多分类输出层              |
| **Swish**      | $f(x) = x \cdot \sigma(\beta x)$                             | 类似ReLU但更平滑       | 1. 性能常优于ReLU<br>2. 平滑无突变点<br>3. 自门控特性        | 1. 计算成本较高<br>2. 实践效果依赖β                     | Transformer/大规模模型    |
| **GELU**       | $f(x) = x \cdot \Phi(x)$                                     | 类似ReLU但更平滑       | 1. 考虑输入分布<br>2. 在NLP任务中表现优异<br>3. 概率解释性强 | 1. 计算复杂(需近似)<br>2. 实现成本高                    | BERT/GPT等Transformer模型 |





```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入数据集
dataset = torchvision.datasets.CIFAR10(root="dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# 设置input
input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


# 非线性激活网络
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


chen = Chen()

writer = SummaryWriter("sigmod_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output_sigmod = chen(imgs)
    writer.add_images("output", output_sigmod, global_step=step)
    step += 1
writer.close()

output = chen(input)
print(output)

```



### 四种卷积神经网络核心特点总结

## 架构对比总览

| 特性         | ResNet                       | GoogLeNet                  | MobileNet                  | MogaNet                           |
| ------------ | ---------------------------- | -------------------------- | -------------------------- | --------------------------------- |
| **核心创新** | 残差连接                     | Inception模块              | 深度可分离卷积             | 多阶门控聚合                      |
| **主要优势** | 训练超深网络<br>解决梯度消失 | 多尺度特征融合<br>计算高效 | 超低计算成本<br>移动端优化 | 高精度低显存<br>全局特征建模      |
| **适用场景** | 通用视觉任务<br>高精度需求   | 中等资源服务器<br>实时推理 | 移动/嵌入式设备<br>IoT应用 | 资源受限的高精度需求<br>边缘计算  |
| **计算效率** | ★★★☆☆                        | ★★★★☆                      | ★★★★★                      | ★★★★☆                             |
| **参数量级** | 11M-25M+                     | 5M-7M                      | 0.5M-5M                    | 5M-90M                            |
| **典型应用** | 图像分类<br>目标检测         | 实时视频分析<br>特征提取   | 手机摄影<br>AR应用         | 边缘AI设备<br>高效Transformer替代 |

