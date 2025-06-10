# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet  # 导入AlexNet模型

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# 创建网络模型
model = AlexNet(num_classes=10)  # 创建AlexNet模型
if torch.cuda.is_available():
    model = model.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.001
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 50

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    model.train()
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试步骤
    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(model, f"Day2\model_save\\alexnet_{i}.pth")  # 修改保存的文件名
    print("模型已保存")

writer.close()

# -----第50轮训练开始-----
# 第19200的训练的loss:0.16690653562545776
# 第19300的训练的loss:0.08273670077323914
# 第19400的训练的loss:0.07210035622119904
# 第19500的训练的loss:0.10545636713504791
# 训练时间999.4201791286469
# 整体测试集上的loss:90.28461158275604
# 整体测试集上的正确率：0.7735999822616577
# 模型已保存