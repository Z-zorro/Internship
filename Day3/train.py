import time
import os
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import multiprocessing

# 导入模型定义
from resnet import ResNet
from googlenet import GoogLeNet
from mobilenet import MobileNetV2
from moganet import MoGANet

def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("此脚本需要GPU支持，但未检测到可用的CUDA设备。")

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (GPU Only)')
    parser.add_argument('--model', type=str, default='resnet', 
                        choices=['resnet', 'googlenet', 'mobilenet', 'moganet'],
                        help='选择要训练的模型 (默认: resnet)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--data_dir', type=str, default='../dataset_chen',
                        help='数据集目录 (默认: ../dataset_chen)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录 (默认: ./checkpoints)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU ID (默认: 0)')
    args = parser.parse_args()

    # 设置GPU设备
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"使用GPU: {torch.cuda.get_device_name(args.gpu_id)}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 准备数据集
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载数据集
    train_data = torchvision.datasets.CIFAR10(root=args.data_dir,
                                            train=True,
                                            transform=transform_train,
                                            download=True)

    test_data = torchvision.datasets.CIFAR10(root=args.data_dir,
                                            train=False,
                                            transform=transform_test,
                                            download=True)

    train_loader = DataLoader(train_data, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=2,
                            pin_memory=True)
    
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=2,
                            pin_memory=True)

    # 创建网络模型
    if args.model == 'resnet':
        model = ResNet(num_classes=10)
    elif args.model == 'googlenet':
        model = GoogLeNet(num_classes=10)
    elif args.model == 'mobilenet':
        model = MobileNetV2(num_classes=10)
    elif args.model == 'moganet':
        model = MoGANet(num_classes=10)
    else:
        raise ValueError(f"未知模型: {args.model}")

    # 将模型移动到GPU
    model = model.to(device)
    print(f"使用模型: {model.__class__.__name__}")

    # 创建损失函数和优化器
    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)

    # 创建TensorBoard记录器
    log_dir = os.path.join("logs_train", f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    # 训练参数
    epochs = 10
    best_accuracy = 0.0

    print(f"-----开始训练{args.model}模型，共{epochs}轮-----")
    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # 统计信息
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.4f}')
        
        # 计算训练准确率
        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1} 训练结果: 损失={avg_train_loss:.4f}, 准确率={train_accuracy:.2f}%')
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 计算测试准确率
        test_accuracy = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        print(f'Epoch {epoch+1} 测试结果: 损失={avg_test_loss:.4f}, 准确率={test_accuracy:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar("train_loss", avg_train_loss, epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)
        writer.add_scalar("test_loss", avg_test_loss, epoch)
        writer.add_scalar("test_accuracy", test_accuracy, epoch)
        writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_path = os.path.join(args.save_dir, f"best_{args.model}_cifar10.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'accuracy': best_accuracy,
            }, save_path)
            print(f"保存最佳模型，准确率={best_accuracy:.2f}%")

    end_time = time.time()
    print(f"训练完成! 总耗时: {(end_time - start_time)/60:.2f}分钟")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    writer.close()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()
    main()


# ResNet
# 保存最佳模型，准确率=89.32%
# 训练完成! 总耗时: 10.25分钟
# 最佳测试准确率: 89.32%

# GoogLeNet
# 保存最佳模型，准确率=73.00%
# 训练完成! 总耗时: 8.64分钟
# 最佳测试准确率: 73.00%


# MobileNetV2
# 保存最佳模型，准确率=66.36%
# 训练完成! 总耗时: 9.96分钟
# 最佳测试准确率: 66.36%