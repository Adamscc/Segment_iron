from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from time import time
# from model.Unet3 import UNet3plus
# from model.UNet3Plus import UNet3Plus
# from CGNet.model import CGNet
from  model.model import Unet
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        # 使用Albumentations定义变换
        self.transforms = A.Compose([
            A.Resize(224, 224),  # 调整图像大小
            A.HorizontalFlip(),  # 随机水平翻转
            A.VerticalFlip(),  # 随机垂直翻转
            A.Normalize(),  # 归一化
            ToTensorV2()  # 转换为张量
        ])
        self.ids = os.listdir(images_dir)
        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # self.images_fps = [os.path.normpath(os.path.join(images_dir, image_id)) for image_id in self.ids]
        # self.masks_fps = [os.path.normpath(os.path.join(masks_dir, image_id)) for image_id in self.ids]
        self.images_fps = [images_dir + "/" + image_id for image_id in self.ids]
        self.masks_fps = [masks_dir + "/" + image_id for image_id in self.ids]

    def __getitem__(self, i):
        # 使用cv2读取图像
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i].replace(".jpg", ".png"))
        # print("mask",mask)
        # print("image",image)
        # print(self.images_fps[i])
        # print(self.masks_fps[i])

        # 将图像和掩码从BGR转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = np.array(image)
        mask = np.array(mask)

        # 应用预处理变换
        transformed = self.transforms(image=image, mask=mask)

        # 返回变换后的图像和掩码（假设掩码是单通道的，选择第一个通道）
        return transformed['image'], transformed['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Evaluate accuracy of a model on the given data set using a GPU."""
    if device is None and isinstance(net, nn.Module):
        device = next(iter(net.parameters())).device
    net.eval()  # Set the model to evaluation mode
    metric = [0.0, 0.0]  # Sum of correct predictions, number of predictions
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric[0] += (net(X).argmax(dim=1) == y).float().sum().item()
            metric[1] += y.numel()
    return metric[0] / metric[1]


def train_batch(net, features, labels, loss, trainer, device):
    """Train a batch of data."""
    if isinstance(features, list):
        features = [x.to(device) for x in features]
    else:
        features = features.to(device)
    labels = labels.to(device)
    net.train()
    trainer.zero_grad()
    predictions = net(features)
    l = loss(predictions, labels)
    l.backward()
    trainer.step()
    with torch.no_grad():
        acc = (predictions.argmax(dim=1) == labels).float().mean().item()
    return l.item(), acc


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler, devices=None):
    if devices is None:
        devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []

    for epoch in range(num_epochs):
        metric = [0.0, 0.0, 0.0,
                  0.0]  # Sum of training loss, sum of training accuracy, no. of examples, no. of predictions
        start_time = time()

        # for i, (features, labels) in enumerate(train_iter):
        for i, (features, labels) in tqdm(enumerate(train_iter), total=len(train_iter),
                                          desc=f"训练轮数 [{epoch + 1}/{num_epochs}]"):
            l, acc = train_batch(net, features, labels.long(), loss, trainer,
                                 devices[0])  ################################
            metric[0] += l * labels.shape[0]
            metric[1] += acc * labels.numel()
            metric[2] += labels.shape[0]
            metric[3] += labels.numel()

        test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
        scheduler.step()

        epoch_time = time() - start_time
        print(
            f"Epoch {epoch} \n--- Loss {metric[0] / metric[2]:.3f} \n---  Train Acc {metric[1] / metric[3]:.3f} \n--- Test Acc {test_acc:.3f} \n--- Time {epoch_time:.1f}s")

        # Save training data
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch)
        time_list.append(epoch_time)

        df = pd.DataFrame({
            'epoch': epochs_list,
            'loss': loss_list,
            'train_acc': train_acc_list,
            'test_acc': test_acc_list,
            'time': time_list
        })
        # 指定文件路径
        file_path = r"savefile/CGNet_train1.xlsx"

        # 获取目录路径
        dir_name = os.path.dirname(file_path)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # 将 DataFrame 保存为 Excel 文件
        df.to_excel(file_path, index=False)

        # Save model checkpoints
        if (epoch + 1) % 5 == 0:
            # torch.save(net.state_dict(), f'checkpoints/Unet_{epoch + 1}.pth')
            torch.save(net, f'checkpoints/UNet_{epoch + 1}_train1.pt')



x_train_dir = "../NEU_Seg-main/NEU_Seg-main/images/training"
y_train_dir = "../NEU_Seg-main/NEU_Seg-main/annotations/training"
x_test_dir = "../NEU_Seg-main/NEU_Seg-main/images/test"
y_test_dir = "../NEU_Seg-main/NEU_Seg-main/annotations/test"

train_dataset = SegDataset(
    x_train_dir,
    y_train_dir,
)
test_dataset = SegDataset(
    x_test_dir,
    y_test_dir,
)

batch_size = 16
num_epochs = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Unet(num_classes=4).to(device)
# model = UnetPlusPlus(num_classes=4).to(device)
# model = UNet3plus(n_classes=4).to(device)
# model = UNet3Plus(n_classes=4).to(device)
# model = CGNet.Context_Guided_Network(4, M=3, N=21)

lossf = nn.CrossEntropyLoss(ignore_index=255)
# 选用adam优化器来训练
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
train_ch13(model, train_loader, test_loader, lossf, optimizer, num_epochs, scheduler)
