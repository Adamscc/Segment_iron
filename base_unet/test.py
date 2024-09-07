import numpy as np
import torch

# da = np.load(r"D:\AILAB\校园算法\挑战赛\赛题2-钢材表面缺陷检测与分割\Seg（分割赛题记分）\test_ground_truths\000001.npy")
# print(da.shape)

model = torch.load(r"D:\AILAB\校园算法\挑战赛\base_unet\checkpoints\Unet_5.pth")