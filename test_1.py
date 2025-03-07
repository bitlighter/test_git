# import os
# import glob
# import cv2
# import numpy as np


# path = "/home/wangf/iddx_img/test/2119/2119_1358.jpg"

# image = cv2.imread(path)

# x1, y1, x2, y2 = [
#                     580,
#                     708,
#                     1202,
#                     750
#                 ]
# cv2.rectangle(image,(int(x1), int(y1)),(int(x2), int(y2)),(0, 255, 0),2)  # 线条宽度

# cv2.imwrite("/home/wangf/demo1/test2119_1358.jpg", image)


# data = np.load("/home/wangf/risky_object/ROL/val/17_M.npz", allow_pickle=True)

# print(data.files)

# print(data['vid_id'])

# from torch.utils.tensorboard import SummaryWriter

# # 创建 SummaryWriter 对象
# writer = SummaryWriter('runs/experiment_1')  # 日志会保存到 'runs/experiment_1' 目录

# # 记录数据
# for n_iter in range(100):
#     writer.add_scalar('Loss/train', n_iter * 0.1, n_iter)
#     writer.add_scalar('Accuracy/train', 1 - n_iter * 0.01, n_iter)

# # 关闭 writer
# writer.close()

import torch

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 PyTorch 是否支持 CUDA
if torch.cuda.is_available():
    print("PyTorch 支持 CUDA。")
    # 检查 PyTorch 使用的 CUDA 版本
    print(f"PyTorch 使用的 CUDA 版本: {torch.version.cuda}")
    # 检查 GPU 设备信息
    print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("PyTorch 不支持 CUDA。")