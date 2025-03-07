import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

model=torch.load("moder_for_cifar10.pth")

test_data=torchvision.datasets.CIFAR10(root="./datasetCIFAR10",train=False,download=True,
                                      transform=torchvision.transforms.ToTensor())
test_data_length=len(test_data)
test_loader = DataLoader(test_data,batch_size=128)

if torch.cuda.is_available():
    model=model.cuda()

loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

model.eval()
total_test_loss=0
total_ac=0
with torch.no_grad():
    for data in test_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = model(imgs)
        loss = loss_fn(output, targets)
        total_test_loss=total_test_loss+loss
        ac=(output.argmax(1)==targets).sum()
        total_ac=total_ac+ac

print("整体测试集的误差：{}".format(total_test_loss))
print("整体测试集的正确率：{}".format(total_ac/test_data_length))