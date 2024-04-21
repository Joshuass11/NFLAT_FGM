
import sys
import torch
import torchvision
print("torch version:",torch.__version__)
print("torchvision version:",torchvision. __version__)
print("python版本:",sys.version)
x = torch.rand(3,3)
print(x)
print("gpu:",torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)
print("环境里面包括pytorch，torchvision， cuda都是对应好匹配的版本")
print("禁止进行修改或者删除操作，如果更换或者删除不属于售后范围")

