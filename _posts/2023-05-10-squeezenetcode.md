---
title : '[IC/Pytorch] 파이토치로 SqueezeNet 구현하기 🗜️' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 SqueezeNet 구현하기


이번 글에서는 실제 파이토치 코드로 SqeezeNet 모델을 구현해본다. SqeezeNet 모델에 대한 설명은 [**<U>SqeezeNet 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/squeezenet/)를 참고하길 바란다. 이번 글에서 사용하는 코드는 [**<U>weiaicunzai의 repository</U>**](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/squeezenet.py)의 코드를 사용했다. 

### 1. **Fire Module**

SqueezeNet을 구성하는 module로, 다음과 같이 구성되어 있다.

![1](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/1c6ae6b4-5b0c-49c8-b7e9-c3d7d372ffd4)


```python
import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_channel, out_channel, squeeze_channel):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channel, squeeze_channel,1),
                                    nn.BatchNorm2d(squeeze_channel),
                                    nn.ReLU(inplace=True))
        
        self.expand1x1 = nn.Sequential(nn.Conv2d(squeeze_channel, int(out_channel/2),1),
                                      nn.BatchNorm2d(int(out_channel/2)),
                                      nn.ReLU(inplace=True))
        
        self.expand3x3 = nn.Sequential(nn.Conv2d(squeeze_channel, int(out_channel/2),3, padding=1),
                                      nn.BatchNorm2d(int(out_channel/2)),
                                      nn.ReLU(inplace=True))
        
    def forward(self, x):
        
        x = self.squeeze(x)
        x = torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ],1)
        
        return x
```

### 2. SqueezeNet

이제 Fire module을 이용해서 전체 SqueezeNet을 구축한다. 다음 이미지의 가운데 모델인 simple bypass를 추가한 SqueezeNet을 구현한다.

![2](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/9f4c2ed4-d800-4659-aa97-a683e7f556d7)



```python
class SqueezeNet(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(3,96,3,padding=1),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2,2))
        
        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2 # bypass (skip connection)
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)
        
        f5 = self.fire5(f4) + f4 # bypass (skip connection)
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6 # bypass (skip connection)
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)
        
        f9 = self.fire9(f8)
        c10 = self.conv10(f9)
        
        x = self.avg(c10)
        x = x.view(x.size(0), -1)
        
        return x
    
def squeezenet(class_num=100):
    return SqueezeNet(class_num=class_num)
```

마지막으로 모델이 잘 구축되었는지 랜덤한 값을 입력데이터로 넣어보자.


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1,3,224,224).to(device)
model = squeezenet().to(device)
output = model(x)
print(output.size())
```

    torch.Size([1, 100])


참고자료:

1.https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/squeezenet.py
