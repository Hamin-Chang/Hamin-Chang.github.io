---
title : '[IC/Pytorch] 파이토치로 ResNeXt 구현하기 ⏭️' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 ResNeXt 구현하기

이번 글에서는 ResNeXt를 실제 pytorch 코드로 구현해본다. ResNeXt에 대한 설명은 [**<U>ResNeXt 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/ResNeXt/)를 참고하길 바란다. pytorch 코드는 [**<U>weiaicunzai's respository</U>**](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnext.py)의 코드를 사용했다.

실제 데이터를 이용해서 모델을 학습하는 과정은 생략하고 pytorch 코드로 구현하는 것으로 이 글을 마치겠다.

### 1. ResNeXt Bottleneck block

ResNeXt block으로는 논문의 Fig 3의 (c)만 사용한다.

이미지10


```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

class ResNextBottleneckC(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        C = CARDINALITY
        D = int(DEPTH * out_channels / BASEWIDTH) # 각 group당 channel 수
        
        self.split_transforms = nn.Sequential(nn.Conv2d(in_channels, C*D, kernel_size=1, groups=C, bias=False),
                                             nn.BatchNorm2d(C*D),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(C*D, C*D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
                                             nn.BatchNorm2d(C*D),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(C*D,out_channels*4, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(out_channels*4))
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels*4, stride=stride, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(out_channels*4))
        
    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))
    
        
```

### 2. ResNeXt 구축


```python
class ResNext(nn.Module):
    def __init__(self, block, num_blocks, class_names=100): # cifar100으로 train하는 코드이기 때문에 class 개수 100
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        
        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, 100)
    
    # Building ResNext block
    def _make_layer(self, block, num_block, out_channels, stride):
        '''block : block type (기본값 resnext bottleneck c)
        num_block : layer 당 block 개수
        out_channels : block의 output channel
        stride : block stride'''
        
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def resnext50():
    # resnext50(c32x4d) network
    return ResNext(ResNextBottleneckC, [3,4,6,3])

def resnext101():
    # resnext101(c32x4d) network
    return ResNext(ResNextBottleneckC, [3,4,23,3])

def resnext152():
    # resnext152(c32x4d) network
    return ResNext(ResNextBottleneckC, [3,4,36,3])
```

모델이 잘 구축되었는지 랜덤한 입력 데이터를 모델에 넣어보고 출력하고 글을 마무리한다.


```python
# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn((3, 3, 224, 224)).to(device)
model = resnext50().to(device)
output = model(x)
print('output size: ', output.size())
```

    output size:  torch.Size([3, 100])


출처 및 참고문헌 :

1. https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnext.py
2. https://arxiv.org/pdf/1611.05431.pdf
3. https://deep-learning-study.tistory.com/558
