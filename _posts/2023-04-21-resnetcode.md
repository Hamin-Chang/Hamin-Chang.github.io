---
title : '[IC/Pytorch] 파이토치로 ResNet 구현하기 ⏭️' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 ResNet 구현하기

이번 글에서는 ResNet을 실제 pytorch 코드로 구현해본다. ResNet의 개념은 [**<U>ResNet 논문리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/resnet/)를 참고하길 바란다. ResNet 구현 코드는 [**<U>roytravel의 repository</U>**](https://github.com/roytravel/paper-implementation/blob/master/resnet/resnet.py)의 코드를 사용했다.

### 1. Basic building block (Identity shortcut)

먼저 ResNet의 가장 핵심인 Identity shortcut을 사용한 Basic building block을 구현한다.

(다음 이미지의 왼쪽)

![1](https://user-images.githubusercontent.com/77332628/233517177-c0e91c1e-504f-4abe-a5a4-800a9fa7c64e.png)

```python
import torch
from torch import Tensor
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion_factor = 1  # identity shortcut이기 떄문에 1로 설정
    def __init__(self, in_channels : int, out_channels: int, stride: int =1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu2 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                                         nn.BatchNorm2d(out_channels*self.expansion_factor))
    
    def forward(self, x:Tensor) -> Tensor:
        out = x 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.residual(out) # identitiy shortcut
        x = self.relu2(x)
        return x
```
### 2. Bottleneck building block (Projection shortcut)

이제 또 다른 구조인 Projection shortcut 방식의 Bottleneck building block을 구현한다.

(다음 이미지의 오른쪽)

![1](https://user-images.githubusercontent.com/77332628/233517177-c0e91c1e-504f-4abe-a5a4-800a9fa7c64e.png)


```python
class BottleNeck(nn.Module):
    expansion_factor = 4
    def __init__(self, in_channels:int, out_channels:int, stride:int =1):
        super(BottleNeck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion_factor, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion_factor)
        self.relu3 = nn.ReLU()
        
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                                         nn.BatchNorm2d(out_channels*self.expansion_factor))
    
    def forward(self, x:Tensor) -> Tensor:
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x += self.residual(out) # projection shortcut
        x = self.relu3(x)
        return x
```

위 두 구조를 이용해서 ResNet 클래스를 구성한다. 이때 다음 표처럼 depth에 따라 layer를 다르게 구성할 수 있도록 ResNet 클래스를 구성한다.

### 3. depth에 따른 ResNet 

![2](https://user-images.githubusercontent.com/77332628/233517181-caf89080-254f-4f02-a10d-31d9e0196dfc.png)


```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(num_features=64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512 * block.expansion_factor, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers)
    
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x:Tensor) -> Tensor :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
```

이제 ResNet의 depth에 따라서 layer의 개수를 맞춰서 모델을 반환하는 클래스를 정의한다.



```python
class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2,2,2,2])
    
    def resnet34(self):
        return ResNet(BasicBlock, [3,4,6,3])
    
    def resnet50(self):
        return ResNet(BottleNeck, [3,4,6,3])
    
    def resnet101(self):
        return ResNet(BottleNeck, [3,4,23,3])
    
    def resnet152(self):
        return ResNet(BottleNeck, [3,8,36,3])
```

마지막으로 간단한 테스트를 위해 모델을 불러오고 랜덤하게 생성한 데이터를 넣어주고 결과를 확인해보고 마무리한다.


```python
model = Model().resnet152()
y= model(torch.randn(1,3,224,224))
print(y.size())
```

    torch.Size([1, 10])

출처 및 참고문헌 :

1. https://github.com/roytravel/paper-implementation/blob/master/resnet/train.py
2. https://roytravel.tistory.com/339
