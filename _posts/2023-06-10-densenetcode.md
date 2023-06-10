---
title : '[IC/Pytorch] 파이토치로 DenseNet 구현하기 🕸️' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---


## Pytorch로 DenseNet 구현하기

이번 글에서는 실제 pytorch 코드로 DenseNet을 구현해본다. DenseNet에 대한 개념은 [**<U>DenseNet 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/DenseNet/)를 참고하길 바란다. pytorch DenseNet 코드는 [**<U>weiaicunzai's repository</U>**](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py)를 참고했다.

실제 데이터를 이용해서 모델을 학습하는 과정은 생략하고 pytorch 코드로 구현하는 것으로 이 글을 마치겠다.


### 1. Bottleneck block

가장 먼저 DenseNet에 사용되는 Bottleneck block을 정의한다.

이미지1

DenseNet에 사용되는 Bottleneck 구조는 특이하게 1x1 conv 연산을 통해서 4 x growth rate개의 feature map을 만들고 그 뒤에 3x3 conv를 통해서 growth rate개의 feature map으로 줄여준다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        
        self.bottle_neck = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(inner_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False))
    
    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)
```

### 2. Transition Layer

pooling 역할을 하는 Transition layer를 정의한다.


```python
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                        nn.AvgPool2d(2, stride=2))
    
    def forward(self, x):
        return self.down_sample(x)
```

### 3. DenseNet

이제 Bottleneck block과 Transition layer를 이용해서 DenseNet 모델을 구축하는 클래스를 정의한다.

논문에서와 마찬가지로 growth rate $k=12$와 compression factor $θ=0.5$로 설정한다.


```python
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100): # cifar-100 데이터로 학습 가정
        super().__init__()
        self.growth_rate = growth_rate
        
        inner_channels = 2*growth_rate
        
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential()
        
        
        
        for index in range(len(nblocks)-1):
            self.features.add_module('dense_block_layer_{}'.format(index), self._make_dense_layers(block,inner_channels,nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        
        self.features.add_module('dense_block{}'.format(len(nblocks)-1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks)-1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(inner_channels, num_class)
        
    def forward(self,x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output
    
    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        
        return dense_block
```

이제 DenseNet 클래스의 인수 중 block수와 growth_rate를 바꿔가며 모델을 생성하면 된다. 우리는 densenet121을 생성한 후 잘 작동하는지 임의의 input을 넣어보고 모델을 돌려보고 글을 마무리한다.


```python
def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

x = torch.randn(3, 3, 64, 64)
model = densenet121()
output = model(x)
print(output.size())
```

    torch.Size([3, 100])


출처 및 참고문헌

1. https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py
2. https://velog.io/@lighthouse97/DenseNet%EC%9D%98-%EC%9D%B4%ED%95%B4
