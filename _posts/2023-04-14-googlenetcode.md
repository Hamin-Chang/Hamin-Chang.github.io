---
title : '[IC/Pytorch] 파이토치로 GoogLeNet 구현하기 ➰' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 GoogLeNet 구현하기

이번 글에서는 GoogLeNet 모델을 실제 pytorch 코드로 구현해본다. 코드는 [<U>aladdinpersson의 repository</U>](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_inceptionet.py)에 있는 코드를 사용한다.

### 1. Inception module

GoogLeNet의 핵심은 Inception module을 구현하는 것이다. Inception module 구현에 앞서 conv_block이란 클래스로 Inception module에서 반복적으로 사용될 conv layer를 구현한다. conv_block은 conv layer 뒤에 batch normalization과 ReLU가 뒤따르는 구조이다.


```python
import torch
import torch.nn as nn
from torch import Tensor

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
```

이제 convblock 클래스를 활용해서 Inception_block 클래스를 구현한다. Inception_block 클래스에서 4개의 branch를 만드는데 이는 Inception module에서 네 갈래로 갈라지는 것을 구현한 것이다.

![1](https://user-images.githubusercontent.com/77332628/231975347-6a8a6e27-39d6-4c13-95e1-4392aaefdb90.png)


```python
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
                conv_block(in_channels, red_3x3, kernel_size=1),
                conv_block(red_3x3, out_3x3t, kernel_size=(3,3),padding=1))
        
        self.branch3 = nn.Sequential(
                conv_block(in_channels, red_5x5, kernel_size=1),
                conv_block(red_5x5, out_5x5, kernel_size=5, padding=2))
        
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, out_1x1pool, kernel_size=1))
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],1)
```

### 2. Auxiliary classifier

![2](https://user-images.githubusercontent.com/77332628/231975353-fb3775bd-ef76-4751-8fb6-3b8cd28f1d71.png)

위 이미지의 auxiliary classifier를 구현한다. 1x1 conv의 output channel의 개수와 dropout rate는 논문과 같이 각각 128개와 0.7로 설정했다.


```python
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3. GoogLeNet

이제 위에서 구현한 클래스들을 사용해서 전체적인 GoogLeNet 코드를 구현해보자. 

다음 표의 아키텍처를 따라서 코드를 구현했다고 한다.

![3](https://user-images.githubusercontent.com/77332628/231975357-043655a5-9441-4f3d-a112-ba203a024349.png)



```python
class GoogLeNet(nn.Module):
    def __init__(self, aux_logits = True, num_classes=1000):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        
        self.conv1 = conv_block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception_block의 인수 순서 : in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        
        # auxiliary classifier 사용
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        
        # Auxiliary classifier는 train 할때만 사용
        # Auxiliary classifier의 output은 따로 추출
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
         x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        # Auxiliary classifier는 train 할때만 사용
        # Auxiliary classifier의 output은 따로 추출
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x
```

출처 및 참고문헌:

1. https://arxiv.org/pdf/1409.4842.pdf
2. https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_inceptionet.py
3. https://roytravel.tistory.com/338
