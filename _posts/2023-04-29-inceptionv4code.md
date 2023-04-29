---
title : '[IC/Pytorch] 파이토치로 Inception-ResNet v2 구현하기 ➰' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 Iception-ResNet-v2 구현하기

이번 글에서는 Inception-v4 모델에 residual block을 결합한 Inception-ResNet-v2 모델을 구현해본다. Inception-v4와 Inception-ResNet에 대한 설명은 [**<U>Inception-v4 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/inceptionv4/)을 참고하길 바란다.

pytorch 코드는 [**<U>Seonghoon-Yu의 repository</U>**](https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/Classification/Inceptionv4(2016).ipynb)의 코드를 사용했다.



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
```

### 1. Stem

먼저 conv layer 클래스를 정의한다.


```python
class BasicConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super().__init__()

    self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, bias=False,**kwargs),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU())
    
  def forward(self,x):
    x = self.conv(x)
    return x
```

stem 클래스를 정의한다.

이미지1


```python
class Stem(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(BasicConv2d(3,32,3,stride=2, padding=0), # 149x149x32
                               BasicConv2d(32,32,3,stride=1, padding=0), # 147x147x32
                               BasicConv2d(32,64,3,stride=1, padding=1) # 147x147x64
                               )
    self.branch3x3_conv = BasicConv2d(64,96,3,stride=2, padding=0) # 73x73x96

    # MaxPool을 3x3로 적용하면 feature map size가 맞지 않아서 4x4로 적용
    self.branch3x3_pool = nn.MaxPool2d(4, stride=2, padding=1) # 73x73x64

    self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, 1, stride=1, padding=0),
                                    BasicConv2d(64, 96, 3, stride=1, padding=0)) # 71x71x96

    self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, 1, stride=1, padding=0),
                                    BasicConv2d(64, 64, (7,1), stride=1, padding=(3,0)),
                                    BasicConv2d(64, 64, (1,7), stride=1, padding=(0,3)),
                                    BasicConv2d(64, 96, 3, stride=1, padding=0)) # 71x71x96

    self.branchpoola = BasicConv2d(192, 192, 3, stride=2, padding=0) # 35x35x192

    # MaxPool을 3x3로 적용하면 feature map size가 맞지 않아서 4x4로 적용
    self.branchpoolb = nn.MaxPool2d(4, 2, 1) # 35x35x192

  def forward(self, x):
    x = self.conv1(x)
    x = torch.cat((self.branch3x3_conv(x), self.branch3x3_pool(x)), dim=1)
    x = torch.cat((self.branch7x7a(x), self.branch7x7b(x)), dim=1)
    x = torch.cat((self.branchpoola(x), self.branchpoolb(x)), dim=1)
    return x

```

Stem module이 잘 구축됐는지 확인해보자. 중간중간 채널 수와 feautre map 크기가 잘 맞는지 확인해야한다.


```python
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn((3,3,299,299)).to(device)
model = Stem().to(device)
output_Stem = model(x)
print('Input size:',x.size())
print('Stem output size:',output_Stem.size())
```

    Input size: torch.Size([3, 3, 299, 299])
    Stem output size: torch.Size([3, 384, 35, 35])


### 2. Inception-ResNetA & ReductionA module

다음으로 Inception-ResNetA module 클래스를 정의한다.

이미지2


```python
class Inception_ResNet_A(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    self.branch1x1 = BasicConv2d(in_channels, 32, 1, stride=1, padding=0)

    self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 32, 1, stride=1, padding=0),
                                   BasicConv2d(32, 32, 3, stride=1, padding=1))
    
    self.branch3x3stack = nn.Sequential(BasicConv2d(in_channels, 32, 1, stride=1, padding=0),
                                        BasicConv2d(32, 48, 3, stride=1, padding=1),
                                        BasicConv2d(48, 64, 3, stride=1, padding=1))
    
    self.reduction1x1 = nn.Conv2d(128, 384, 1, stride=1, padding=0)
    self.shortcut = nn.Conv2d(in_channels, 384, 1, stride=1, padding=0)
    self.bn = nn.BatchNorm2d(384)
    self.relu = nn.ReLU()

  def forward(self, x):
    x_shortcut = self.shortcut(x)
    x = torch.cat((self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
    x = self.reduction1x1(x)
    x = self.bn(x_shortcut + x)
    x = self.relu(x)
    return x
```

Inception-ResNetA module이 잘 구축됐는지 확인해보자.



```python
model = Inception_ResNet_A(output_Stem.size()[1]).to(device)
output_resA = model(output_Stem)
print('Input size:',output_Stem.size())
print('Stem output size:',output_resA.size())
```

    Input size: torch.Size([3, 384, 35, 35])
    Stem output size: torch.Size([3, 384, 35, 35])


다음으로 ReductionA 클래스를 정의한다.

이미지3

위 이미지에서 ReductionA에서는 k=256, l=256, m=384, n=384를 사용한다.


```python
class ReductionA(nn.Module):
  def __init__(self, in_channels, k, l, m, n):
    super().__init__()

    self.branchpool = nn.MaxPool2d(3,2)
    self.branch3x3 = BasicConv2d(in_channels, n, 3, stride=2, padding=0)
    self.branch3x3stack = nn.Sequential(BasicConv2d(in_channels, k, 1, stride=1, padding=0),
                                         BasicConv2d(k, l, 3, stride=1, padding=1),
                                         BasicConv2d(l, m , 2, stride=2, padding=0))
    self.output_channels = in_channels + n + m
    
  def forward(self, x):
    x = torch.cat((self.branchpool(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
    return x

# ReductionA 잘 구축되었는지 확인
print('Input size:',output_resA.size())
model = ReductionA(output_resA.size()[1], 256, 256, 384, 384).to(device)
output_rA = model(output_resA)
print('Stem output size:',output_rA.size())
```

    Input size: torch.Size([3, 384, 35, 35])
    Stem output size: torch.Size([3, 1152, 17, 17])


### 3. Inception-ResNetB & ReductionB module

Inception-ResNetB module 클래스 정의한다.

이미지4




```python
class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 128, 1, stride=1, padding=0),
            BasicConv2d(128, 160, (1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, (7,1), stride=1, padding=(3,0))
        )

        self.reduction1x1 = nn.Conv2d(384, 1152, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 1152, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1152)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat((self.branch1x1(x), self.branch7x7(x)), dim=1)
        x = self.reduction1x1(x) * 0.1
        x = self.bn(x + x_shortcut)
        x = self.relu(x)
        return x

# check Inception-ResNetB module
model = Inception_ResNet_B(output_rA.size()[1]).to(device)
output_resB = model(output_rA)
print('Input size:',output_rA.size())
print('Stem output size:',output_resB.size())
```

    Input size: torch.Size([3, 1152, 17, 17])
    Stem output size: torch.Size([3, 1152, 17, 17])


ReductionB module 클래스를 정의한다.

이미지5


```python
class ReductionB(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    
    self.branchpool = nn.MaxPool2d(3,2)
    self.branch3x3a = nn.Sequential(BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
                                    BasicConv2d(256, 384, 3, stride=2, padding=0))
    
    self.branch3x3b = nn.Sequential(BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
                                    BasicConv2d(256, 288, 3, stride=2, padding=0))
    
    self.branch3x3stack = nn.Sequential(BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
                                        BasicConv2d(256, 288, 3, stride=1, padding=1),
                                        BasicConv2d(288, 320, 3, stride=2, padding=0))
    
  def forward(self, x):
    x = torch.cat((self.branchpool(x), self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x)), dim=1)
    return x

# check ReductionB
model = ReductionB(output_resB.size()[1]).to(device)
output_rB = model(output_resB)
print('Input size:', output_resB.size())
print('output size:', output_rB.size())
```

    Input size: torch.Size([3, 1152, 17, 17])
    output size: torch.Size([3, 2144, 8, 8])


### 4. Inception-ResNetC module

마지막으로 Inception-ResNetC module 클래스를 정의한다.

이미지6


```python
class Inception_ResNet_C(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
    self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 192, 1, stride=1, padding=0),
                                   BasicConv2d(192, 224, (1,3), stride=1, padding=(0,1)),
                                   BasicConv2d(224, 256, (3,1), stride=1, padding=(1,0)))
    
    self.reduction1x1 = nn.Conv2d(448, 2144, 1, stride=1, padding=0)
    self.shortcut = nn.Conv2d(in_channels, 2144, 1, stride=1, padding=0)
    self.bn = nn.BatchNorm2d(2144)
    self.relu = nn.ReLU()

  def forward(self, x):
    x_shortcut = self.shortcut(x)
    x = torch.cat((self.branch1x1(x), self.branch3x3(x)), dim=1)
    x = self.reduction1x1(x) * 0.1
    x = self.bn(x_shortcut + x)
    x = self.relu(x)
    return x

# check Inception-ResNetC
model = Inception_ResNet_C(output_rB.size()[1]).to(device)
output_resC = model(output_rB)
print('Input size:', output_rB.size())
print('output size:', output_resC.size())

```

    Input size: torch.Size([3, 2144, 8, 8])
    output size: torch.Size([3, 2144, 8, 8])


### 5. Inception-ResNet-v2

이제 위에서 정의한 모듈들을 사용해서 전체 모델을 구축한다.

이미지7


```python
class InceptionResNetV2(nn.Module):
  def __init__(self, A, B, C, k=256, l=256, m=384, n=384, num_classes=10, init_weights=True):
    super().__init__()
    blocks = []
    blocks.append(Stem())
    for i in range(A):
      blocks.append(Inception_ResNet_A(384))
    blocks.append(ReductionA(384, k, l, m, n))
    for i in range(B):
      blocks.append(Inception_ResNet_B(1152))
    blocks.append(ReductionB(1152))
    for i in range(C):
      blocks.append(Inception_ResNet_C(2144))
    
    self.features = nn.Sequential(*blocks)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.dropout = nn.Dropout2d(0.2) # = keep 0.8
    self.linear = nn.Linear(2144, num_classes)

    # 가중치 초기화
    if init_weights:
      self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.dropout(x)
    x = self.linear(x)
    return x
  
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

```

전체 모델이 잘 구축되었는지 확인해보자. 각 Inception module은 논문의 2배로 설정해서 모델을 구축한다.


```python
model = InceptionResNetV2(10, 20, 10).to(device)
summary(model, (3,299,299), device=device.type)
```

    /usr/local/lib/python3.10/dist-packages/torch/nn/functional.py:1331: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)


    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 32, 149, 149]             864
           BatchNorm2d-2         [-1, 32, 149, 149]              64
                  ReLU-3         [-1, 32, 149, 149]               0
           BasicConv2d-4         [-1, 32, 149, 149]               0
                Conv2d-5         [-1, 32, 147, 147]           9,216
           BatchNorm2d-6         [-1, 32, 147, 147]              64
                  ReLU-7         [-1, 32, 147, 147]               0
           BasicConv2d-8         [-1, 32, 147, 147]               0
                Conv2d-9         [-1, 64, 147, 147]          18,432
          BatchNorm2d-10         [-1, 64, 147, 147]             128
                 ReLU-11         [-1, 64, 147, 147]               0
          BasicConv2d-12         [-1, 64, 147, 147]               0
               Conv2d-13           [-1, 96, 73, 73]          55,296
          BatchNorm2d-14           [-1, 96, 73, 73]             192
                 ReLU-15           [-1, 96, 73, 73]               0
          BasicConv2d-16           [-1, 96, 73, 73]               0
            MaxPool2d-17           [-1, 64, 73, 73]               0
               Conv2d-18           [-1, 64, 73, 73]          10,240
          BatchNorm2d-19           [-1, 64, 73, 73]             128
                 ReLU-20           [-1, 64, 73, 73]               0
          BasicConv2d-21           [-1, 64, 73, 73]               0
               Conv2d-22           [-1, 96, 71, 71]          55,296
          BatchNorm2d-23           [-1, 96, 71, 71]             192
                 ReLU-24           [-1, 96, 71, 71]               0
          BasicConv2d-25           [-1, 96, 71, 71]               0
               Conv2d-26           [-1, 64, 73, 73]          10,240
          BatchNorm2d-27           [-1, 64, 73, 73]             128
                 ReLU-28           [-1, 64, 73, 73]               0
          BasicConv2d-29           [-1, 64, 73, 73]               0
               Conv2d-30           [-1, 64, 73, 73]          28,672
          BatchNorm2d-31           [-1, 64, 73, 73]             128
                 ReLU-32           [-1, 64, 73, 73]               0
          BasicConv2d-33           [-1, 64, 73, 73]               0
               Conv2d-34           [-1, 64, 73, 73]          28,672
          BatchNorm2d-35           [-1, 64, 73, 73]             128
                 ReLU-36           [-1, 64, 73, 73]               0
          BasicConv2d-37           [-1, 64, 73, 73]               0
               Conv2d-38           [-1, 96, 71, 71]          55,296
          BatchNorm2d-39           [-1, 96, 71, 71]             192
                 ReLU-40           [-1, 96, 71, 71]               0
          BasicConv2d-41           [-1, 96, 71, 71]               0
               Conv2d-42          [-1, 192, 35, 35]         331,776
          BatchNorm2d-43          [-1, 192, 35, 35]             384
                 ReLU-44          [-1, 192, 35, 35]               0
          BasicConv2d-45          [-1, 192, 35, 35]               0
            MaxPool2d-46          [-1, 192, 35, 35]               0
                 Stem-47          [-1, 384, 35, 35]               0
               Conv2d-48          [-1, 384, 35, 35]         147,840
               Conv2d-49           [-1, 32, 35, 35]          12,288
          BatchNorm2d-50           [-1, 32, 35, 35]              64
                 ReLU-51           [-1, 32, 35, 35]               0
          BasicConv2d-52           [-1, 32, 35, 35]               0
               Conv2d-53           [-1, 32, 35, 35]          12,288
          BatchNorm2d-54           [-1, 32, 35, 35]              64
                 ReLU-55           [-1, 32, 35, 35]               0
          BasicConv2d-56           [-1, 32, 35, 35]               0
               Conv2d-57           [-1, 32, 35, 35]           9,216
          BatchNorm2d-58           [-1, 32, 35, 35]              64
                 ReLU-59           [-1, 32, 35, 35]               0
          BasicConv2d-60           [-1, 32, 35, 35]               0
               Conv2d-61           [-1, 32, 35, 35]          12,288
          BatchNorm2d-62           [-1, 32, 35, 35]              64
                 ReLU-63           [-1, 32, 35, 35]               0
          BasicConv2d-64           [-1, 32, 35, 35]               0
               Conv2d-65           [-1, 48, 35, 35]          13,824
          BatchNorm2d-66           [-1, 48, 35, 35]              96
                 ReLU-67           [-1, 48, 35, 35]               0
          BasicConv2d-68           [-1, 48, 35, 35]               0
               Conv2d-69           [-1, 64, 35, 35]          27,648
          BatchNorm2d-70           [-1, 64, 35, 35]             128
                 ReLU-71           [-1, 64, 35, 35]               0
          BasicConv2d-72           [-1, 64, 35, 35]               0
               Conv2d-73          [-1, 384, 35, 35]          49,536
          BatchNorm2d-74          [-1, 384, 35, 35]             768
                 ReLU-75          [-1, 384, 35, 35]               0
    Inception_ResNet_A-76          [-1, 384, 35, 35]               0
               Conv2d-77          [-1, 384, 35, 35]         147,840
               Conv2d-78           [-1, 32, 35, 35]          12,288
          BatchNorm2d-79           [-1, 32, 35, 35]              64
                 ReLU-80           [-1, 32, 35, 35]               0
          BasicConv2d-81           [-1, 32, 35, 35]               0
               Conv2d-82           [-1, 32, 35, 35]          12,288
          BatchNorm2d-83           [-1, 32, 35, 35]              64
                 ReLU-84           [-1, 32, 35, 35]               0
          BasicConv2d-85           [-1, 32, 35, 35]               0
               Conv2d-86           [-1, 32, 35, 35]           9,216
          BatchNorm2d-87           [-1, 32, 35, 35]              64
                 ReLU-88           [-1, 32, 35, 35]               0
          BasicConv2d-89           [-1, 32, 35, 35]               0
               Conv2d-90           [-1, 32, 35, 35]          12,288
          BatchNorm2d-91           [-1, 32, 35, 35]              64
                 ReLU-92           [-1, 32, 35, 35]               0
          BasicConv2d-93           [-1, 32, 35, 35]               0
               Conv2d-94           [-1, 48, 35, 35]          13,824
          BatchNorm2d-95           [-1, 48, 35, 35]              96
                 ReLU-96           [-1, 48, 35, 35]               0
          BasicConv2d-97           [-1, 48, 35, 35]               0
               Conv2d-98           [-1, 64, 35, 35]          27,648
          BatchNorm2d-99           [-1, 64, 35, 35]             128
                ReLU-100           [-1, 64, 35, 35]               0
         BasicConv2d-101           [-1, 64, 35, 35]               0
              Conv2d-102          [-1, 384, 35, 35]          49,536
         BatchNorm2d-103          [-1, 384, 35, 35]             768
                ReLU-104          [-1, 384, 35, 35]               0
    Inception_ResNet_A-105          [-1, 384, 35, 35]               0
              Conv2d-106          [-1, 384, 35, 35]         147,840
              Conv2d-107           [-1, 32, 35, 35]          12,288
         BatchNorm2d-108           [-1, 32, 35, 35]              64
                ReLU-109           [-1, 32, 35, 35]               0
         BasicConv2d-110           [-1, 32, 35, 35]               0
              Conv2d-111           [-1, 32, 35, 35]          12,288
         BatchNorm2d-112           [-1, 32, 35, 35]              64
                ReLU-113           [-1, 32, 35, 35]               0
         BasicConv2d-114           [-1, 32, 35, 35]               0
              Conv2d-115           [-1, 32, 35, 35]           9,216
         BatchNorm2d-116           [-1, 32, 35, 35]              64
                ReLU-117           [-1, 32, 35, 35]               0
         BasicConv2d-118           [-1, 32, 35, 35]               0
              Conv2d-119           [-1, 32, 35, 35]          12,288
         BatchNorm2d-120           [-1, 32, 35, 35]              64
                ReLU-121           [-1, 32, 35, 35]               0
         BasicConv2d-122           [-1, 32, 35, 35]               0
              Conv2d-123           [-1, 48, 35, 35]          13,824
         BatchNorm2d-124           [-1, 48, 35, 35]              96
                ReLU-125           [-1, 48, 35, 35]               0
         BasicConv2d-126           [-1, 48, 35, 35]               0
              Conv2d-127           [-1, 64, 35, 35]          27,648
         BatchNorm2d-128           [-1, 64, 35, 35]             128
                ReLU-129           [-1, 64, 35, 35]               0
         BasicConv2d-130           [-1, 64, 35, 35]               0
              Conv2d-131          [-1, 384, 35, 35]          49,536
         BatchNorm2d-132          [-1, 384, 35, 35]             768
                ReLU-133          [-1, 384, 35, 35]               0
    Inception_ResNet_A-134          [-1, 384, 35, 35]               0
              Conv2d-135          [-1, 384, 35, 35]         147,840
              Conv2d-136           [-1, 32, 35, 35]          12,288
         BatchNorm2d-137           [-1, 32, 35, 35]              64
                ReLU-138           [-1, 32, 35, 35]               0
         BasicConv2d-139           [-1, 32, 35, 35]               0
              Conv2d-140           [-1, 32, 35, 35]          12,288
         BatchNorm2d-141           [-1, 32, 35, 35]              64
                ReLU-142           [-1, 32, 35, 35]               0
         BasicConv2d-143           [-1, 32, 35, 35]               0
              Conv2d-144           [-1, 32, 35, 35]           9,216
         BatchNorm2d-145           [-1, 32, 35, 35]              64
                ReLU-146           [-1, 32, 35, 35]               0
         BasicConv2d-147           [-1, 32, 35, 35]               0
              Conv2d-148           [-1, 32, 35, 35]          12,288
         BatchNorm2d-149           [-1, 32, 35, 35]              64
                ReLU-150           [-1, 32, 35, 35]               0
         BasicConv2d-151           [-1, 32, 35, 35]               0
              Conv2d-152           [-1, 48, 35, 35]          13,824
         BatchNorm2d-153           [-1, 48, 35, 35]              96
                ReLU-154           [-1, 48, 35, 35]               0
         BasicConv2d-155           [-1, 48, 35, 35]               0
              Conv2d-156           [-1, 64, 35, 35]          27,648
         BatchNorm2d-157           [-1, 64, 35, 35]             128
                ReLU-158           [-1, 64, 35, 35]               0
         BasicConv2d-159           [-1, 64, 35, 35]               0
              Conv2d-160          [-1, 384, 35, 35]          49,536
         BatchNorm2d-161          [-1, 384, 35, 35]             768
                ReLU-162          [-1, 384, 35, 35]               0
    Inception_ResNet_A-163          [-1, 384, 35, 35]               0
              Conv2d-164          [-1, 384, 35, 35]         147,840
              Conv2d-165           [-1, 32, 35, 35]          12,288
         BatchNorm2d-166           [-1, 32, 35, 35]              64
                ReLU-167           [-1, 32, 35, 35]               0
         BasicConv2d-168           [-1, 32, 35, 35]               0
              Conv2d-169           [-1, 32, 35, 35]          12,288
         BatchNorm2d-170           [-1, 32, 35, 35]              64
                ReLU-171           [-1, 32, 35, 35]               0
         BasicConv2d-172           [-1, 32, 35, 35]               0
              Conv2d-173           [-1, 32, 35, 35]           9,216
         BatchNorm2d-174           [-1, 32, 35, 35]              64
                ReLU-175           [-1, 32, 35, 35]               0
         BasicConv2d-176           [-1, 32, 35, 35]               0
              Conv2d-177           [-1, 32, 35, 35]          12,288
         BatchNorm2d-178           [-1, 32, 35, 35]              64
                ReLU-179           [-1, 32, 35, 35]               0
         BasicConv2d-180           [-1, 32, 35, 35]               0
              Conv2d-181           [-1, 48, 35, 35]          13,824
         BatchNorm2d-182           [-1, 48, 35, 35]              96
                ReLU-183           [-1, 48, 35, 35]               0
         BasicConv2d-184           [-1, 48, 35, 35]               0
              Conv2d-185           [-1, 64, 35, 35]          27,648
         BatchNorm2d-186           [-1, 64, 35, 35]             128
                ReLU-187           [-1, 64, 35, 35]               0
         BasicConv2d-188           [-1, 64, 35, 35]               0
              Conv2d-189          [-1, 384, 35, 35]          49,536
         BatchNorm2d-190          [-1, 384, 35, 35]             768
                ReLU-191          [-1, 384, 35, 35]               0
    Inception_ResNet_A-192          [-1, 384, 35, 35]               0
              Conv2d-193          [-1, 384, 35, 35]         147,840
              Conv2d-194           [-1, 32, 35, 35]          12,288
         BatchNorm2d-195           [-1, 32, 35, 35]              64
                ReLU-196           [-1, 32, 35, 35]               0
         BasicConv2d-197           [-1, 32, 35, 35]               0
              Conv2d-198           [-1, 32, 35, 35]          12,288
         BatchNorm2d-199           [-1, 32, 35, 35]              64
                ReLU-200           [-1, 32, 35, 35]               0
         BasicConv2d-201           [-1, 32, 35, 35]               0
              Conv2d-202           [-1, 32, 35, 35]           9,216
         BatchNorm2d-203           [-1, 32, 35, 35]              64
                ReLU-204           [-1, 32, 35, 35]               0
         BasicConv2d-205           [-1, 32, 35, 35]               0
              Conv2d-206           [-1, 32, 35, 35]          12,288
         BatchNorm2d-207           [-1, 32, 35, 35]              64
                ReLU-208           [-1, 32, 35, 35]               0
         BasicConv2d-209           [-1, 32, 35, 35]               0
              Conv2d-210           [-1, 48, 35, 35]          13,824
         BatchNorm2d-211           [-1, 48, 35, 35]              96
                ReLU-212           [-1, 48, 35, 35]               0
         BasicConv2d-213           [-1, 48, 35, 35]               0
              Conv2d-214           [-1, 64, 35, 35]          27,648
         BatchNorm2d-215           [-1, 64, 35, 35]             128
                ReLU-216           [-1, 64, 35, 35]               0
         BasicConv2d-217           [-1, 64, 35, 35]               0
              Conv2d-218          [-1, 384, 35, 35]          49,536
         BatchNorm2d-219          [-1, 384, 35, 35]             768
                ReLU-220          [-1, 384, 35, 35]               0
    Inception_ResNet_A-221          [-1, 384, 35, 35]               0
              Conv2d-222          [-1, 384, 35, 35]         147,840
              Conv2d-223           [-1, 32, 35, 35]          12,288
         BatchNorm2d-224           [-1, 32, 35, 35]              64
                ReLU-225           [-1, 32, 35, 35]               0
         BasicConv2d-226           [-1, 32, 35, 35]               0
              Conv2d-227           [-1, 32, 35, 35]          12,288
         BatchNorm2d-228           [-1, 32, 35, 35]              64
                ReLU-229           [-1, 32, 35, 35]               0
         BasicConv2d-230           [-1, 32, 35, 35]               0
              Conv2d-231           [-1, 32, 35, 35]           9,216
         BatchNorm2d-232           [-1, 32, 35, 35]              64
                ReLU-233           [-1, 32, 35, 35]               0
         BasicConv2d-234           [-1, 32, 35, 35]               0
              Conv2d-235           [-1, 32, 35, 35]          12,288
         BatchNorm2d-236           [-1, 32, 35, 35]              64
                ReLU-237           [-1, 32, 35, 35]               0
         BasicConv2d-238           [-1, 32, 35, 35]               0
              Conv2d-239           [-1, 48, 35, 35]          13,824
         BatchNorm2d-240           [-1, 48, 35, 35]              96
                ReLU-241           [-1, 48, 35, 35]               0
         BasicConv2d-242           [-1, 48, 35, 35]               0
              Conv2d-243           [-1, 64, 35, 35]          27,648
         BatchNorm2d-244           [-1, 64, 35, 35]             128
                ReLU-245           [-1, 64, 35, 35]               0
         BasicConv2d-246           [-1, 64, 35, 35]               0
              Conv2d-247          [-1, 384, 35, 35]          49,536
         BatchNorm2d-248          [-1, 384, 35, 35]             768
                ReLU-249          [-1, 384, 35, 35]               0
    Inception_ResNet_A-250          [-1, 384, 35, 35]               0
              Conv2d-251          [-1, 384, 35, 35]         147,840
              Conv2d-252           [-1, 32, 35, 35]          12,288
         BatchNorm2d-253           [-1, 32, 35, 35]              64
                ReLU-254           [-1, 32, 35, 35]               0
         BasicConv2d-255           [-1, 32, 35, 35]               0
              Conv2d-256           [-1, 32, 35, 35]          12,288
         BatchNorm2d-257           [-1, 32, 35, 35]              64
                ReLU-258           [-1, 32, 35, 35]               0
         BasicConv2d-259           [-1, 32, 35, 35]               0
              Conv2d-260           [-1, 32, 35, 35]           9,216
         BatchNorm2d-261           [-1, 32, 35, 35]              64
                ReLU-262           [-1, 32, 35, 35]               0
         BasicConv2d-263           [-1, 32, 35, 35]               0
              Conv2d-264           [-1, 32, 35, 35]          12,288
         BatchNorm2d-265           [-1, 32, 35, 35]              64
                ReLU-266           [-1, 32, 35, 35]               0
         BasicConv2d-267           [-1, 32, 35, 35]               0
              Conv2d-268           [-1, 48, 35, 35]          13,824
         BatchNorm2d-269           [-1, 48, 35, 35]              96
                ReLU-270           [-1, 48, 35, 35]               0
         BasicConv2d-271           [-1, 48, 35, 35]               0
              Conv2d-272           [-1, 64, 35, 35]          27,648
         BatchNorm2d-273           [-1, 64, 35, 35]             128
                ReLU-274           [-1, 64, 35, 35]               0
         BasicConv2d-275           [-1, 64, 35, 35]               0
              Conv2d-276          [-1, 384, 35, 35]          49,536
         BatchNorm2d-277          [-1, 384, 35, 35]             768
                ReLU-278          [-1, 384, 35, 35]               0
    Inception_ResNet_A-279          [-1, 384, 35, 35]               0
              Conv2d-280          [-1, 384, 35, 35]         147,840
              Conv2d-281           [-1, 32, 35, 35]          12,288
         BatchNorm2d-282           [-1, 32, 35, 35]              64
                ReLU-283           [-1, 32, 35, 35]               0
         BasicConv2d-284           [-1, 32, 35, 35]               0
              Conv2d-285           [-1, 32, 35, 35]          12,288
         BatchNorm2d-286           [-1, 32, 35, 35]              64
                ReLU-287           [-1, 32, 35, 35]               0
         BasicConv2d-288           [-1, 32, 35, 35]               0
              Conv2d-289           [-1, 32, 35, 35]           9,216
         BatchNorm2d-290           [-1, 32, 35, 35]              64
                ReLU-291           [-1, 32, 35, 35]               0
         BasicConv2d-292           [-1, 32, 35, 35]               0
              Conv2d-293           [-1, 32, 35, 35]          12,288
         BatchNorm2d-294           [-1, 32, 35, 35]              64
                ReLU-295           [-1, 32, 35, 35]               0
         BasicConv2d-296           [-1, 32, 35, 35]               0
              Conv2d-297           [-1, 48, 35, 35]          13,824
         BatchNorm2d-298           [-1, 48, 35, 35]              96
                ReLU-299           [-1, 48, 35, 35]               0
         BasicConv2d-300           [-1, 48, 35, 35]               0
              Conv2d-301           [-1, 64, 35, 35]          27,648
         BatchNorm2d-302           [-1, 64, 35, 35]             128
                ReLU-303           [-1, 64, 35, 35]               0
         BasicConv2d-304           [-1, 64, 35, 35]               0
              Conv2d-305          [-1, 384, 35, 35]          49,536
         BatchNorm2d-306          [-1, 384, 35, 35]             768
                ReLU-307          [-1, 384, 35, 35]               0
    Inception_ResNet_A-308          [-1, 384, 35, 35]               0
              Conv2d-309          [-1, 384, 35, 35]         147,840
              Conv2d-310           [-1, 32, 35, 35]          12,288
         BatchNorm2d-311           [-1, 32, 35, 35]              64
                ReLU-312           [-1, 32, 35, 35]               0
         BasicConv2d-313           [-1, 32, 35, 35]               0
              Conv2d-314           [-1, 32, 35, 35]          12,288
         BatchNorm2d-315           [-1, 32, 35, 35]              64
                ReLU-316           [-1, 32, 35, 35]               0
         BasicConv2d-317           [-1, 32, 35, 35]               0
              Conv2d-318           [-1, 32, 35, 35]           9,216
         BatchNorm2d-319           [-1, 32, 35, 35]              64
                ReLU-320           [-1, 32, 35, 35]               0
         BasicConv2d-321           [-1, 32, 35, 35]               0
              Conv2d-322           [-1, 32, 35, 35]          12,288
         BatchNorm2d-323           [-1, 32, 35, 35]              64
                ReLU-324           [-1, 32, 35, 35]               0
         BasicConv2d-325           [-1, 32, 35, 35]               0
              Conv2d-326           [-1, 48, 35, 35]          13,824
         BatchNorm2d-327           [-1, 48, 35, 35]              96
                ReLU-328           [-1, 48, 35, 35]               0
         BasicConv2d-329           [-1, 48, 35, 35]               0
              Conv2d-330           [-1, 64, 35, 35]          27,648
         BatchNorm2d-331           [-1, 64, 35, 35]             128
                ReLU-332           [-1, 64, 35, 35]               0
         BasicConv2d-333           [-1, 64, 35, 35]               0
              Conv2d-334          [-1, 384, 35, 35]          49,536
         BatchNorm2d-335          [-1, 384, 35, 35]             768
                ReLU-336          [-1, 384, 35, 35]               0
    Inception_ResNet_A-337          [-1, 384, 35, 35]               0
           MaxPool2d-338          [-1, 384, 17, 17]               0
              Conv2d-339          [-1, 384, 17, 17]       1,327,104
         BatchNorm2d-340          [-1, 384, 17, 17]             768
                ReLU-341          [-1, 384, 17, 17]               0
         BasicConv2d-342          [-1, 384, 17, 17]               0
              Conv2d-343          [-1, 256, 35, 35]          98,304
         BatchNorm2d-344          [-1, 256, 35, 35]             512
                ReLU-345          [-1, 256, 35, 35]               0
         BasicConv2d-346          [-1, 256, 35, 35]               0
              Conv2d-347          [-1, 256, 35, 35]         589,824
         BatchNorm2d-348          [-1, 256, 35, 35]             512
                ReLU-349          [-1, 256, 35, 35]               0
         BasicConv2d-350          [-1, 256, 35, 35]               0
              Conv2d-351          [-1, 384, 17, 17]         393,216
         BatchNorm2d-352          [-1, 384, 17, 17]             768
                ReLU-353          [-1, 384, 17, 17]               0
         BasicConv2d-354          [-1, 384, 17, 17]               0
          ReductionA-355         [-1, 1152, 17, 17]               0
              Conv2d-356         [-1, 1152, 17, 17]       1,328,256
              Conv2d-357          [-1, 192, 17, 17]         221,184
         BatchNorm2d-358          [-1, 192, 17, 17]             384
                ReLU-359          [-1, 192, 17, 17]               0
         BasicConv2d-360          [-1, 192, 17, 17]               0
              Conv2d-361          [-1, 128, 17, 17]         147,456
         BatchNorm2d-362          [-1, 128, 17, 17]             256
                ReLU-363          [-1, 128, 17, 17]               0
         BasicConv2d-364          [-1, 128, 17, 17]               0
              Conv2d-365          [-1, 160, 17, 17]         143,360
         BatchNorm2d-366          [-1, 160, 17, 17]             320
                ReLU-367          [-1, 160, 17, 17]               0
         BasicConv2d-368          [-1, 160, 17, 17]               0
              Conv2d-369          [-1, 192, 17, 17]         215,040
         BatchNorm2d-370          [-1, 192, 17, 17]             384
                ReLU-371          [-1, 192, 17, 17]               0
         BasicConv2d-372          [-1, 192, 17, 17]               0
              Conv2d-373         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-374         [-1, 1152, 17, 17]           2,304
                ReLU-375         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-376         [-1, 1152, 17, 17]               0
              Conv2d-377         [-1, 1152, 17, 17]       1,328,256
              Conv2d-378          [-1, 192, 17, 17]         221,184
         BatchNorm2d-379          [-1, 192, 17, 17]             384
                ReLU-380          [-1, 192, 17, 17]               0
         BasicConv2d-381          [-1, 192, 17, 17]               0
              Conv2d-382          [-1, 128, 17, 17]         147,456
         BatchNorm2d-383          [-1, 128, 17, 17]             256
                ReLU-384          [-1, 128, 17, 17]               0
         BasicConv2d-385          [-1, 128, 17, 17]               0
              Conv2d-386          [-1, 160, 17, 17]         143,360
         BatchNorm2d-387          [-1, 160, 17, 17]             320
                ReLU-388          [-1, 160, 17, 17]               0
         BasicConv2d-389          [-1, 160, 17, 17]               0
              Conv2d-390          [-1, 192, 17, 17]         215,040
         BatchNorm2d-391          [-1, 192, 17, 17]             384
                ReLU-392          [-1, 192, 17, 17]               0
         BasicConv2d-393          [-1, 192, 17, 17]               0
              Conv2d-394         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-395         [-1, 1152, 17, 17]           2,304
                ReLU-396         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-397         [-1, 1152, 17, 17]               0
              Conv2d-398         [-1, 1152, 17, 17]       1,328,256
              Conv2d-399          [-1, 192, 17, 17]         221,184
         BatchNorm2d-400          [-1, 192, 17, 17]             384
                ReLU-401          [-1, 192, 17, 17]               0
         BasicConv2d-402          [-1, 192, 17, 17]               0
              Conv2d-403          [-1, 128, 17, 17]         147,456
         BatchNorm2d-404          [-1, 128, 17, 17]             256
                ReLU-405          [-1, 128, 17, 17]               0
         BasicConv2d-406          [-1, 128, 17, 17]               0
              Conv2d-407          [-1, 160, 17, 17]         143,360
         BatchNorm2d-408          [-1, 160, 17, 17]             320
                ReLU-409          [-1, 160, 17, 17]               0
         BasicConv2d-410          [-1, 160, 17, 17]               0
              Conv2d-411          [-1, 192, 17, 17]         215,040
         BatchNorm2d-412          [-1, 192, 17, 17]             384
                ReLU-413          [-1, 192, 17, 17]               0
         BasicConv2d-414          [-1, 192, 17, 17]               0
              Conv2d-415         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-416         [-1, 1152, 17, 17]           2,304
                ReLU-417         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-418         [-1, 1152, 17, 17]               0
              Conv2d-419         [-1, 1152, 17, 17]       1,328,256
              Conv2d-420          [-1, 192, 17, 17]         221,184
         BatchNorm2d-421          [-1, 192, 17, 17]             384
                ReLU-422          [-1, 192, 17, 17]               0
         BasicConv2d-423          [-1, 192, 17, 17]               0
              Conv2d-424          [-1, 128, 17, 17]         147,456
         BatchNorm2d-425          [-1, 128, 17, 17]             256
                ReLU-426          [-1, 128, 17, 17]               0
         BasicConv2d-427          [-1, 128, 17, 17]               0
              Conv2d-428          [-1, 160, 17, 17]         143,360
         BatchNorm2d-429          [-1, 160, 17, 17]             320
                ReLU-430          [-1, 160, 17, 17]               0
         BasicConv2d-431          [-1, 160, 17, 17]               0
              Conv2d-432          [-1, 192, 17, 17]         215,040
         BatchNorm2d-433          [-1, 192, 17, 17]             384
                ReLU-434          [-1, 192, 17, 17]               0
         BasicConv2d-435          [-1, 192, 17, 17]               0
              Conv2d-436         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-437         [-1, 1152, 17, 17]           2,304
                ReLU-438         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-439         [-1, 1152, 17, 17]               0
              Conv2d-440         [-1, 1152, 17, 17]       1,328,256
              Conv2d-441          [-1, 192, 17, 17]         221,184
         BatchNorm2d-442          [-1, 192, 17, 17]             384
                ReLU-443          [-1, 192, 17, 17]               0
         BasicConv2d-444          [-1, 192, 17, 17]               0
              Conv2d-445          [-1, 128, 17, 17]         147,456
         BatchNorm2d-446          [-1, 128, 17, 17]             256
                ReLU-447          [-1, 128, 17, 17]               0
         BasicConv2d-448          [-1, 128, 17, 17]               0
              Conv2d-449          [-1, 160, 17, 17]         143,360
         BatchNorm2d-450          [-1, 160, 17, 17]             320
                ReLU-451          [-1, 160, 17, 17]               0
         BasicConv2d-452          [-1, 160, 17, 17]               0
              Conv2d-453          [-1, 192, 17, 17]         215,040
         BatchNorm2d-454          [-1, 192, 17, 17]             384
                ReLU-455          [-1, 192, 17, 17]               0
         BasicConv2d-456          [-1, 192, 17, 17]               0
              Conv2d-457         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-458         [-1, 1152, 17, 17]           2,304
                ReLU-459         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-460         [-1, 1152, 17, 17]               0
              Conv2d-461         [-1, 1152, 17, 17]       1,328,256
              Conv2d-462          [-1, 192, 17, 17]         221,184
         BatchNorm2d-463          [-1, 192, 17, 17]             384
                ReLU-464          [-1, 192, 17, 17]               0
         BasicConv2d-465          [-1, 192, 17, 17]               0
              Conv2d-466          [-1, 128, 17, 17]         147,456
         BatchNorm2d-467          [-1, 128, 17, 17]             256
                ReLU-468          [-1, 128, 17, 17]               0
         BasicConv2d-469          [-1, 128, 17, 17]               0
              Conv2d-470          [-1, 160, 17, 17]         143,360
         BatchNorm2d-471          [-1, 160, 17, 17]             320
                ReLU-472          [-1, 160, 17, 17]               0
         BasicConv2d-473          [-1, 160, 17, 17]               0
              Conv2d-474          [-1, 192, 17, 17]         215,040
         BatchNorm2d-475          [-1, 192, 17, 17]             384
                ReLU-476          [-1, 192, 17, 17]               0
         BasicConv2d-477          [-1, 192, 17, 17]               0
              Conv2d-478         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-479         [-1, 1152, 17, 17]           2,304
                ReLU-480         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-481         [-1, 1152, 17, 17]               0
              Conv2d-482         [-1, 1152, 17, 17]       1,328,256
              Conv2d-483          [-1, 192, 17, 17]         221,184
         BatchNorm2d-484          [-1, 192, 17, 17]             384
                ReLU-485          [-1, 192, 17, 17]               0
         BasicConv2d-486          [-1, 192, 17, 17]               0
              Conv2d-487          [-1, 128, 17, 17]         147,456
         BatchNorm2d-488          [-1, 128, 17, 17]             256
                ReLU-489          [-1, 128, 17, 17]               0
         BasicConv2d-490          [-1, 128, 17, 17]               0
              Conv2d-491          [-1, 160, 17, 17]         143,360
         BatchNorm2d-492          [-1, 160, 17, 17]             320
                ReLU-493          [-1, 160, 17, 17]               0
         BasicConv2d-494          [-1, 160, 17, 17]               0
              Conv2d-495          [-1, 192, 17, 17]         215,040
         BatchNorm2d-496          [-1, 192, 17, 17]             384
                ReLU-497          [-1, 192, 17, 17]               0
         BasicConv2d-498          [-1, 192, 17, 17]               0
              Conv2d-499         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-500         [-1, 1152, 17, 17]           2,304
                ReLU-501         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-502         [-1, 1152, 17, 17]               0
              Conv2d-503         [-1, 1152, 17, 17]       1,328,256
              Conv2d-504          [-1, 192, 17, 17]         221,184
         BatchNorm2d-505          [-1, 192, 17, 17]             384
                ReLU-506          [-1, 192, 17, 17]               0
         BasicConv2d-507          [-1, 192, 17, 17]               0
              Conv2d-508          [-1, 128, 17, 17]         147,456
         BatchNorm2d-509          [-1, 128, 17, 17]             256
                ReLU-510          [-1, 128, 17, 17]               0
         BasicConv2d-511          [-1, 128, 17, 17]               0
              Conv2d-512          [-1, 160, 17, 17]         143,360
         BatchNorm2d-513          [-1, 160, 17, 17]             320
                ReLU-514          [-1, 160, 17, 17]               0
         BasicConv2d-515          [-1, 160, 17, 17]               0
              Conv2d-516          [-1, 192, 17, 17]         215,040
         BatchNorm2d-517          [-1, 192, 17, 17]             384
                ReLU-518          [-1, 192, 17, 17]               0
         BasicConv2d-519          [-1, 192, 17, 17]               0
              Conv2d-520         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-521         [-1, 1152, 17, 17]           2,304
                ReLU-522         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-523         [-1, 1152, 17, 17]               0
              Conv2d-524         [-1, 1152, 17, 17]       1,328,256
              Conv2d-525          [-1, 192, 17, 17]         221,184
         BatchNorm2d-526          [-1, 192, 17, 17]             384
                ReLU-527          [-1, 192, 17, 17]               0
         BasicConv2d-528          [-1, 192, 17, 17]               0
              Conv2d-529          [-1, 128, 17, 17]         147,456
         BatchNorm2d-530          [-1, 128, 17, 17]             256
                ReLU-531          [-1, 128, 17, 17]               0
         BasicConv2d-532          [-1, 128, 17, 17]               0
              Conv2d-533          [-1, 160, 17, 17]         143,360
         BatchNorm2d-534          [-1, 160, 17, 17]             320
                ReLU-535          [-1, 160, 17, 17]               0
         BasicConv2d-536          [-1, 160, 17, 17]               0
              Conv2d-537          [-1, 192, 17, 17]         215,040
         BatchNorm2d-538          [-1, 192, 17, 17]             384
                ReLU-539          [-1, 192, 17, 17]               0
         BasicConv2d-540          [-1, 192, 17, 17]               0
              Conv2d-541         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-542         [-1, 1152, 17, 17]           2,304
                ReLU-543         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-544         [-1, 1152, 17, 17]               0
              Conv2d-545         [-1, 1152, 17, 17]       1,328,256
              Conv2d-546          [-1, 192, 17, 17]         221,184
         BatchNorm2d-547          [-1, 192, 17, 17]             384
                ReLU-548          [-1, 192, 17, 17]               0
         BasicConv2d-549          [-1, 192, 17, 17]               0
              Conv2d-550          [-1, 128, 17, 17]         147,456
         BatchNorm2d-551          [-1, 128, 17, 17]             256
                ReLU-552          [-1, 128, 17, 17]               0
         BasicConv2d-553          [-1, 128, 17, 17]               0
              Conv2d-554          [-1, 160, 17, 17]         143,360
         BatchNorm2d-555          [-1, 160, 17, 17]             320
                ReLU-556          [-1, 160, 17, 17]               0
         BasicConv2d-557          [-1, 160, 17, 17]               0
              Conv2d-558          [-1, 192, 17, 17]         215,040
         BatchNorm2d-559          [-1, 192, 17, 17]             384
                ReLU-560          [-1, 192, 17, 17]               0
         BasicConv2d-561          [-1, 192, 17, 17]               0
              Conv2d-562         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-563         [-1, 1152, 17, 17]           2,304
                ReLU-564         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-565         [-1, 1152, 17, 17]               0
              Conv2d-566         [-1, 1152, 17, 17]       1,328,256
              Conv2d-567          [-1, 192, 17, 17]         221,184
         BatchNorm2d-568          [-1, 192, 17, 17]             384
                ReLU-569          [-1, 192, 17, 17]               0
         BasicConv2d-570          [-1, 192, 17, 17]               0
              Conv2d-571          [-1, 128, 17, 17]         147,456
         BatchNorm2d-572          [-1, 128, 17, 17]             256
                ReLU-573          [-1, 128, 17, 17]               0
         BasicConv2d-574          [-1, 128, 17, 17]               0
              Conv2d-575          [-1, 160, 17, 17]         143,360
         BatchNorm2d-576          [-1, 160, 17, 17]             320
                ReLU-577          [-1, 160, 17, 17]               0
         BasicConv2d-578          [-1, 160, 17, 17]               0
              Conv2d-579          [-1, 192, 17, 17]         215,040
         BatchNorm2d-580          [-1, 192, 17, 17]             384
                ReLU-581          [-1, 192, 17, 17]               0
         BasicConv2d-582          [-1, 192, 17, 17]               0
              Conv2d-583         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-584         [-1, 1152, 17, 17]           2,304
                ReLU-585         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-586         [-1, 1152, 17, 17]               0
              Conv2d-587         [-1, 1152, 17, 17]       1,328,256
              Conv2d-588          [-1, 192, 17, 17]         221,184
         BatchNorm2d-589          [-1, 192, 17, 17]             384
                ReLU-590          [-1, 192, 17, 17]               0
         BasicConv2d-591          [-1, 192, 17, 17]               0
              Conv2d-592          [-1, 128, 17, 17]         147,456
         BatchNorm2d-593          [-1, 128, 17, 17]             256
                ReLU-594          [-1, 128, 17, 17]               0
         BasicConv2d-595          [-1, 128, 17, 17]               0
              Conv2d-596          [-1, 160, 17, 17]         143,360
         BatchNorm2d-597          [-1, 160, 17, 17]             320
                ReLU-598          [-1, 160, 17, 17]               0
         BasicConv2d-599          [-1, 160, 17, 17]               0
              Conv2d-600          [-1, 192, 17, 17]         215,040
         BatchNorm2d-601          [-1, 192, 17, 17]             384
                ReLU-602          [-1, 192, 17, 17]               0
         BasicConv2d-603          [-1, 192, 17, 17]               0
              Conv2d-604         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-605         [-1, 1152, 17, 17]           2,304
                ReLU-606         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-607         [-1, 1152, 17, 17]               0
              Conv2d-608         [-1, 1152, 17, 17]       1,328,256
              Conv2d-609          [-1, 192, 17, 17]         221,184
         BatchNorm2d-610          [-1, 192, 17, 17]             384
                ReLU-611          [-1, 192, 17, 17]               0
         BasicConv2d-612          [-1, 192, 17, 17]               0
              Conv2d-613          [-1, 128, 17, 17]         147,456
         BatchNorm2d-614          [-1, 128, 17, 17]             256
                ReLU-615          [-1, 128, 17, 17]               0
         BasicConv2d-616          [-1, 128, 17, 17]               0
              Conv2d-617          [-1, 160, 17, 17]         143,360
         BatchNorm2d-618          [-1, 160, 17, 17]             320
                ReLU-619          [-1, 160, 17, 17]               0
         BasicConv2d-620          [-1, 160, 17, 17]               0
              Conv2d-621          [-1, 192, 17, 17]         215,040
         BatchNorm2d-622          [-1, 192, 17, 17]             384
                ReLU-623          [-1, 192, 17, 17]               0
         BasicConv2d-624          [-1, 192, 17, 17]               0
              Conv2d-625         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-626         [-1, 1152, 17, 17]           2,304
                ReLU-627         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-628         [-1, 1152, 17, 17]               0
              Conv2d-629         [-1, 1152, 17, 17]       1,328,256
              Conv2d-630          [-1, 192, 17, 17]         221,184
         BatchNorm2d-631          [-1, 192, 17, 17]             384
                ReLU-632          [-1, 192, 17, 17]               0
         BasicConv2d-633          [-1, 192, 17, 17]               0
              Conv2d-634          [-1, 128, 17, 17]         147,456
         BatchNorm2d-635          [-1, 128, 17, 17]             256
                ReLU-636          [-1, 128, 17, 17]               0
         BasicConv2d-637          [-1, 128, 17, 17]               0
              Conv2d-638          [-1, 160, 17, 17]         143,360
         BatchNorm2d-639          [-1, 160, 17, 17]             320
                ReLU-640          [-1, 160, 17, 17]               0
         BasicConv2d-641          [-1, 160, 17, 17]               0
              Conv2d-642          [-1, 192, 17, 17]         215,040
         BatchNorm2d-643          [-1, 192, 17, 17]             384
                ReLU-644          [-1, 192, 17, 17]               0
         BasicConv2d-645          [-1, 192, 17, 17]               0
              Conv2d-646         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-647         [-1, 1152, 17, 17]           2,304
                ReLU-648         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-649         [-1, 1152, 17, 17]               0
              Conv2d-650         [-1, 1152, 17, 17]       1,328,256
              Conv2d-651          [-1, 192, 17, 17]         221,184
         BatchNorm2d-652          [-1, 192, 17, 17]             384
                ReLU-653          [-1, 192, 17, 17]               0
         BasicConv2d-654          [-1, 192, 17, 17]               0
              Conv2d-655          [-1, 128, 17, 17]         147,456
         BatchNorm2d-656          [-1, 128, 17, 17]             256
                ReLU-657          [-1, 128, 17, 17]               0
         BasicConv2d-658          [-1, 128, 17, 17]               0
              Conv2d-659          [-1, 160, 17, 17]         143,360
         BatchNorm2d-660          [-1, 160, 17, 17]             320
                ReLU-661          [-1, 160, 17, 17]               0
         BasicConv2d-662          [-1, 160, 17, 17]               0
              Conv2d-663          [-1, 192, 17, 17]         215,040
         BatchNorm2d-664          [-1, 192, 17, 17]             384
                ReLU-665          [-1, 192, 17, 17]               0
         BasicConv2d-666          [-1, 192, 17, 17]               0
              Conv2d-667         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-668         [-1, 1152, 17, 17]           2,304
                ReLU-669         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-670         [-1, 1152, 17, 17]               0
              Conv2d-671         [-1, 1152, 17, 17]       1,328,256
              Conv2d-672          [-1, 192, 17, 17]         221,184
         BatchNorm2d-673          [-1, 192, 17, 17]             384
                ReLU-674          [-1, 192, 17, 17]               0
         BasicConv2d-675          [-1, 192, 17, 17]               0
              Conv2d-676          [-1, 128, 17, 17]         147,456
         BatchNorm2d-677          [-1, 128, 17, 17]             256
                ReLU-678          [-1, 128, 17, 17]               0
         BasicConv2d-679          [-1, 128, 17, 17]               0
              Conv2d-680          [-1, 160, 17, 17]         143,360
         BatchNorm2d-681          [-1, 160, 17, 17]             320
                ReLU-682          [-1, 160, 17, 17]               0
         BasicConv2d-683          [-1, 160, 17, 17]               0
              Conv2d-684          [-1, 192, 17, 17]         215,040
         BatchNorm2d-685          [-1, 192, 17, 17]             384
                ReLU-686          [-1, 192, 17, 17]               0
         BasicConv2d-687          [-1, 192, 17, 17]               0
              Conv2d-688         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-689         [-1, 1152, 17, 17]           2,304
                ReLU-690         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-691         [-1, 1152, 17, 17]               0
              Conv2d-692         [-1, 1152, 17, 17]       1,328,256
              Conv2d-693          [-1, 192, 17, 17]         221,184
         BatchNorm2d-694          [-1, 192, 17, 17]             384
                ReLU-695          [-1, 192, 17, 17]               0
         BasicConv2d-696          [-1, 192, 17, 17]               0
              Conv2d-697          [-1, 128, 17, 17]         147,456
         BatchNorm2d-698          [-1, 128, 17, 17]             256
                ReLU-699          [-1, 128, 17, 17]               0
         BasicConv2d-700          [-1, 128, 17, 17]               0
              Conv2d-701          [-1, 160, 17, 17]         143,360
         BatchNorm2d-702          [-1, 160, 17, 17]             320
                ReLU-703          [-1, 160, 17, 17]               0
         BasicConv2d-704          [-1, 160, 17, 17]               0
              Conv2d-705          [-1, 192, 17, 17]         215,040
         BatchNorm2d-706          [-1, 192, 17, 17]             384
                ReLU-707          [-1, 192, 17, 17]               0
         BasicConv2d-708          [-1, 192, 17, 17]               0
              Conv2d-709         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-710         [-1, 1152, 17, 17]           2,304
                ReLU-711         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-712         [-1, 1152, 17, 17]               0
              Conv2d-713         [-1, 1152, 17, 17]       1,328,256
              Conv2d-714          [-1, 192, 17, 17]         221,184
         BatchNorm2d-715          [-1, 192, 17, 17]             384
                ReLU-716          [-1, 192, 17, 17]               0
         BasicConv2d-717          [-1, 192, 17, 17]               0
              Conv2d-718          [-1, 128, 17, 17]         147,456
         BatchNorm2d-719          [-1, 128, 17, 17]             256
                ReLU-720          [-1, 128, 17, 17]               0
         BasicConv2d-721          [-1, 128, 17, 17]               0
              Conv2d-722          [-1, 160, 17, 17]         143,360
         BatchNorm2d-723          [-1, 160, 17, 17]             320
                ReLU-724          [-1, 160, 17, 17]               0
         BasicConv2d-725          [-1, 160, 17, 17]               0
              Conv2d-726          [-1, 192, 17, 17]         215,040
         BatchNorm2d-727          [-1, 192, 17, 17]             384
                ReLU-728          [-1, 192, 17, 17]               0
         BasicConv2d-729          [-1, 192, 17, 17]               0
              Conv2d-730         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-731         [-1, 1152, 17, 17]           2,304
                ReLU-732         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-733         [-1, 1152, 17, 17]               0
              Conv2d-734         [-1, 1152, 17, 17]       1,328,256
              Conv2d-735          [-1, 192, 17, 17]         221,184
         BatchNorm2d-736          [-1, 192, 17, 17]             384
                ReLU-737          [-1, 192, 17, 17]               0
         BasicConv2d-738          [-1, 192, 17, 17]               0
              Conv2d-739          [-1, 128, 17, 17]         147,456
         BatchNorm2d-740          [-1, 128, 17, 17]             256
                ReLU-741          [-1, 128, 17, 17]               0
         BasicConv2d-742          [-1, 128, 17, 17]               0
              Conv2d-743          [-1, 160, 17, 17]         143,360
         BatchNorm2d-744          [-1, 160, 17, 17]             320
                ReLU-745          [-1, 160, 17, 17]               0
         BasicConv2d-746          [-1, 160, 17, 17]               0
              Conv2d-747          [-1, 192, 17, 17]         215,040
         BatchNorm2d-748          [-1, 192, 17, 17]             384
                ReLU-749          [-1, 192, 17, 17]               0
         BasicConv2d-750          [-1, 192, 17, 17]               0
              Conv2d-751         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-752         [-1, 1152, 17, 17]           2,304
                ReLU-753         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-754         [-1, 1152, 17, 17]               0
              Conv2d-755         [-1, 1152, 17, 17]       1,328,256
              Conv2d-756          [-1, 192, 17, 17]         221,184
         BatchNorm2d-757          [-1, 192, 17, 17]             384
                ReLU-758          [-1, 192, 17, 17]               0
         BasicConv2d-759          [-1, 192, 17, 17]               0
              Conv2d-760          [-1, 128, 17, 17]         147,456
         BatchNorm2d-761          [-1, 128, 17, 17]             256
                ReLU-762          [-1, 128, 17, 17]               0
         BasicConv2d-763          [-1, 128, 17, 17]               0
              Conv2d-764          [-1, 160, 17, 17]         143,360
         BatchNorm2d-765          [-1, 160, 17, 17]             320
                ReLU-766          [-1, 160, 17, 17]               0
         BasicConv2d-767          [-1, 160, 17, 17]               0
              Conv2d-768          [-1, 192, 17, 17]         215,040
         BatchNorm2d-769          [-1, 192, 17, 17]             384
                ReLU-770          [-1, 192, 17, 17]               0
         BasicConv2d-771          [-1, 192, 17, 17]               0
              Conv2d-772         [-1, 1152, 17, 17]         443,520
         BatchNorm2d-773         [-1, 1152, 17, 17]           2,304
                ReLU-774         [-1, 1152, 17, 17]               0
    Inception_ResNet_B-775         [-1, 1152, 17, 17]               0
           MaxPool2d-776           [-1, 1152, 8, 8]               0
              Conv2d-777          [-1, 256, 17, 17]         294,912
         BatchNorm2d-778          [-1, 256, 17, 17]             512
                ReLU-779          [-1, 256, 17, 17]               0
         BasicConv2d-780          [-1, 256, 17, 17]               0
              Conv2d-781            [-1, 384, 8, 8]         884,736
         BatchNorm2d-782            [-1, 384, 8, 8]             768
                ReLU-783            [-1, 384, 8, 8]               0
         BasicConv2d-784            [-1, 384, 8, 8]               0
              Conv2d-785          [-1, 256, 17, 17]         294,912
         BatchNorm2d-786          [-1, 256, 17, 17]             512
                ReLU-787          [-1, 256, 17, 17]               0
         BasicConv2d-788          [-1, 256, 17, 17]               0
              Conv2d-789            [-1, 288, 8, 8]         663,552
         BatchNorm2d-790            [-1, 288, 8, 8]             576
                ReLU-791            [-1, 288, 8, 8]               0
         BasicConv2d-792            [-1, 288, 8, 8]               0
              Conv2d-793          [-1, 256, 17, 17]         294,912
         BatchNorm2d-794          [-1, 256, 17, 17]             512
                ReLU-795          [-1, 256, 17, 17]               0
         BasicConv2d-796          [-1, 256, 17, 17]               0
              Conv2d-797          [-1, 288, 17, 17]         663,552
         BatchNorm2d-798          [-1, 288, 17, 17]             576
                ReLU-799          [-1, 288, 17, 17]               0
         BasicConv2d-800          [-1, 288, 17, 17]               0
              Conv2d-801            [-1, 320, 8, 8]         829,440
         BatchNorm2d-802            [-1, 320, 8, 8]             640
                ReLU-803            [-1, 320, 8, 8]               0
         BasicConv2d-804            [-1, 320, 8, 8]               0
          ReductionB-805           [-1, 2144, 8, 8]               0
              Conv2d-806           [-1, 2144, 8, 8]       4,598,880
              Conv2d-807            [-1, 192, 8, 8]         411,648
         BatchNorm2d-808            [-1, 192, 8, 8]             384
                ReLU-809            [-1, 192, 8, 8]               0
         BasicConv2d-810            [-1, 192, 8, 8]               0
              Conv2d-811            [-1, 192, 8, 8]         411,648
         BatchNorm2d-812            [-1, 192, 8, 8]             384
                ReLU-813            [-1, 192, 8, 8]               0
         BasicConv2d-814            [-1, 192, 8, 8]               0
              Conv2d-815            [-1, 224, 8, 8]         129,024
         BatchNorm2d-816            [-1, 224, 8, 8]             448
                ReLU-817            [-1, 224, 8, 8]               0
         BasicConv2d-818            [-1, 224, 8, 8]               0
              Conv2d-819            [-1, 256, 8, 8]         172,032
         BatchNorm2d-820            [-1, 256, 8, 8]             512
                ReLU-821            [-1, 256, 8, 8]               0
         BasicConv2d-822            [-1, 256, 8, 8]               0
              Conv2d-823           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-824           [-1, 2144, 8, 8]           4,288
                ReLU-825           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-826           [-1, 2144, 8, 8]               0
              Conv2d-827           [-1, 2144, 8, 8]       4,598,880
              Conv2d-828            [-1, 192, 8, 8]         411,648
         BatchNorm2d-829            [-1, 192, 8, 8]             384
                ReLU-830            [-1, 192, 8, 8]               0
         BasicConv2d-831            [-1, 192, 8, 8]               0
              Conv2d-832            [-1, 192, 8, 8]         411,648
         BatchNorm2d-833            [-1, 192, 8, 8]             384
                ReLU-834            [-1, 192, 8, 8]               0
         BasicConv2d-835            [-1, 192, 8, 8]               0
              Conv2d-836            [-1, 224, 8, 8]         129,024
         BatchNorm2d-837            [-1, 224, 8, 8]             448
                ReLU-838            [-1, 224, 8, 8]               0
         BasicConv2d-839            [-1, 224, 8, 8]               0
              Conv2d-840            [-1, 256, 8, 8]         172,032
         BatchNorm2d-841            [-1, 256, 8, 8]             512
                ReLU-842            [-1, 256, 8, 8]               0
         BasicConv2d-843            [-1, 256, 8, 8]               0
              Conv2d-844           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-845           [-1, 2144, 8, 8]           4,288
                ReLU-846           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-847           [-1, 2144, 8, 8]               0
              Conv2d-848           [-1, 2144, 8, 8]       4,598,880
              Conv2d-849            [-1, 192, 8, 8]         411,648
         BatchNorm2d-850            [-1, 192, 8, 8]             384
                ReLU-851            [-1, 192, 8, 8]               0
         BasicConv2d-852            [-1, 192, 8, 8]               0
              Conv2d-853            [-1, 192, 8, 8]         411,648
         BatchNorm2d-854            [-1, 192, 8, 8]             384
                ReLU-855            [-1, 192, 8, 8]               0
         BasicConv2d-856            [-1, 192, 8, 8]               0
              Conv2d-857            [-1, 224, 8, 8]         129,024
         BatchNorm2d-858            [-1, 224, 8, 8]             448
                ReLU-859            [-1, 224, 8, 8]               0
         BasicConv2d-860            [-1, 224, 8, 8]               0
              Conv2d-861            [-1, 256, 8, 8]         172,032
         BatchNorm2d-862            [-1, 256, 8, 8]             512
                ReLU-863            [-1, 256, 8, 8]               0
         BasicConv2d-864            [-1, 256, 8, 8]               0
              Conv2d-865           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-866           [-1, 2144, 8, 8]           4,288
                ReLU-867           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-868           [-1, 2144, 8, 8]               0
              Conv2d-869           [-1, 2144, 8, 8]       4,598,880
              Conv2d-870            [-1, 192, 8, 8]         411,648
         BatchNorm2d-871            [-1, 192, 8, 8]             384
                ReLU-872            [-1, 192, 8, 8]               0
         BasicConv2d-873            [-1, 192, 8, 8]               0
              Conv2d-874            [-1, 192, 8, 8]         411,648
         BatchNorm2d-875            [-1, 192, 8, 8]             384
                ReLU-876            [-1, 192, 8, 8]               0
         BasicConv2d-877            [-1, 192, 8, 8]               0
              Conv2d-878            [-1, 224, 8, 8]         129,024
         BatchNorm2d-879            [-1, 224, 8, 8]             448
                ReLU-880            [-1, 224, 8, 8]               0
         BasicConv2d-881            [-1, 224, 8, 8]               0
              Conv2d-882            [-1, 256, 8, 8]         172,032
         BatchNorm2d-883            [-1, 256, 8, 8]             512
                ReLU-884            [-1, 256, 8, 8]               0
         BasicConv2d-885            [-1, 256, 8, 8]               0
              Conv2d-886           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-887           [-1, 2144, 8, 8]           4,288
                ReLU-888           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-889           [-1, 2144, 8, 8]               0
              Conv2d-890           [-1, 2144, 8, 8]       4,598,880
              Conv2d-891            [-1, 192, 8, 8]         411,648
         BatchNorm2d-892            [-1, 192, 8, 8]             384
                ReLU-893            [-1, 192, 8, 8]               0
         BasicConv2d-894            [-1, 192, 8, 8]               0
              Conv2d-895            [-1, 192, 8, 8]         411,648
         BatchNorm2d-896            [-1, 192, 8, 8]             384
                ReLU-897            [-1, 192, 8, 8]               0
         BasicConv2d-898            [-1, 192, 8, 8]               0
              Conv2d-899            [-1, 224, 8, 8]         129,024
         BatchNorm2d-900            [-1, 224, 8, 8]             448
                ReLU-901            [-1, 224, 8, 8]               0
         BasicConv2d-902            [-1, 224, 8, 8]               0
              Conv2d-903            [-1, 256, 8, 8]         172,032
         BatchNorm2d-904            [-1, 256, 8, 8]             512
                ReLU-905            [-1, 256, 8, 8]               0
         BasicConv2d-906            [-1, 256, 8, 8]               0
              Conv2d-907           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-908           [-1, 2144, 8, 8]           4,288
                ReLU-909           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-910           [-1, 2144, 8, 8]               0
              Conv2d-911           [-1, 2144, 8, 8]       4,598,880
              Conv2d-912            [-1, 192, 8, 8]         411,648
         BatchNorm2d-913            [-1, 192, 8, 8]             384
                ReLU-914            [-1, 192, 8, 8]               0
         BasicConv2d-915            [-1, 192, 8, 8]               0
              Conv2d-916            [-1, 192, 8, 8]         411,648
         BatchNorm2d-917            [-1, 192, 8, 8]             384
                ReLU-918            [-1, 192, 8, 8]               0
         BasicConv2d-919            [-1, 192, 8, 8]               0
              Conv2d-920            [-1, 224, 8, 8]         129,024
         BatchNorm2d-921            [-1, 224, 8, 8]             448
                ReLU-922            [-1, 224, 8, 8]               0
         BasicConv2d-923            [-1, 224, 8, 8]               0
              Conv2d-924            [-1, 256, 8, 8]         172,032
         BatchNorm2d-925            [-1, 256, 8, 8]             512
                ReLU-926            [-1, 256, 8, 8]               0
         BasicConv2d-927            [-1, 256, 8, 8]               0
              Conv2d-928           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-929           [-1, 2144, 8, 8]           4,288
                ReLU-930           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-931           [-1, 2144, 8, 8]               0
              Conv2d-932           [-1, 2144, 8, 8]       4,598,880
              Conv2d-933            [-1, 192, 8, 8]         411,648
         BatchNorm2d-934            [-1, 192, 8, 8]             384
                ReLU-935            [-1, 192, 8, 8]               0
         BasicConv2d-936            [-1, 192, 8, 8]               0
              Conv2d-937            [-1, 192, 8, 8]         411,648
         BatchNorm2d-938            [-1, 192, 8, 8]             384
                ReLU-939            [-1, 192, 8, 8]               0
         BasicConv2d-940            [-1, 192, 8, 8]               0
              Conv2d-941            [-1, 224, 8, 8]         129,024
         BatchNorm2d-942            [-1, 224, 8, 8]             448
                ReLU-943            [-1, 224, 8, 8]               0
         BasicConv2d-944            [-1, 224, 8, 8]               0
              Conv2d-945            [-1, 256, 8, 8]         172,032
         BatchNorm2d-946            [-1, 256, 8, 8]             512
                ReLU-947            [-1, 256, 8, 8]               0
         BasicConv2d-948            [-1, 256, 8, 8]               0
              Conv2d-949           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-950           [-1, 2144, 8, 8]           4,288
                ReLU-951           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-952           [-1, 2144, 8, 8]               0
              Conv2d-953           [-1, 2144, 8, 8]       4,598,880
              Conv2d-954            [-1, 192, 8, 8]         411,648
         BatchNorm2d-955            [-1, 192, 8, 8]             384
                ReLU-956            [-1, 192, 8, 8]               0
         BasicConv2d-957            [-1, 192, 8, 8]               0
              Conv2d-958            [-1, 192, 8, 8]         411,648
         BatchNorm2d-959            [-1, 192, 8, 8]             384
                ReLU-960            [-1, 192, 8, 8]               0
         BasicConv2d-961            [-1, 192, 8, 8]               0
              Conv2d-962            [-1, 224, 8, 8]         129,024
         BatchNorm2d-963            [-1, 224, 8, 8]             448
                ReLU-964            [-1, 224, 8, 8]               0
         BasicConv2d-965            [-1, 224, 8, 8]               0
              Conv2d-966            [-1, 256, 8, 8]         172,032
         BatchNorm2d-967            [-1, 256, 8, 8]             512
                ReLU-968            [-1, 256, 8, 8]               0
         BasicConv2d-969            [-1, 256, 8, 8]               0
              Conv2d-970           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-971           [-1, 2144, 8, 8]           4,288
                ReLU-972           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-973           [-1, 2144, 8, 8]               0
              Conv2d-974           [-1, 2144, 8, 8]       4,598,880
              Conv2d-975            [-1, 192, 8, 8]         411,648
         BatchNorm2d-976            [-1, 192, 8, 8]             384
                ReLU-977            [-1, 192, 8, 8]               0
         BasicConv2d-978            [-1, 192, 8, 8]               0
              Conv2d-979            [-1, 192, 8, 8]         411,648
         BatchNorm2d-980            [-1, 192, 8, 8]             384
                ReLU-981            [-1, 192, 8, 8]               0
         BasicConv2d-982            [-1, 192, 8, 8]               0
              Conv2d-983            [-1, 224, 8, 8]         129,024
         BatchNorm2d-984            [-1, 224, 8, 8]             448
                ReLU-985            [-1, 224, 8, 8]               0
         BasicConv2d-986            [-1, 224, 8, 8]               0
              Conv2d-987            [-1, 256, 8, 8]         172,032
         BatchNorm2d-988            [-1, 256, 8, 8]             512
                ReLU-989            [-1, 256, 8, 8]               0
         BasicConv2d-990            [-1, 256, 8, 8]               0
              Conv2d-991           [-1, 2144, 8, 8]         962,656
         BatchNorm2d-992           [-1, 2144, 8, 8]           4,288
                ReLU-993           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-994           [-1, 2144, 8, 8]               0
              Conv2d-995           [-1, 2144, 8, 8]       4,598,880
              Conv2d-996            [-1, 192, 8, 8]         411,648
         BatchNorm2d-997            [-1, 192, 8, 8]             384
                ReLU-998            [-1, 192, 8, 8]               0
         BasicConv2d-999            [-1, 192, 8, 8]               0
             Conv2d-1000            [-1, 192, 8, 8]         411,648
        BatchNorm2d-1001            [-1, 192, 8, 8]             384
               ReLU-1002            [-1, 192, 8, 8]               0
        BasicConv2d-1003            [-1, 192, 8, 8]               0
             Conv2d-1004            [-1, 224, 8, 8]         129,024
        BatchNorm2d-1005            [-1, 224, 8, 8]             448
               ReLU-1006            [-1, 224, 8, 8]               0
        BasicConv2d-1007            [-1, 224, 8, 8]               0
             Conv2d-1008            [-1, 256, 8, 8]         172,032
        BatchNorm2d-1009            [-1, 256, 8, 8]             512
               ReLU-1010            [-1, 256, 8, 8]               0
        BasicConv2d-1011            [-1, 256, 8, 8]               0
             Conv2d-1012           [-1, 2144, 8, 8]         962,656
        BatchNorm2d-1013           [-1, 2144, 8, 8]           4,288
               ReLU-1014           [-1, 2144, 8, 8]               0
    Inception_ResNet_C-1015           [-1, 2144, 8, 8]               0
    AdaptiveAvgPool2d-1016           [-1, 2144, 1, 1]               0
          Dropout2d-1017                 [-1, 2144]               0
             Linear-1018                   [-1, 10]          21,450
    ================================================================
    Total params: 126,798,378
    Trainable params: 126,798,378
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 1.02
    Forward/backward pass size (MB): 940.05
    Params size (MB): 483.70
    Estimated Total Size (MB): 1424.77
    ----------------------------------------------------------------


참고한 repository에서는 STL10 dataset으로 훈련도 진행하지만 이번 글에서는 모델을 pytorch 코드로 실제로 구현하는 것이 목적이기 때문에 모델 훈련은 건너뛴다.

출처 및 참고문헌 :

1. https://deep-learning-study.tistory.com/537
2. https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/Classification/Inceptionv4(2016).ipynb
