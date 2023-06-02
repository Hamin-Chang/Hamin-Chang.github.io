---
title : '[IC/Pytorch] 파이토치로 PyramidNet 구현하기 🐫' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 PyramidNet 구현하기

이번 글에서는 실제 pytorch 코드로 pyramidNet을 구현해본다. PyramidNet에 대한 개념은 [<U>**PyramidNet 논문 리뷰**</U>](https://hamin-chang.github.io/cv-imageclassification/pyramidnet/)를 참고하길 바란다. 코드는 [<U>**dyhan0920's repository**</U>](https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py)를 참고했다.

### 1. Residual Blocks 구현

먼저 논문에서 사용한 BasicBlock과 Bottleneck Block을 구현한다.

논문에서처럼 다음 이미지의 (d)파트의 block을 사용한다. BasicBlock은 (d)의 왼쪽, Bottleneck Block은 오른쪽이다.

![1](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/d12b5121-1084-4e4f-8963-99a8831c6ccf)


```python
import torch
import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo

# 3x3 convolution with padding 정의
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    outchannel_ratio = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else : 
            shortcut = x
            featuremap_size = out.size()[2:4]
        
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        
        # Zero-padded Identitiy-mapping Shortcut
        if residual_channel != shortcut_channel : 
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else : 
            out += shortcut
        
        return out
```


```python
class Bottleneck(nn.Module):
    outchannel_ratio = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck,self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes*Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes*Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out = self.bn4(out)
        
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
            
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
            
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel-shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding),1)
        
        else :
            out += shortcut
        
        return out
```

### 2. PyramidNet 구축

위에서 구현한 residual block들을 이용해서 이제 PyramidNet 모델을 구축할건데, 논문에서처럼 사용하는 dataset이 CIFAR일때와 ImageNet일때를 나눠서 모델을 구축한다.

논문에서는 dataset이 CIFAR 계열일때만 나와있다.

![2](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/bc2c1447-5d37-4aea-b20f-b8fd383b28d1)



```python
class PyramidNet(nn.Module):
    
    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2,2), stride = (2,2), ceil_mode=True)
    
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim))*block.outchannel_ratio, int(round(temp_featuremap_dim)),1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        
        return nn.Sequential(*layers)
    
    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth-2)/9)
                block = Bottleneck
            else:
                n = int((depth-2)/6)
                block = Bottleneck
            
            self.addrate = alpha / (3*n*1.0)
            
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)
            
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        
        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            
            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else :
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)
                
                layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])
                
            self.inplanes = 64
            self.addrate = alpha / (sum(layers[depth]) * 1.0)
            
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)
            
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
    def forward(self,x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.bn_final(x)
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        
        return x
```

출처 및 참고문헌

1. https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py
2. https://arxiv.org/pdf/1610.02915.pdf
