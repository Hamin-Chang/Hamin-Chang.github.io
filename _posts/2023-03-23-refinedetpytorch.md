---
title : '[OD/Pytorch] 파이토치로 RefineDet 구현하기 👨‍🔧' 
layout: single
toc: true
toc_sticky: true
categories:
  - ODCode
---

## Pytorch로 RefineDet 구현하기

이번 글에서는 RefineDet을 실제 pytorch 코드로 구현해볼 것이다. 코드는 [<U>luuuyi의 repository</U>](https://github.com/luuuyi/RefineDet.PyTorch/blob/master/models/refinedet.py)에서 가져왔다. RefineDet에 대한 개념적인 부분은 [<U>RefineDet 논문리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/refinedet/https://hamin-chang.github.io/cv-objectdetection/refinedet/)를 참고하길 바란다.

### 1. 라이브러리 import & PriorBox, L2Norm 정의

먼저 필요한 라이브러리들을 로드하고 모델 구축에 필요한 클래스들을 정의한다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#from __future__ import division
from math import sqrt as sqrt
from itertools import product as product

# prior box의 좌표를 center-offset 형태로 계산하는 클래스
class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios']) # feature map location에서 prior box의 수 (4 or 6)
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
              if v <= 0:
                    raise ValueError('Variances must be greater than 0')
                
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center
                cx = (j+0.5) / f_k
                cy = (i+0.5) / f_k

                # aspect_ratio : 1
                # rel size : min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio : 1
                # rel_size : sqrt(s_k * s_(k+1))
                if self.max_sizes:
                        s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                        mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

```


```python
from torch.autograd import Function
import torch.nn.init as init

# L2 regularization Module
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
  
    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
```

### 2. RefineDet 구축 모듈 정의
이제 본격적으로 RefineDet을 구축해볼건데, 먼저 모델의 phase와 size가 결정되고, base model과 extra layers, ARM, ODM, TCB가 주어졌을 떄 RefineDet을 구축하는 모듈을 정의한다.


```python
voc_refinedet = {
    '320': {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320',
    },
    '512': {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [64, 32, 16, 8],
        'min_dim': 512,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320',
    }
}

coco_refinedet = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
```


```python
class RefineDet(nn.Module):
    def __init__(self, phase, size, base, extras, ARM, ODM, TCB, num_classes):
        '''
        phase : (string) test 모드 or train 모드
        size : 입력 이미지 크기
        base : VGG16 layers for input, 300 or 500 크기
        extras : multiox loc과 conf layers에 필요한 extra layers
        head : loc and conf conv layers로 구성된 "multibox head" 
        '''
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes==21]
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
              self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.conv4_3_L2Norm = L2Norm(512,10)
        self.conv5_3_L2Norm = L2Norm(512,8)
        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])

        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])

        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)

        def forward(self, x):
            '''phase에 따라 return하는 값이 다르다
            test : tensor of output class label predictions, 
            confidence score, and corresponding location for each object detected. Shape : [batch, topk, 7]

            train : list of concat outputs from:
              1: confidence layers, Shape : [batch*num_priors, num_classes]
              2: localization layers, Shape : [batch, num_priors*4]
              3: priorbox layers, Shape : [2, num_priors*4]'''
            # x => input image or batch of images , Shape : [batch, 3, 300, 300]
            sources = list()
            tcb_source = list()
            arm_loc = list()
            arm_conf = list()
            odm_loc = list()
            odm_conf = list()

            # VGG에서 conv4_3 relu와 conv5_3 relu까지만 사용한다.
            for k in range(30):
                x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

    
            for k in range(30, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(x)

            # extra layers를 적용하고, source layer의 출력을 저장한다. 
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)

            # source layer에서 ARM을 적용한다.
            for (x,l,c) in zip(sources, self.arm_loc, self.arm_conf):
                arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
            print('sources size :', [x.size() for x in sources])
    
            # TCB feature 계산
            p = None
            for k, v in enumerate(sources[::-1]):
                s = v 
                for i in range(3):
                    s = self.tcb0[(3-k)*3 + i](s)
                if k !=0 :
                    u = p
                    u = self.tcb1[3-k](u)
                    s += u
                for i in range(3):
                    s = self.tcb2[(3-k)*3 + i](s)
                p=s
                tcb_source().append(s)
            print('tcb sources size :', (x.size() for x in tcb_source))
            tcb_source.reverse()
    
            # source layer에 ODM 적용
            for (x,l,c) in zip(tcb_source, self.odm_loc, self.odm_conf):
                odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
            print('ARM loc size :',arm_loc.size(), 'ARM conf size :', arm_conf.size(),'ODM loc size:', odm_loc.size(), 'ODM conf size :', odm_conf.size())
    
            # 훈련 모드냐 테스트 모드냐에 따라 다르게 계산한다.
            if self.phase == 'test':
                output = self.detect(
                    arm_loc.view(arm_loc.size(0), -1, 4),    # arm loc predictions
                    self.softmax(arm_conf.view(arm_conf.size(0), -1 , 2)), # arm conf precictions
                    odm_loc.view(odm_loc.size(0), -1 ,4),    # odm loc predictions
                    self.softmax(odm_conf.view(odm_conf.size(0), -1, self.num_classes)), # odm conf predictions 
                    self.priors.type(type(x.data)))  # default boxes
            else:
                output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors)
        
            return output
        
        def load_weights(self, base_file):
            other, ext = os.path.splitext(base_file)
            if ext == '.pkl' or '.pth':
                print('Loading weights into state dict..')
                self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc:storage))
                print('Finished')
            else:
                print('Sorry only .pth and .pkl files supported')
```

### 3. VGG, ARM, ODM, TCB 정의

다음 함수는 backbone model로 사용할 VGG를 구축하는 함수인데 이는  [<U>https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py</U>](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)에서 make_layers 함수를 참고했다고 한다.


```python
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels=v
    
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return layers
```

feature scaling을 위해 VGG에 추가되는 extra layers도 정의해준다.


```python
def add_extras(cfg, size, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1,3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1,3)[flag])]
            
            flag = not flag
        in_channels = v
    return layers
```

이제 RefineDet의 핵심인 ARM과 ODM layers와 TCB를 추가해보자.


```python
def arm_multibox(vgg, extra_layers, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    vgg_sources = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2],3):
        arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]*2, kernel_size=3, padding=1)]
        
    return (arm_loc_layers, arm_conf_layers)

def odm_multibox(vgg, extra_layers, cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        odm_loc_layers += [nn.Conv2d(256, cfg[k]*4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k]*num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        odm_loc_layers += [nn.Conv2d(256, cfg[k]*4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k]*num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layer = []
    feature_upsample_layer = []
    feature_pred_layer = []
    for k, v in enumerate(cfg):
        feature_scale_layer += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, 3, padding=1)]
        feature_pred_layer += [nn.ReLU(inplace=True),
                              nn.Conv2d(256, 256, 3, padding=1),
                              nn.ReLU(inplace=True)]
        if k != len(cfg) -1 :
            feature_upsample_layer += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layer, feature_upsample_layer, feature_pred_layer)
```


```python
# 모델 구축에 사용되는 configs
base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],}  # number of boxes per feature map location


tcb = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],}

```

### 4. 최종 RefineDet 구축

이제 위에서 정의한 모듈과 클래스와 함수들을 사용해서 RefineDet을 정의해준다. 이때, phase가 test인지 train인지에 따라 계산이 다르고, 모델의 크기가 320, 512인가에 따라 config에서 사용하는 값이 다르다.


```python
def build_refinedet(phase, size=320, num_classes=21):
    if phase != 'test' and phase != 'train':
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return
    
    base_ = vgg(base[str(size)], 3)
    extras_ = add_extras(extras[str(size), size, 1024])
    ARM_ = arm_multibox(base_, extras_, mbox[str(size)])
    ODM_ = odm_multibox(base_, extras_, mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(phase, size, base_, extras_, ARM_, ODM_, TCB_, num_classes)
```

Inference 부분 등 여러 부분을 생략했지만, 직접 코드로 RefineDet을 구축해보니 더욱 깊이 이해할 수 있는 기회였다.

출처 : [<U>luuuyi의 repository</U>](https://github.com/luuuyi/RefineDet.PyTorch/blob/master/models/refinedet.py)

https://github.com/luuuyi/RefineDet.PyTorch/blob/master/models/refinedet.py
