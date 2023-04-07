---
title : '[OD/Pytorch] 파이토치로 M2Det 구현하기 🍰' 
layout: single
toc: true
toc_sticky: true
categories:
  - ODCode
---

## Pytorch로 M2Det 구현하기

![0](https://user-images.githubusercontent.com/77332628/229323072-03a28773-38d7-4d02-9ebb-ced6e5dbb909.png)

이번 글에서는 M2Det를 실제 pytorch 코드로 구현 해본다. 코드는 [<U>qijiezhao의 repository</U>](https://github.com/VDIGPKU/M2Det)에서 가져왔다. M2Det에 대한 자세한 동작 방법에 대한 설명은 [<U>M2Det 논문리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/M2DET/)을 참고하길 바란다. 

### 1. 사전 모듈 및 함수 정의 
먼저 M2Det 모델을 구축하는 클래스에 사용되는 모듈들을 정의한다.먼저 M2Det 모델을 구축하는 클래스에 사용되는 모듈들을 정의한다.

먼저 BasicConv 모듈은 이어서 나오는 모듈들에서 사용되는 conv 연산을 편하게 하기 위해서 미리 정의하는 모듈이다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                             padding= padding, dilation=dilation, groups= groups, bias = bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLUL(inplace = True) if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
```

#### 1.1 TUM
TUM 모듈은 이름 그대로 TUM의 역할을 하는 모듈이다.


![1](https://user-images.githubusercontent.com/77332628/229323075-8ea2d5ec-a19c-4cbe-8278-e9df017e535d.png)

여기서 특이한 점은 upsample을 interpolation 함수를 이용해서 수행한다는 점이다.


```python
class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2*self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes
        
        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)),
                              BasicConv(self.in1, self.planes, 3, 2, 1))
        
        for i in range(self.scales-2):
            if not i == self.scales -3 :
                self.layers.add_module('{}'.format(len(self.layers)),
                                      BasicConv(self.planes, self.planes, 3, 2, 1))
            
            else:
                self.layers.add_module('{}'.format(len(self.layers)),
                                      BasicConv(self.planes, self.planes, 3, 1, 0))
                
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))
        self.latlayer = nn.Sequential()
        
        for i in range(self.scales-2):
            self.latlayer.add_module('{}'.format(len(self.latlayer)),
                                    BasicConv(self.planes, self.planes, 3, 1, 1))
        
        self.latlayer.add_module('{}'.format(len(self.latlayer)), BasicConv(self.in1, self.planes, 3, 1, 1))
    
        if self.is_smooth:
            smooth = list()
            for i in range(self.scales-1):
                smooth.append(BasicConv(self.planes, self.planes, 1, 1, 0))
            
            self.smooth = nn.Sequential(*smooth)
    
    def _upsample_add(self, x, y, fuse_type = 'interp'):
        _,_,H,W = y.size()
        if fuse_type == 'interp':
            return F.interpolate(x, size=(H,W), mode='nearest') + y  # interpolation으로 upsample 수행
        
        else : 
            raise NotImplementedError
    
    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x,y],1)
        conved_feat = [x]
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)
            
        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(self._upsample_add(
                                deconved_feat[i],self.latlayer[i](conved_feat[len(self.layers)-1-i])))
        
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(self.smooth[i](deconved_feat[i+1]))
            return smoothed_feat
        return deconved_feat
            
```

이제 backbone 모델을 정의하고 get_backbone 함수로 backbone 함수를 불러오는데, 이번 글에서는 backbone network로 vgg16만 사용하는 것으로 생각할 것이기 때문에 resnet과 senet 부분은 주석처리 했다.


```python
def vgg(cfg, i ,batch_norm = False):
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
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def get_backbone(backbone_name='vgg16'):
    if backbone_name == 'vgg16':
        base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        return vgg(base, 3, batch_norm=False)
    '''elif backbone_name in senet.__all__:
        return getattr(senet,backbone_name)(num_classes=1000, pretrained='imagenet')
    elif backbone_name in resnet.__all__:
        return getattr(resnet,backbone_name)(pretrained=True)'''
```

### 1.2 SFAM
SFAM 모듈은 이름 그대로 SFAM 역할을 하는 모듈이다.

![2](https://user-images.githubusercontent.com/77332628/229323078-6a15ff6e-7cfe-4aa7-b222-92afba6e94bd.png)

위 이미지에서의 연산을 수행하는데, 이때 global average pooling은 AdaptiveAvgPool2d를 사용해서 수행한다.


```python
class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio
        
        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels, 
                                           self.planes*self.num_levels//16,
                                           1,1,0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // 16,
                                           self.planes*self.num_levels,
                                           1,1,0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf * _tmp_f)
        return attention_feat        
```

마지막으로 check_argu라는 함수를 정의할건데, 이는 m2det module을 구축할 때 arguments가 available한지 확인하는 용도이다.


```python
def check_argu(key, value):
    if key == 'backbone':
        assert value in ['vgg16','resnet18','resnet34','resnet50','resnet101','resnet152'
          'se_resnet50','se_resnet101', 'senet154', 'se_resnet152', 
          'se_resnext50_32x4d', 'se_resnext101_32x4d'], 'Not implemented yet!'
        
    elif key == 'net_family':
        assert value in ['vgg', 'res'], 'Only support vgg and res family Now'
    elif key == 'base_out':
        assert len(value) == 2, 'We have to ensure that the base feature is formed with 2 backbone features'
    elif key == 'planes':
        pass # No rule for plane now.
    elif key == 'num_levels':
        assert value>1, 'At last, you should leave 2 levels'
    elif key == 'num_scales':
        pass # num_scales should equals to len(step_pattern), len(size_pattern)-1,
    elif key == 'sfam':
        pass
    elif key == 'smooth':
        pass
    elif key == 'num_classes':
        pass
    return True

from termcolor import cprint

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info , str):
            cprint(info, _type[0], attrs = [_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs = [_type[1]])
    else:
        print(info)
```

### 2, M2Det 구축하기

이제 위에서 정의한 모듈들과 함수들을 활용해서 M2Det 모듈을 구성한다.

FFM 모듈은 TUM과 SFAM과는 다르게 사전에 모듈을 정의하지 않고 바로 M2Det 클래스 중간에 구현한다.


![3](https://user-images.githubusercontent.com/77332628/229323079-92d5f3c5-5f47-4e9a-b4d8-035f954b84b6.png)

(FFM의 동작 이미지)


```python
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os, sys, time
from torch.nn import init as init

class M2Det(nn.Module):
    def __init__(self, phase, size, config=None):
        super(M2Det, self).__init__()
        self.phase = phase
        self.size = size
        self.init_params(config)
        print_info('===> Constructing M2Det model', ['yellow','bold'])
        self.construct_modules()
    
    def init_params(self, config=None): # config 읽어오기
        assert config is not None, 'Error : no config'
        for key,value in config.items():
            if check_argu(key, value):
                setattr(self, key, value)
    
    def construct_modules(self,):
        # TUMs 구축하기
        for i in range(self.num_levels):
            if i == 0:
                setattr(self, 'unet{}'.format(i+1),
                       TUM(first_level=True,
                          input_planes = self.planes//2,
                          is_smooth = self.smooth,
                          scales = self.num_scales,
                          side_channel=512))
        
            else:
                setattr(self, 'unet{}'.format(i+1),
                       TUM(first_level=False,
                          input_planes = self.planes//2,
                          is_smooth = self.smooth,
                          scales = self.num_scales,
                          side_channel=self.planes))
            
        # base features 구성하기 (논문에선 FFM)
        if 'vgg' in self.net_family: # Backbone model이 VGG일때
            self.base = nn.ModuleList(get_backbone(self.backbone))
            shallow_in, shallow_out = 512, 256
            deep_in, deep_out = 1024, 512
        elif 'res' in self.net_family: # Backbone model이 ResNet이거나 ResNeXt일때
            self.base = get_backbone(self.backbone)
            shallow_in, shallow_out = 512, 256
            deep_in, deep_out = 2048, 512
        self.reduce = BasicConv(shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce = BasicConv(deep_in, deep_out, kernel_size=1, stride=1)
        
        # construct others
        if self.phase =='test':
            self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256*8)
        self.leach = nn.ModuleList([BasicConv(
                                    deep_out+shallow_out, self.planes//2,
                                    kernel_size=(1,1), stride=(1,1))]*self.num_levels)
        #SFAM module 구현
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)
    
    def forward(self, x):
        loc, conf = list(), list()
        base_feats = list()
        if 'vgg' in self.net_family:
            for k in range(len(self.base)):
                x = self.base[k](x)
                if k in self.base_out:
                    base_feats.append(x)
        elif 'res' in self.net_family:
            base_feats = self.base(x, self.base_out)
        base_feature = torch.cat((self.reduce(base_feats[0],
                                             F.interpolate(self.up_reduce(base_feats[1])),
                                             scale_factor=2, mode='nearest')),1)
        
        # tum_outs => multi-level, mutli-scale feature maps
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature), 'none')]
        for i in range(1, self.num_levels,1):
            tum_outs.append(getattr(self,'unet{}'.format(i+1))(
                                    self.leach[i](base_feature),tum_outs[i-1][-1]))
        
        # 같은 scale끼리 concat
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs], 1) for i in range(self.num_scales,0,-1)]
        
        # SFAM forward하기
        if self.sfam:
            sources = self.sfam_module(sources)
        sources[0] = self.Norm(sources[0])
        
        for (x,l,c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.phase=='test':
            output = (loc.view(loc.size(0), -1, 4),                 # loc preds
                     self.softmax(conf.view(-1, self.num_classes))) # conf preds
        else:
            output = (loc.view(loc.size(0),-1, 4),
                     conf.view(conf.size(0), -1, self.num_classes))
        return output
    
    # 모델 초기화해주기
    def inti_model(self, base_model_path):
        if self.backbone == 'vgg16':
            if isinstance(base_model_path, str):
                base_weights = torch.load(base_model_path)
                print_info('Loading base network...')
                self.base.load_state_dict(base_weights)
        elif 'res' in self.backbone:
            pass # pre-trained seresnet은 모델을 정의할 때 초기화 완료되어 있음
        
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        
        print_info('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        
        for i in range(self.num_levels):
            getattr(self, 'unet{}'.format(i+1)).apply(weights_init)
        self.reduce.apply(weights_init)
        self.up_reduce.apply(weights_init)
        self.leach.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)
    
    # pre-trained 가중치 로드하기
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print_info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print_info('Finished!')
        else:
            print_info('Sorry only .pth and .pkl files supported.')
        
def build_net(phase='train', size=320, config=None):
    if not phase in ['test', 'train']:
        raise ValueError("Error: Phase not recognized")
    
    if not size in [320, 512, 704, 800]:
        raise NotImplementedError("Error: Sorry only M2Det320,M2Det512 M2Det704 or M2Det800 are supported!")
    
    return M2Det(phase, size, config)                
            
                
        
```

출처 및 참고 사이트: https://github.com/VDIGPKU/M2Det


