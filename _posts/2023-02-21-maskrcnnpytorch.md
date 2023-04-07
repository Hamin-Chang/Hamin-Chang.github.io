---
title : '[OD/Pytorch] 파이토치로 Mask R-CNN 구현하기 🎭'
layout: single
toc: true
toc_sticky: true
categories:
  - ODCode
---

## Pytorch로 Mask R-CNN 구현하기 (FPN 구현 포함)

이번 글에서는 [<U>Matterport의 repository</U>](https://github.com/multimodallearning/pytorch-mask-rcnn)를 참고해서 pytorch로 Mask R-CNN을 직접 구현해보겠다. Mask R-CNN 모델에 대한 설명은 [<U>Mask R-CNN 논문 리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/maskrcnn/)를 참고하길 바란다.

### 1. 라이브러리 로드, 필요한 함수 및 클래스 정의

먼저 모델 구축에 필요한 라이브러리들을 로드해주자.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import utils
import visualize
nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
```

다음 코드는 전체 코드를 돌렸을 때 text massage가 뜨게하는 함수와 iteration 진행 현황을 보여주는 기능을 하는 함수를 정의하는 과정이다. Mask R-CNN 구현의 직접적인 부분은 아니기 때문에 가볍게 보고 넘어가면 좋을거 같다.


```python
# text massage를 print하는 함수, Numpy array가 생성되면 array의 shape, min, max value 출력한다.
def log(text, array=None):
   if array is not None:
     text = text.ljust(25)
     text += ("shape : {:20}, min: {:10.5f} max: {:10.5f}".format(
         str(array.shape), array.min() if array.size else "",
         array.max() if array.size else ""))
     
# terminal progress bar 생성하기 위한 함수
def printProgressbar(iteration, total, prefix='',suffix='', decimals=1, length=100, fill='█'):
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
  
  if iteration == total:
    print()
```

참고한 repository에서는 본격적으로 Mask R-CNN을 구현하기 전에 이에 필요한 함수들을 정의한다. 이 부분도 큰 비중을 두고 볼 필요는 없을 것 같다.


```python
from typing_extensions import Required
def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]

# pytorch에는 Log2 함수가 없기 때문에 만들어준다.
def log2(x):
  ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
  if x.is_cuda:
    ln2 = ln2.cuda()
  return torch.log(x) / ln2

# tensorflow의 padding='same'을 구현하는 클래스
class SamePad2d(nn.Module):
  def __init__(self, kernel_size, stride):
    super(SamePad2d, self).__init__()
    self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
    self.stride = torch.nn.modules.utils._pair(stride)

  def forward(self,input):
    in_width = input.size()[2]
    in_height = input.size()[3]
    out_width = math.ceil(float(in_width)/float(self.stride[0]))
    out_height = math.ceil(float(in_height) / float(self.stride[1]))
    pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
    pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
    pad_left = math.floor(pad_along_width / 2)
    pad_top = math.floor(pad_along_height / 2)
    pad_right = pad_along_width - pad_left
    pad_bottom = pad_along_height - pad_top
    return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
  
  def __repr__(self):
    return self.__class__.__name__
```

### 2. Backbone Network 구현

다음으로는 Mask R-CNN의 Backbone network로 사용할 ResNet-FPN을 구현하자. 먼저 FPN을 구현해보자.


```python
# FPN에서 TopDown + Lateral connection 구현 class
class TopDownLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(TopDownLayer, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    self.padding2 = SamePad2d(kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

  def forward(self, x, y):
    y = F.upsample(y, scale_factor=2)
    x = self.conv1(x)
    return self.conv2(self.padding2(x+y))

# Lateral connection 이후 3x3 conv를 적용해서 P6~P1 feature map 출력
class FPN(nn.Module):
  def __init__(self, C1, C2, C3, C4, C5, out_channels):
    super(FPN, self).__init__()
    self.out_channels = out_channels
    self.C1 = C1
    self.C2 = C2
    self.C3 = C3
    self.C4 = C4
    self.C5 = C5
    self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
    self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride =1)
    self.P5_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
                                  nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
    self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride =1)
    self.P4_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
                                  nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
    self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride =1)
    self.P3_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
                                  nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
    self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride =1)
    self.P2_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
                                  nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
  def forward(self, x):
    x = self.C1(x)
    x = self.C2(x)
    c2_out = x
    x = self.C3(x)
    c3_out = x
    x = self.C4(x)
    c4_out = x
    x = self.C5(x)
    p5_out = self.P5_conv1(x)
    p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
    p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
    p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

    p5_out = self.P5_conv2(p5_out)
    p4_out = self.P4_conv2(p4_out)
    p3_out = self.P3_conv2(p3_out)
    p2_out = self.P2_conv2(p2_out)

    # P6는 RPN에서 5번째 anchor box에 사용되며, P5를 2배 subsampling한다.
    p6_out = self.P6(p5_out)

    return [p2_out, p3_out, p4_out, p5_out, p6_out]
    
```

FPN 모델 다음에 ResNet을 구현해보자.


```python
# ResNet에 사용되는 Bottleneck 구조를 위한 class
class Bottleneck(nn.Moduule):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
    self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum= 0.01)
    self.padding2 = SamePad2d(kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
    self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum = 0.01)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
    self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum = 0.01)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.padding2(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

# Backbone network로 사용되는 ResNet Network 구현
class ResNet(nn.Module):
  def __init__(self, architecture, stage5 = False):
    super(ResNet, self).__init__()
    assert architecture in ['resnet50', 'resnet101']
    self.inplanes = 64
    self.layers = [3,4,{'resnet50':6, 'resnet101':23}[architecture],3]
    self.block = Bottleneck
    self.stage5 = stage5

    self.C1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size = 7, stride=2, padding=3),
        nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
        nn.ReLU(inplace=True),
        SamePad2d(kernel_size=3, stride=2),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.C2 = self.make_layer(self.block, 64, self.layers[0])
    self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
    self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
    if self.stage5:
      self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
    else:
      self.C5 = None

    def stages(self):
      return [self.C1, self.C2, self.C3, self.C4, self.C5]
    
    def make_layer(self, block, planes, blocks, stride=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes * block.expansion, eps = 0.001, momentum=0.01))
      
      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, stride, downsample))
        
      return nn.Sequential(*layers)

```

### 3. Faster R-CNN 구현

이제 Faster R-CNN의 학습과정을 구현할건데, 개념적인 부분은 [<U>Faster R-CNN 논문 리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/ffrcnn/#4-faster-rcnn-%ED%9B%88%EB%A0%A8%ED%95%98%EA%B8%B0)를 참고하면서 따라오면 더 이해하기 쉬울것이다. (물론 명칭 같은 것들이 완전히 똑같지는 않다.)

![2](https://user-images.githubusercontent.com/77332628/220282137-206639c1-f562-4936-9698-43dabaf30563.png)

#### 3.1 Proposal layer

anchor score를 받아서 다음 단계에 전달할 region proposal를 추출한다. Non maximums suppression을 적용해서 상위 몇개의 proposals만 전달한다.



```python
# bbox의 delta 값을 bbox에 적용하는 함수
# boxes : [N,4] <= y1, x1, y2, x2
# deltas : [N,4] <= [dy, dx, log(dh), log(dw)]
def apply_box_deltas(boxes, deltas):
  # y, x, h, w로 변환
  height = boxes[:,2] - boxes[:,0]
  width = boxes[:,3] - boxes[:,1]
  center_y = boxes[:,0] + 0.5 * height
  center_x = boxes[:,1] + 0.5 * width
  # deltas 값 적용
  center_y += deltas[:,0] * height
  center_x += deltas[:,1] * width
  height *= torch.exp(deltas[:,2])
  width *= torch.exp(deltas[:,3])
  # 적용한 값을 bbox 좌표 y1,x1,y2,x2로 다시 변환
  y1 = center_y - 0.5 * height
  x1 = center_x - 0.5 * width
  y2 = y1 + height
  x2 = x1 + width
  result = torch.stack([y1, x1, y2, x2], dim=1)
  return result

def clip_boxes(boxes, window):
  # boxes : [N,4] <= y1,x1,y2,x2
  # window : [4] <= y1,x1,y2,x2
  boxes = torch.stack([boxes[:,0].clamp(float(window[0]), float(window[2])),
                       boxes[:,1].clamp(float(window[1]), float(window[3])),
                       boxes[:,2].clamp(float(window[0]), float(window[2])),
                       boxes[:,3].clamp(float(window[1]), float(window[3]))],1)
  return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
  '''
  Inputs : 
        rpn_probs : [batch, anchors, (bg prob, fg prob)]
        rpn_bbox : [batch, anchors, (dy, dx, log(dh), log(dw))]
  
  Returns : Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
  '''
  inputs[0] = inputs[0].squeeze(0)
  inputs[1] = inputs[1].squeeze(0)

  # Box scores. foreground class confidence 사용 
  scores = inputs[0][:,1] # [batch, num_rois,1]  

  # Box deltas <= [batch, num_rois, 4]
  deltas = inputs[1]
  std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1,4])).float(), requires_grad=False)
  if config.GPU_COUNT:
    std_dev = std_dev.cuda()
  deltas = deltas * std_dev

  # anchor score에 따라 상위 몇개의 anchor만 가지고 간다.
  pre_nms_list = min(6000, anchors.size()[0])
  scores, order = scores.sort(descending = True)
  order = order[:pre_nms_list]
  scores = scores[:pre_nms_list]
  deltas = deltas[order.data, :]
  anchors = anchors[order.data, :]

  # deltas를 anchor에 적용해서 refined anchor를 얻는다.
  # [batch, N, (y1, x1, y2, x2)]
  boxes = apply_box_deltas(anchors,deltas)

  # image 경계에 맞게 anchor를 clip 한다. 
  # [batch, N, (y1, x1, y2, x2)]
  height, width = config.IMAGE_SHAPE[:2]
  window = np.array([0,0,height,width]).astype(np.float32)
  boxes = clip_boxes(boxes, window)

  # 작은 bbox들을 filter out해야하지만, Xinlei Chen의 논문에 의하면
  # filtering하는 것이 작은 객체 탐지에 악영향을 미치기 떄문에 생략한다.

  # Non-maximum suppression
  keep = nms(torch.cat((boxes, scores.unsqueeze(1)),1).data, nms_threshold)
  keep = keep[:proposal_count]
  boxes = boxes[keep, :]

  # boxes의 각 차원의 값을 0~1사이로 정규화한다.
  norm = Variable(torch.from_numpy(np.array([height,width,height,width])).float(), requires_grad=False)
  if config.GPU_COUNT:
    norm = norm.cuda()
  normalized_boxes = boxes/norm

  # 다시 batch 차원을 추가해준다.
  normalized_boxes = normalized_boxes.unsqueeze(0)

  return normalized_boxes
```

#### 3.2 RoI align layer

Mask R-CNN 모델에서는 고정된 크기의 feature map을 얻기 위해 RoI pooling 대신 RoI align을 사용한다. RoI align에 대한 자세한 설명은 [<U>Mask R-CNN 논문 리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/maskrcnn/)를 참고하길 바란다. 

Params : 
  - pool_size : 출력 pooled region의 [height, width]. 보통 [7, 7] 사이즈
  - image_shape : 입력 이미지의 [height, width, channels]

Inputs :
  - boxes : 정규화된 좌표의 [batch, num_boxes, (y1,x1,y2,x2)]
  - Feature maps : FPN에서 추출한 feature map pyramids, [batch, channels, height, width]

Outputs :
  - [num_boxes, height, width, channels]의 shape의 Pooled regions. height과 width는 pool_size로 고정

그리고 각 RoI를 적절한 level에 할당해야한다. 이는 다음의 공식을 이용해서 진행한다.

![1](https://user-images.githubusercontent.com/77332628/220282131-3a1f1158-e2a1-4db1-8b14-f8589349f0cf.png)




```python
def pyramid_roi_align(inputs, pool_size, image_shape):

  for i in range(len(inputs)):
    inputs[i] = inputs[i].unsqueeze(0)

  # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
  boxes = inputs[0]

  # 각기 다른 level의 feature map pyramids, [batch, height, width, channels]
  feature_maps = inputs[1:] 

  # RoI를 적절한 pyramid level에 할당
  y1, x1, y2, x2 = boxes.chunk(4, dim=1)
  h = y2 - y1
  w = x2 - x1

  # 위 이미지의 공식을 이용
  image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
  if boxes.is_cuda:
    image_area = image_area.cuda()
  roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
  roi_level = roi_level.round().int()
  roi_level = roi_level.clamp(2,5) # pyramid level P2~P5

  # 이제 각 pyramid level(P2~P5)에서 RoIalign 수행
  pooled = []
  box_to_level = []
  for i, level in enumerate(range(2,6)):
    ix = roi_level == level
    if not ix.any():
      continue
    ix = torch.nonzero(ix)[:,0]
    level_boxes = boxes[ix.data, :]

    # 어떤 box가 어떤 level에 mapping 되었는지 추적
    box_to_level.append(ix.data)

    # RoI에 대한 gradient propagation는 멈춘다.
    level_boxes = level.boxes.detach()

    # Crop and Resize를 수행해서 RoI align을 구현
    ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
    if level_boxes.is_cuda():
      ind = ind.cuda()
    feature_maps[i] = feature_maps[i].unsqueeze(0) # CropAndResizeFunction가 batch 차원 필요로 함
    pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
    pooled.append(pooled_features)

  # 모든 level의 pooled feature를 합친다.
  pooled = torch.cat(pooled, dim=0)

  # box_to_level도 모든 level을 하나로 합친다.
  box_to_level = torch.cat(box_to_level, dim=0)

  # pooled feature과 original box를 맞춰서 다시 재배열한다.
  _, box_to_level = torch.sort(box_to_level)
  pooled = pooled[box_to_level, :, :]
  
  return pooled
```

#### 3.3 Detection target layer 

Detection target layer는 Fast RCNN을 학습시키기 유용한 proposals을 subsample하는데, 이는 ground truth 값과의 IoU를 계산해서 기준값을 기준으로 상위 몇개의 proposals만 사용한다.




                   



```python
# 두개의 box의 IoU를 계산하는 함수
# boxes1, boxes2 : [N, (y1,x1,y2,x2)]
def bbox_overlaps(boxes1, boxes2):
  # 먼저 모든 boxes1와 boxes2를 loop 없이 비교하기 위해
  # tf.tile()과 같은 기능인 boxes1와 boxes2 각각의 좌표들을 N개 만큼 반복해서 이어붙인다.
  boxes1_repeat = boxes2.size()[0]
  boxes2_repeat = boxes1.size()[0]
  boxes1 = boxes1.repeat(boxes1_repeat, 1).view(-1,4)
  boxes2 = boxes2.repeat(boxes2_repeat, 1)

  # 이제 intersection을 계산한다.
  b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
  b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
  y1 = torch.max(b1_y1, b2_y1)[:,0]
  x1 = torch.max(b1_x1, b2_x2)[:,0]
  y2 = torch.min(b1_y2, b2_y2)[:,0]
  x2 = torch.min(b1_x2, b2_x2)[:,0]
  zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
  if y1.is_cuda:
    zeros = zeros.cuda()
  intersection = torch.max(x2-x1, zeros) * torch.max(y2-y1,zeros)

  # 두 box의 union 영역을 계산한다.
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area[:,0] + b2_area[:,0] - intersection

  # 마지막으로 IoU를 계산하고 [boxes1, boxes2]로 reshape
  iou = intersection/union
  overlaps = iou.view(boxes2_repeat, boxes1_repeat)

  return overlaps
  
```

그 다음으로 Detection target layer를 구현하는데, Detection  target layer의 입력값과 반환값은 다음과 같다.

Input :
  - proposals : 정규화된 좌표의  [batch, N, (y1, x1, y2, x2)]로 만약 proposal이 부족하면 zero padding이 수행될 수 있다.
  - gt_class_id : [batch, MAX_GT_INSTANCES] Integer class ID.
  - gt_boxes: 정규화된 좌표의 [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
  - gt_masks : boolean type의 [batch, height, width, MAX_GT_INSTANCES]

Returns : 
  - (target) rois : 정규화된 좌표의 [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
  - target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class ID.
  - target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw), class_id)] bbox 수정 delta 값.
  - target_mask : [batch, TRAIN_ROIS_PER_IMAGE, height, width] 


```python
def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
  proposals = proposals.squeeze(0)
  gt_class_ids = gt_class_ids.squeeze(0)
  gt_boxes = gt_boxes.squeeze(0)
  gt_masks = gt_masks.squeeze(0)

  # COCO crowd box는 훈련 데이터에서 제외한다. COCO crowd box는 negative class ID를 가지고 있다.
  # (이 과정을 진행하는 이유는 모르겠지만, 아마 positive class ID만 사용하려는게 아닌가 싶다.)
  if torch.nonzero(gt_class_ids<0).size():
    crowd_ix = torch.nonzero(gt_class_ids < 0)[:,0]
    non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:,0]
    crowd_boxes = gt_boxes[crowd_ix.data, :]
    crowd_masks = gt_masks[crowd_ix.data, :, :]
    gt_class_ids = gt_class_ids[non_crowd_ix.data]
    gt_boxes = gt_boxes[non_crowd_ix.data, :]
    gt_masks = gt_masks[non_crowd_ix.data, :,:]

    # propsals와 crowd boxes의 overlap 계산
    crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
    crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
    no_crowd_bool = crowd_iou_max < 0.001
  
  else: 
    no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
    if config.GPU_COUNT:
      no_crowd_bool = no_crowd_bool.cuda()

  # proposals와 gt_boxes overlap 계산하고  positive/negative RoIs 결정
  overlaps = bbox_overlaps(proposals, gt_boxes)
  roi_iou_max = torch.max(overlaps, dim=1)[0]

  # positive RoI의 기준은 iou >= 0.5
  positive_roi_bool = roi_iou_max >= 0.5

  # Positive RoIs
  if torch.nonzero(positive_roi_bool).size():
    positive_indices = torch.nonzero(positive_roi_bool)[:,0]
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO) # positive 비율이 33% 되도록
    rand_idx = torch.randperm(positive_indices.size()[0])
    rand_idx = rand_idx[:positive_count]
    if config.GPU_COUNT:
      rand_idx = rand_idx.cuda()
    positive_indices = positive_indices[rand_idx]
    positive_count = positive_indices.size()[0]
    positive_rois = proposals[positive_indices.data, :]

    # positive RoIs를 GT boxes에 매치 시킨다.
    positive_overlaps = overlaps[positive_indices.data, :]
    roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
    roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
    roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

    # positive RoIs에 대한 bbox refinement delta 값 계산
    deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
    std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), required_grad=False)
    if config.GPU_COUNT:
      std_dev = std_dev.cuda()
    deltas /= std_dev # 정규화

    # positive RoIs에 GT mask를 매치시킨다.
    roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]

    # mask target 계산
    boxes = positive_rois
    if config.USE_MINI_MASK:
      # RoI 좌표를 normalized image space에서 normalized mini-mask space로 변환
      y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
      gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
      gt_h = gt_y2 - gt_y1
      gt_w = gt_y2 - gt_y1
      y1 = (y1 - gt_y1) / gt_h
      x1 = (x1 - gt_x1) / gt_w
      y2 = (y2 - gt_y1) / gt_h
      y2 = (y2 - gt_y1) / gt_h
      boxes = torch.cat([y1, x1, y2, x2], dim=1)
    box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
    if config.GPU_COUNT:
      box_ids = box_ids.cuda()
    masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1],0)\
                     (roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
    masks = masks.squeeze(1)

    # binary cross entropy loss를 사용하기 위해 0.5 이상의 mask pixel 가지는 것들은 반올림.
    mask = torch.round(masks)
  else:
    positive_count = 0

  # Negative RoIs는 GT box와의 iou < 0.5인 proposals이다. COCO crowds는 사용하지 않는다.
  negative_roi_bool = roi_iou_max < 0.5
  negative_roi_bool = negative_roi_bool & no_crowd_bool
  # positive/negative 비율이 유지되도록 충분한 Negative RoIs를 추가해준다.
  if torch.nonzero(negative_roi_bool).size() and positive_count > 0:
    negative_indices = torch.nonzero(negative_roi_bool)[:,0]
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = int(r * positive_count - positive_count)
    rand_idx = torch.randperm(negative_indices.size()[0])
    rand_idx = rand_idx[:negative_count]
    if config.GPU_COUNT :
      radn_idx = rand_idx.cuda()
    negative_indices = negative_indices[rand_idx]
    negative_count = negative_indices.size()[0]
    negative_rois = proposals[negative_indices.data, :]
  else: 
    negative_count = 0
  
  # positive/negative RoIs들을 합치고, negative RoIs에는 사용되지 않는 bbox delat값과 mask를 0으로 padding한다.
  if positive_count > 0 and negative_count > 0 :
    rois = torch.cat((positive_rois, negative_rois), dim=0)
    zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
    roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros],dim=0)

    zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
    deltas = torch.cat([deltas, zeros], dim=0)

    zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), requires_grad=False).int()
    masks = torch.cat([masks, zeros], dim=0)
  elif positive_count > 0:
    rois = positive_rois
  elif negative_count > 0:
    rois = negative_rois
    zeros = Variable(torch.zeros(negative_count), requires_grad=False)
    roi_gt_class_ids = zeros

    zeros = Variable(torch.zeros(negative_count,4), requires_grad=False).int()
    deltas = zeros

    zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
    masks = zeros
  
  else:
    rois = Variable(torch.FloatTensor(), requires_grad=False)
    roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
    deltas = Variable(torch.FloatTensor(), requires_grad=False)
    masks = Variable(torch.FloatTensor(), requires_grad=False)
  
  return rois, roi_gt_class_ids, deltas, masks
```

#### 3.4 Detection Layer
최종 detection bbox를 출력하는 Detection layer를 구현해보자.


```python
def clip_to_window(window, boxes):
  '''
  window : (y1, x1, y2, x2), The window in the image we want to clip to
  boxes : [N ,(y1, x1, y2, x2)]
  '''
  boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
  boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
  boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
  boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))
   
  return boxes

def refine_detections(rois, probs, deltas, window, config):
  
  # Class IDs per RoI
  _, class_ids = torch.max(probs, dim=1)


  idx = torch.arange(class_ids.size()[0]).long()
  class_scores = probs[idx, class_ids.data]
  deltas_specific = deltas[idx, class_ids.data] # Class-specific bbox deltas

  # bbox delta값 적용
  # Shape : [boxes, (y1, x1, y2, x2)] in normalized coordinates
  std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1,4])).float(), requires_grad = False)
  refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

  # 좌표를 이미지의 영역으로 변환
  height, width = config.IMAGE_SHAPE[:2]
  scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
  refined_rois *= scale

  # Clip boxes to image window
  refined_rois = clip_to_window(window, refined_rois)

  # 픽셀 단위를 다루기 때문에 정수로 반올림
  refined_rois = torch.round(refined_rois)

  # 배경 bbox Filter out
  keep_bool = class_ids > 0

  # 낮은 confidence score의 bbox Filter out
  if config.DETECTION_MIN_CONFIDENCE:
    keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
  keep = torch.nonzero(keep_bool)[:,0]

  # per-class NMS 적용
  pre_nms_class_ids = class_ids[keep.data]
  pre_nms_scores = class_scores[keep.data]
  pre_nms_rois =refined_rois[keep.data]

  for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
    # pick detections of this class
    ixs = torch.nonzero(pre_nms_class_ids == class_ids)[:,0]

    # Sort
    ix_rois = pre_nms_rois[ixs.data]
    ix_scores = pre_nms_scores[ixs]
    ix_scores, order = ix_scores.sort(descending=True)
    ix_rois = ix_rois[order.data, :]

    class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

    class_keep = keep[ixs[order[class_keep].data].data]

    if i==0:
      nms_keep = class_keep
    else :
      nms_keep = unique1d(torch.cast((nms_keep, class_keep)))
  keep = intersect1d(keep, nms_keep)

  # Keep top detections
  roi_count = config.DETECTION_MAX_INSTANCES
  top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
  keep = keep[top_ids.data]
  
  # 출력값을 [N, (y1, x1, y2, x2. class_id, score)]의 형태로 배열
  # 좌표값은 이미지 영역의 값
  result = torch.cat((refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(),
                      class_scores[keep.data].unsqueeze(1)), dim=1)
  
  return result

def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
  '''
  classified proposal boxes와 bbox delta값을 받아서 최종 detection boxes를 출력한다.
  '''

  rois = rois.squeeze(0)

  _, _, window , _ = parse_image_meta(image_meta)
  window = window[0]
  detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

  return detections
```

#### 3.5 Region Proposal Network (RPN)

입력값으로

- anchors_per_location : feature map에서 각 pixel마다 anchor의 개수
- anchor_stride : anchor 수의 밀도를 조정하는 값. (보통 1로 설정해서 매 pixel마다 anchor가 있게 하거나 2로 해서 pixel하나 건너뛰어서 anchor가 있게 설정한다.)

를 받아서.

- rpn_logits : Anchor classifier logit (before softmax), [batch, H, W, 2]
- rpn_probs : Anchor classifier probabilites , [batch, H, W, 2]
- rpn_bbox : Anchor에 적용될 deltas, [batch, H, W, (dy, dx, log(dh), log(dw))]

를 반환한다. 참고로 rpn의 probabilites값이 2개인 이유는 RPN에서는 객체의 유무 확률만 반환하기 때문이다.



```python
class RPN(nn.Module):
  def __init__(self, anchors_per_location, anchor_stride, depth):
    super(RPN, self).__init__()
    self.anchors_per_location = anchors_per_location
    self.anchor_stride = anchor_stride
    self.depth = depth

    self.padding = SamePad2d(kernel_size=3, stride= self.anchor_stride)
    self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
    self.relu = nn.ReLU(inplace=True)
    self.conv_class = nn.Conv2d(512, 2*anchors_per_location, kernel_size=1, stride=1)
    self.softmax = nn.Softmax(dim=2)
    self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

  def forward(self, x):
    # shared convolutional base of the RPN
    x = self.relu(self.conv_shared(self.padding(x)))

    # Anchor score [batch, anchors_per_location * 2, height, width]
    rpn_class_logits = self.conv_class(x)

    # reshape to [batch, 2, anchors]
    rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
    rpn_class_logits = rpn_class_logits.contiguous()
    rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

    # softmax 처리
    rpn_probs = self.softmax(rpn_class_logits)

    # bbox refinement, [batch, H, W, anchors_per_location, depth]
    rpn_bbox = self.conv_bbox(x)

    # reshape to [batch, 4, anchors]
    rpn_bbox = rpn_bbox.permute(0,2,3,1)
    rpn_bbox = rpn_bbox.contiguous()
    rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

    return [rpn_class_logits, rpn_probs, rpn_bbox]
    
```

#### 3.6 Classification, BBOX regressor, Mask segment Branch

이제 최종 결과값을 도출하는 Classification과 BBR을 수행하는 class와 Mask segment를 수행하는 class를 정의한다.



```python
class Classifier(nn.Module):
  def __init__(self, depth, pool_size, image_shape, num_classes):
    super(Classifier, self).__init__()
    self.depth = depth
    self.pool_size = pool_size
    self.image_shape = image_shape
    self.num_classes = num_classes
    self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
    self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
    self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
    self.relu = nn.ReLU(inplace=True)

    self.linear_class = nn.Linear(1024, num_classes)
    self.softmax = nn.Softmax(dim=1)

    self.linear_bbox = nn.Linear(1024, num_classes * 4)

  def forward(self, x, rois):
    x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = x.view(-1, 1024)

    # classification score
    mrcnn_class_logits = self.linear_class(x)
    mrcnn_probs = self.softmax(mrcnn_class_logits)

    # bbox regressor
    mrcnn_bbox = self.linear_bbox(x)
    mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

    return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

class Mask(nn.Module):
  def __init__(self, depth, pool_size, image_shape, num_classes):
    super(Mask, self).__init__()
    self.depth = depth
    self.pool_size = pool_size
    self.image_shape = image_shape
    self.num_classes = num_classes
    self.padding = SamePad2d(kernel_size=3, stride=1)
    self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1 )
    self.bn1 = nn.BatchNorm2d(256, eps=0.001)
    self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn2 = nn.BatchNorm2d(256, eps=0.001)
    self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn3 = nn.BatchNorm2d(256, eps=0.001)
    self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn4 = nn.BatchNorm2d(256, eps=0.001)
    self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x , rois):
    x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
    x = self.conv1(self.padding(x))
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(self.padding(x))
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(self.padding(x))
    x = self.bn3(x)
    x = self.relu(x)
    x = self.conv4(self.padding(x))
    x = self.bn4(x)
    x = self.relu(x)
    x = self.deconv(x)
    x = self.relu(x)
    x = self.conv5(x)
    x = self.sigmoid(x)

    return x
```

이것으로 Mask R-CNN 모델의 전체적인 구조를 코드로 직접 구현해봤다. 참고한 repository에서는 손실함수와 데이터 셋을 다루는 코드도 다루지만, 이번 글에서는 모델의 구조를 파악하는 것이 주 목적이었기 때문에 여기까지만 다루겠다. 이번 코드를 공부하면서 참고한 repository의 파이썬, 파이토치가 예전 버전이기도 하고, 논문에서 배운 것과 워딩이 다른 것들도 있어서 완벽히 이해하지는 못했지만, 블랙박스 같았던 모델을 직접 코드로 구현해봤다는 것에서 큰 의미가 있다고 생각한다.

다음 글에서는 kaggle에서 실제로 Mask R-CNN으로 객체 탐지를 수행한 것을 다뤄보겠다.
