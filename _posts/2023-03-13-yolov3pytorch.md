---
title : '[OD/Pytorch] 파이토치로 YOLO v3 구현하기 🤟' 
layout: single
toc: true
toc_sticky: true
categories:
  - ODCode
---

## Pytorch로 YOLO v3 구현하기

이번 글에서는 이전 글에서 다룬 YOLO v3 모델을 파이토치로 직접 구현해본다. [<U>개인 블로그</U>](https://deep-learning-study.tistory.com/568)를 참고해서 모델을 구현했는데, 참고한 블로그에서는 DataLoader부분부터 모델을 직접 훈련하는 단계까지 모두 구현했지만, 이번 글에서는 YOLO v3의 동작 방식을 이해하는 것이 목적이기 때문에, YOLO layer과 Darknet과 손실함수만 구현하도록 하겠다. YOLO v3에 대한 개념적인 내용은 [<U>YOLO v3 논문 리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/yolov3/)를 참고하면 된다.

![1](https://user-images.githubusercontent.com/77332628/224607382-a6b753b9-98e6-4d89-8ad0-b5ce46ced3cf.png)

YOLO v3는 간단히 말해서 DarkNet으로 feature을 추출하고, FPN을 거쳐서 예측을 하는 구조다. 

### 0. 구성 요소 layer 정의


```python
# 우선 yolo layer과 DarkNet에 사용될 BasicConv와 ResidualBlock 정의
import torch
from torch import nn
import numpy as np

class BasicConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1))
    
  def forward(self, x):
    return self.conv(x)

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    
    self.residual = nn.Sequential(
        BasicConv(channels, channels//2, 1, stride=1, padding=0),
        BasicConv(channels//2, channels, 3, stride=1, padding=1))
    
    self.shortcut = nn.Sequential()
  def forward(self,x):
    x_shortcut = self.shortcut(x)
    x_residual = self.residual(x)

    return x_shortcut + x_residual
```


```python
# FPN에서 사용하는 Top_down layer 정의
class Top_down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        BasicConv(in_channels, out_channels, 1, stride=1, padding=0),
        BasicConv(out_channels, out_channels*2, 3, stride=1, padding=1),
        BasicConv(out_channels*2, out_channels, 1, stride=1, padding=0),
        BasicConv(out_channels, out_channels*2, 3, stride=1, padding=1),
        BasicConv(out_channels*2, out_channels, 1, stride=1, padding=0))
    
  def forward(self, x):
    return self.conv(x)
```

### 1. YOLO Layer 구현


```python
# YOLO layer는 13x13, 26x26, 52x52 feature map에서 예측을 수행
class YOLOlayer(nn.Module):
  def __init__(self, channels, anchors, num_classes=20, img_dim=416):
    super().__init__()
    self.anchors = anchors # 3 anchors per YOLO layer
    self.num_anchors = len(anchors) # =3
    self.num_classes = num_classes 
    self.img_dim = img_dim
    self.grid_size = 0

    # 예측을 수행하기 전에 사용하는 smooth conv layer
    self.conv = nn.Sequential(
        BasicConv(channels, channels*2, 3, stride=1, padding=1),
        nn.Conv2d(channels*2, 75, 1, stride=1, padding=0))
  
  def forward(self,x):
    x = self.conv(x)

    # prediction 
    # x : [batch, channels, W, H]
    batch_size = x.size(0)
    grid_size = x.size(2) # S = 13 or 26 or 52
    device = x.device

    prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
    # shape = (batch, 3, 25, S, S)

    # shape change (batch, 3, 25, S, S) -> (batch, 3, S, S, 25)
    prediction = prediction.permute(0, 1, 3, 4, 2)
    prediction = prediction.contiguous()

    obj_score = torch.sigmoid(prediction[..., 4]) # Confidence : 1 if object , else 0
    pred_cls = torch.sigmoid(prediction[..., 5:]) # 예측 class confidence

    # grid_size 갱신
    if grid_size != self.grid_size:
      # grid_size를 갱신하고 transform_outputs 함수를 위해 anchor box 전처리
      self.compute_grid_offsets(grid_size, cuda= x.is_cuda)

    # bbox coordinates 계산
    pred_boxes = self.transform_outputs(prediction)

    '''output shape(batch, num_anchors x S x S, 25)
    ex) 13x13 => [batch, 507, 25], 26x26 => [batch, 2028, 25], 52x52 => [batch, 8112, 25]
    최종적으로 모델은 10647(=507+2028+8112)개의 bbox를 예측한다.'''
    output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                        obj_score.view(batch_size, -1, 1),
                        pred_cls.view(batch_size, -1, self.num_classes)),-1)

    return output

  # grid_size를 갱신하고 transform_outputs 함수를 위해 anchor box 전처리 함수 정의
  def compute_grid_offsets(self, grid_size, cuda=True):
    self.grid_size = grid_size # 13, 26, 52
    self.stride = self.img_dim / self.grid_size

    # transform_outputs 함수에서 bbox의 x,y 좌표를 예측할 때 사용하는 cell index 생성
    # (1, 1, S, S)
    self.grid_x = torch.arange(grid_size).repeat(1, 1, grid_size, 1).type(torch.float32)
    self.grid_y = torch.arange(grid_size).repeat(1, 1, grid_size, 1).transpose(3,2).type(torch.float32)
     
    # anchors를 feature map크기로 정규화, [0~1] 범위
    scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
    # tensor로 변환
    self.scaled_anchors = torch.tensor(scaled_anchors)

    # transform outputs 함수에서 bbox의 w,h 예측할 때 사용
    # shape (3,2) -> (1, 3, 1, 1) 변환
    self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
    self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

  # 예측한 bbox의 좌표 계산 함수 정의
  def transform_outputs(self, prediction):
    # prediction = (batch, num_anchors, S, S, coordinates + classes)
    device = prediction.device
    x = torch.sigmoid(prediction[..., 0]) # sigmoid(box x) => sigmoid 사용해서 [0~1] 범위
    y = torch.sigmoid(prediction[..., 1]) # sigmoid(box y) => sigmoid 사용해서 [0~1] 범위
    w = prediction[..., 2] # 예측한 bbox의 너비
    h = prediction[..., 3] # 예측한 bbox의 높이

    pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
    pred_boxes[..., 0] = x.data + self.grid_x # sigmoid(box x) + x 좌표 offset
    pred_boxes[..., 1] = y.data + self.grid_y # sigmoid(box y) + y 좌표 offset
    pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
    
    return pred_boxes * self.stride
```

### 2. DarkNet-53 구현

YOLO v3에서 사용되는 DarkNet의 구조는 다음과 같다.

![2](https://user-images.githubusercontent.com/77332628/224607389-8cb29513-c59c-48e3-9d67-1f29e64c57ed.png)




```python
class DarkNet(nn.Module):
  def __init__(self, anchors, num_blocks=[1,2,8,8,4], num_classes=20):
    super().__init__()

    # feature extractor
    self.conv1 = BasicConv(3, 32, 3, stride=1, padding=1)
    self.res_block_1 = self._make_residual_block(64, num_blocks[0]) # 208x208
    self.res_block_2 = self._make_residual_block(128, num_blocks[1]) # 104x104
    self.res_block_3 = self._make_residual_block(256, num_blocks[2]) # 52x52, FPN lateral connection
    self.res_block_4 = self._make_residual_block(512, num_blocks[3]) # 26x26, FPN lateral connection
    self.res_block_5 = self._make_residual_block(1024, num_blocks[4]) # 13x13, Top layer

    # FPN Top down, conv + upsampling 수행
    self.topdown_1 = Top_down(1024, 512)
    self.topdown_2 = Top_down(768, 256)
    self.topdown_3 = Top_down(384, 128)

    # FPN lateral connection, 차원 축소 위해 사용
    self.lateral_1 = BasicConv(512, 256, 1, stride=1, padding=0)
    self.lateral_2 = BasicConv(256, 128, 1, stride=1, padding=0)

    # prediction, 13x13, 26x26, 52x52 feature map에서 예측 수행
    self.yolo_1 = YOLOlayer(512, anchors=anchors[2]) # 13x13에서 예측
    self.yolo_2 = YOLOlayer(256, anchors=anchors[1]) # 26x26에서 예측
    self.yolo_3 = YOLOlayer(128, anchors=anchors[0]) # 52x52에서 예측

    # upsample
    self.upsample = nn.Upsample(scale_factor=2)

  def forward(self, x):
    # feature extractor
    x = self.conv1(x)
    c1 = self.res_block_1(x)
    c2 = self.res_block_2(c1)
    c3 = self.res_block_3(c2)
    c4 = self.res_block_4(c3)
    c5 = self.res_block_5(c4)

    # FPN Top down, upsampling + lateral connection 수행
    p5 = self.topdown_1(c5)
    p4 = self.topdown_2(torch.cat((self.upsample(p5), self.lateral_1(c4)),1))
    p3 = self.topdown_3(torch.cat((self.upsample(p4), self.lateral_2(c3)),1))

    # prediction 수행
    yolo_1 = self.yolo_1(p5)
    yolo_2 = self.yolo_2(p4)
    yolo_3 = self.yolo_3(p3)

    return torch.cat((yolo_1, yolo_2, yolo_3), 1), [yolo_1, yolo_2, yolo_3]

  def _make_residual_block(self, in_channels, num_block):
    blocks = []

    # down-sample
    blocks.append(BasicConv(in_channels//2, in_channels, 3, stride=2, padding=1))

    for i in range(num_block):
      blocks.append(ResidualBlock(in_channels))
    
    return nn.Sequential(*blocks)

```

구축한 모델을 확인해보자.


```python
anchors = [[(10,13),(16,30),(33,23)],[(30,61),(62,45),(59,119)],[(116,90),(156,198),(373,326)]]
x = torch.randn(1,3,416,416)
with torch.no_grad():
  model = DarkNet(anchors)
  output_cat, output = model(x)
  print(output_cat.size())
  print(output[0].size(), output[1].size(), output[2].size())
```

    torch.Size([1, 10647, 25])
    torch.Size([1, 507, 25]) torch.Size([1, 2028, 25]) torch.Size([1, 8112, 25])


### 4. Loss function 구현

참고한 블로그에서는 손실함수를 구현하는 것이 너무 복잡하기 때문에 다음 깃허브를 참고했다고 한다.

https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter05/Chapter%205.ipynb 


```python
def get_loss_batch(output, targets, params_loss, opt=None):
  ignore_thres = params_loss['ignore_thres']
  scaled_anchors = params_loss['scaled_anchors'] # 정규화된 anchors
  mse_loss = params_loss['mse_loss'] # nn.MSELoss
  bce_loss = params_loss['bce_loss'] # nn.BCELoss

  num_yolos = params_loss['num_yolos'] # =3
  num_anchors = params_loss['num_anchors'] # =3
  obj_scale = params_loss['obj_scale'] # =1
  noobj_scale = params_loss['noobj_scale'] # =100

  loss = 0.0

  for yolo_ind in range(num_yolos):
    yolo_out = output[yolo_ind] # (batch, num_boxes, class+coordinates)
    batch_size, num_bbxs, _ = yolo_out.shape

    # get grid size
    gz_2 = num_bbxs / num_anchors # 예) at 13x13 => 507/3 = 169
    grid_size = int(np.sqrt(gz_2)) 

    # (batch, num_boxes, class+coordinates) -> (batch, num_anchors, S, S, class+coordinates) 변환
    yolo_out = yolo_out.view(batch_size, num_anchors, grid_size, grid_size, -1)

    pred_boxes = yolo_out[:, :, :, :, :4] # get bbox coordinates
    x, y, w, h = transform_bbox(pred_boxes, scaled_anchors[yolo_ind]) # cell 내에서 x,y,w,h 값
    pred_conf = yolo_out[:,:,:,:,4] # get confidence
    pred_cls_prob = yolo_out[:,:,:,:,5:] # class 확률

    # 타깃값 정의
    yolo_targets = get_yolo_targets({
        'pred_cls_prob':pred_cls_prob,
        'pred_boxes':pred_boxes,
        'targets':targets,
        'anchors':scaled_anchors[yolo_ind],
        'ignore_thres':ignore_thres
    })

    obj_mask=yolo_targets["obj_mask"]        
    noobj_mask=yolo_targets["noobj_mask"]            
    tx=yolo_targets["tx"]                
    ty=yolo_targets["ty"]                    
    tw=yolo_targets["tw"]                        
    th=yolo_targets["th"]                            
    tcls=yolo_targets["tcls"]                                
    t_conf=yolo_targets["t_conf"]

    loss_x = mse_loss(x[obj_mask], tx[obj_mask])
    loss_y = mse_loss(y[obj_mask], ty[obj_mask])
    loss_w = mse_loss(w[obj_mask], tw[obj_mask])
    loss_h = mse_loss(h[obj_mask], th[obj_mask])

    loss_conf_obj = bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
    loss_conf_noobj = bce_loss(pred_conf[noobj_mask], t_conf[noobj_mask])
    loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
    loss_cls = bce_loss(pred_cls_prob[obj_mask], tcls[obj_mask])
    loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

  if opt is not None:
    opt.zero_grad()
    loss.backward()
    opt.step()
  
  return loss.item()
```

transform_bbox 함수는 전체 이미지의 x,y 좌표에서 cell 내의 x,y 좌표로 변환하고 w,h를 anchor box에 맞게 변환하는 함수다.



```python
def transform_bbox(bbox,anchors):
  # bbox : predicted bbox coordinates
  # anchors : scaled anchors

  x = bbox[:,:,:,:,0]
  y = bbox[:,:,:,:,1]
  w = bbox[:,:,:,:,2]
  h = bbox[:,:,:,:,3]
  anchor_w = anchors[:,0].view((1,3,1,1))
  anchor_h = anchors[:,1].view((1,3,1,1))

  x = x-x.floor() # 전체 이미지의 x 좌표 -> cell내의 x 좌표로 변환
  y = y-y.floor() # 전체 이미지의 y 좌표 -> cell내의 y 좌표로 변환
  w = torch.log(w / anchor_w + 1e-16)
  h = torch.log(h / anchor_h + 1e-16)
  return x, y, w, h
```

get_target_yolo는 ground truth와 IoU가 가장 높은 anchor를 object가 있다고 할당(responsible for)하고, IoU가 threshold보다 큰 anchor와 나머지 anchor는 무시한다. 또한 bbox 예측좌표, cls, confidence를 생성한다.


```python
# anchor와 target box의 IoU 계산
def get_iou_WH(wh1, wh2):
  wh2 = wh2.t() # .t() : 전치
  w1, h1 = wh1[0], wh1[1]
  w2, h2 = wh2[0], wh2[1]
  inter_area = torch.min(w1,w2) * torch.min(h1,h2)
  union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
  return inter_area / union_area

```


```python
def get_yolo_targets(params):
  pred_boxes = params['pred_boxes']
  pred_cls_prob = params['pred_cls_prob']
  target = params['targets'] # (batchsize, cls, cx, cy, w, h)
  anchors = params['anchors']
  ignore_thres = params['ignore_thres']

  batch_size = pred_boxes.size(0)
  num_anchors = pred_boxes.size(1)
  grid_size = pred_boxes.size(2)
  num_cls = pred_cls_prob.size(-1)

  sizeT = batch_size, num_anchors, grid_size, grid_size
  obj_mask = torch.zeros(sizeT, dtype=torch.uint8)
  noobj_mask = torch.ones(sizeT, dtype=torch.uint8)
  tx = torch.zeros(sizeT, dtype=torch.float32)
  ty = torch.zeros(sizeT, dtype=torch.float32)
  tw = torch.zeros(sizeT, dtype=torch.float32)
  th = torch.zeros(sizeT, dtype=torch.float32)

  sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
  tcls = torch.zeros(sizeT, dtype=torch.float32)

  # target = (batch, cx, cy, w, h, class)
  target_bboxes = target[:, 1:5] * grid_size
  t_xy = target_bboxes[:, :2]
  t_wh = target_bboxes[:, 2:]
  t_x, t_y = t_xy.t() # .t() : 전치
  t_w, t_h = t_wh.t() 

  grid_i, grid_j = t_xy.long().t() # .long() : int변환

  # anchor와 target의 IoU 계산
  iou_with_anchors = [get_iou_WH(anchor, t_wh) for anchor in anchors] 
  iou_with_anchors = torch.stack(iou_with_anchors)
  best_iou_wa, best_anchor_ind = iou_with_anchors.max(0) # iou가 가장 높은 anchor 추출

  batch_inds, target_labels = target[:,0].long(), target[:,5].long()
  obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1 # iou가 가장 높은 anchor 할당
  noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

  # IoU > threshold인 anchor
  # IoU가 가장 높은 anchor만 할당하면 된다.
  for ind, iou_wa in enumerate(iou_with_anchors.t()):
    noobj_mask[batch_inds[ind], iou_wa > ignore_thres, grid_j[ind], grid_i[ind]] = 0

  # cell 내에서 x,y 변환
  tx[batch_inds, best_anchor_ind, gird_j, grid_i] = t_x - t_x.float()
  ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - t_y.float()

  anchor_w = anchors[best_anchor_ind][:, 0]
  tw[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)

  anchor_h = anchors[best_anchor_ind][:, 1]
  th[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

  tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

  output = {
      'obj_mask' : obj_mask,
      'noobj_mask' : noobj_mask,
      'tx' : tx,
      'ty' : ty,
      'tw' : tw,
      'th' : th,
      'tcls' : tcls,
      't_conf' : obj_mask.float()
  }

  return output
```

이렇게 YOLO v3모델을 직접 코드로 구현해봤다. YOLO layer과 DarkNet까지는 이해가 잘 됐지만 아무래도 손실함수는 좀 복잡한 경향이 있는 것 같다. 특히, IoU가 가장 높은 anchor만 할당하는 과정을 구현한 get_yolo_targets 함수에서 이해가 쉽지 않았다.

출처 및 참고자료

개인 블로그 (https://deep-learning-study.tistory.com/568)
