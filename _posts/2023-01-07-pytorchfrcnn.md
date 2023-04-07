---
title : '[OD/Pytorch] 파이토치로 Fast R-CNN 구현하기 📦'
layout: single
toc: true
toc_sticky: true
categories:
  - ODCode
---

## Pytorch로 Fast R-CNN 구현하기

이번 글에서는 [gary1346aa님의 github repository](https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch)에 올라온 파이토치로 구현한 Fast RCNN 코드를 다뤄보겠다. Fast RCNN의 핵심 아이디어를 다룬 부분들에 대해서 분석해본다. Fast RCNN에 대한 개념은 이전 글 ([**링크**](https://hamin-chang.github.io/cv-objectdetection/frcnn/))을 참고하면 된다.

### 1. RoI Pooling

![pyfrcnn1](https://user-images.githubusercontent.com/77332628/211156319-461968ae-e8ba-411c-bbde-ae9da7eaaceb.png)

먼저 Fast RCNN의 핵심 아이디어인 RoI pooling에 대해 살펴보자. 이전 글 ([**링크**](https://hamin-chang.github.io/cv-objectdetection/frcnn/))에서 언급했듯이, RoI pooling은 원본 이미지를 주입하는 사전 훈련된 VGG16 모델의 마지막 max pooling layer를 대체해서 고정된 크기의 feature map을 다음 fc layer에 전달한다. RoI pooling을 수행하는 feature map의 크기는 14x14x512이다. 이 점을 생각하면서 코드를 살펴보자.


```python
import torch.nn as nn
import numpy as np

class RoIPool(nn.Module):
  def __init__(self, output_size):
    super().__init__()
    self.maxpool = nn.AdaptiveMaxPool2d(output_size)
    self.size = output_size
  
  def forward(self, images, rois, roi_idx):
    n = rois.shape[0]  # RoI의 개수

    # 고정된 크기의 입력 데이터가 들어오기 때문에 전부 14x14
    h = images.size(2)  # feature map 높이
    w = images.size(3)  # feature map 너비

    # RoI의 좌표 (x1,y1,x2,y2) 행렬 (상대 좌표로 들어옴)
    x1 = rois[:,0]
    y1 = rois[:,1]
    x2 = rois[:,2]
    y2 = rois[:,3]

    # RoI의 상대좌표를 feature map에 맞게 절대좌표로 변환
    x1 = np.floor(x1 * w).astype(int)
    x2 = np.ceil(x2 * w).astype(int)
    y1 = np.floor(y1 * h).astype(int)
    y2 = np.ceil(y1 * h).astype(int)
```

위의 코드에서 RoI의 좌표를 다루는 부분이 있는데, 이는 코드의 데이터셋이 미리 Selective search를 이미지에 적용해서 RoI를 pkl 형식으로 제공하고 있는데, 이는 원본 이미지 크기에서 RoI가 차지하는 비율 형식으로 되어 있어서 이를 feature map의 크기 14x14에 맞게 절대좌표로 변환해서 feature map 내에서 region proposal이 encode하는 영역을 찾는 것이다.

이어서 RoI Projection을 하는 코드를 이어 붙여보자.


```python
import torch
import torch.nn as nn
import numpy as np

class RoIPool(nn.Module):
  def __init__(self, output_size):
    super().__init__()
    self.maxpool = nn.AdaptiveMaxPool2d(output_size)
    self.size = output_size
  
  def forward(self, images, rois, roi_idx):
    n = rois.shape[0]  # RoI의 개수

    h = images.size(2)  
    w = images.size(3)  

    x1 = rois[:,0]
    y1 = rois[:,1]
    x2 = rois[:,2]
    y2 = rois[:,3]

    x1 = np.floor(x1 * w).astype(int)
    x2 = np.ceil(x2 * w).astype(int)
    y1 = np.floor(y1 * h).astype(int)
    y2 = np.ceil(y1 * h).astype(int)

    # RoI Projection 수행 
    res = []

    # RoI 수만큼 순회
    for i in range(n):  
      img = images[roi_idx[i]].unsqueeze(0) # i번째 roi_idx에 해당하는 feature map
      img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]] # 잘라내기
      img = self.maxpool(img) # Adaptive Max Pooling 적용
      res.append(img)
    res = torch.cat(res,dim=0)
    return res # 7x7 크기의 feature map이 RoI 수만큼 저장된 리스트 반환
```

Max Pooling을 사용할 때 입력 feature map 크기와 출력 feature map의 크기를 고려해서 stride와 kernel의 크기를 조정하는 Adaptive Max Pooling을 사용한다. RoI Projection을 거쳐서 최종적으로 7x7 크기의 feature map이 RoI 수만큼 저장된 리스트가 반환된다.

### 2. Initializing pre-trained CNN
다음으로 사전 훈련된 VGG16 모델을 load 한 후 detection task에 맞게 CNN을 수정하는 코드를 분석해보자.


```python
from torch.autograd.variable import Variable
import torchvision

class RCNN(nn.Module):
  def __init__(self):
    super().__init__()

    rawnet = torchvision.models.vgg16_bn(pretrained=True) # 사전 훈련된 VGG16_bn 모델 로드
    self.seq = nn.Sequential(*list(rawnet.features.children())[:-1]) # 마지막 max pooling 제거
    self.roipool = RoIPool(output_size=(7,7)) # 마지막 pooling layer를 RoI Pooling layer로 대체
    self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1]) # 마지막 fc layer 제거

    _x = Variable(torch.Tensor(1,3,224,224)) # VGG16에 입력되는 데이터의 크기(224x244 RGB) 정의
    _r = np.array([[0., 0., 1., 1.]])
    _ri = np.array([0])

    # 원본 이미지를 conv layer, roi pooling layer, fc layer에 순차적으로 입력해서 고정된 크기 (7x7)의 feature vector 얻는다.
    _x = self.feature(self.roipool(self.seq(_x),_r,_ri).view(1,-1)) 

    feature_dim = _x.size(1)
    self.cls_score = nn.Linear(feature_dim, N_CLASS + 1) # feature vector를 Classifier에 주입
    self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1)) # feature vector를 BBR에 주입
    
    self.cel = nn.CrossEntropyLoss() # 분류 손실함수 정의
    self.sl1 = nn.SmoothL1Loss()  # BBR 손실함수 정의

  def forward(self, inp, rois, ridx):
    res = inp # 입력 이미지 데이터
    res = self.seq(res) # pooling 이전까지의 과정
    res = self.roipool(res,rois,ridx) # RoI Pooling 
    res = res.detach() # 연산 x
    res = res.view(res.size(0), -1) # fc layer에 주입하기 위해 펼치기
    feat = self.feature(res) # fc layer에서 feature 추출

    cls_score = self.cls_score(feat) # 분류 점수
    bbox = self.bbox(feat).view(-1, N_CLASS+1, 4) # BBR 결과

    return cls_score, bbox
```

### 3. Multi-taks Loss
위의 코드에 이어서 Multi-task Loss를 구현한 부분을 분석해보자.


```python
from torch.autograd.variable import Variable
import torchvision

class RCNN(nn.Module):
  def __init__(self):
    super().__init__()

    rawnet = torchvision.models.vgg16_bn(pretrained=True)
    self.seq = nn.Sequential(*list(rawnet.features.children())[:-1]) 
    self.roipool = RoIPool(output_size=(7,7)) 
    self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1]) 

    _x = Variable(torch.Tensor(1,3,224,224)) 
    _r = np.array([[0., 0., 1., 1.]])
    _ri = np.array([0])
    _x = self.feature(self.roipool(self.seq(_x),_r,_ri).view(1,-1)) 

    feature_dim = _x.size(1)
    self.cls_score = nn.Linear(feature_dim, N_CLASS + 1) 
    self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1)) 

    self.cel = nn.CrossEntropyLoss()
    self.sl1 = nn.SmoothL1Loss()  

  def forward(self, inp, rois, ridx):
    res = inp 
    res = self.seq(res) 
    res = self.roipool(res,rois,ridx) 
    res = res.detach() 
    res = res.view(res.size(0), -1) 
    feat = self.feature(res) 

    cls_score = self.cls_score(feat)
    bbox = self.bbox(feat).view(-1, N_CLASS+1, 4) 

    return cls_score, bbox

  def calc_loss(self, probs, bbox, labels, gt_bbox):  # Multi task Loss 구현
    loss_sc = self.cel(probs, labels) # 크로스엔트로피 손실로 Classifier 손실 구현

    lbl = labels.view(-1,1,1).expand(labels.size(0),1,4)
    mask = (labels != 0).float().view(-1,1).expand(labels.size(0),4)
    loss_loc = self.sl1(bbox.gather(1,lbl).squeeze(1) * mask, gt_bbox * mask)

    lmb = 1.0
    loss = loss_sc + lmb * loss_loc

    return loss, loss_sc, loss_loc
```

Classifier의 loss는 Crossentropy loss를 통해서 구하면 되지만, BBR의 loss는 구하기 살짝 복잡하다. Multi-task loss를 다시 한번 보자.

![pyfrcnn2](https://user-images.githubusercontent.com/77332628/211156320-ce1a88c1-0ba0-47ad-b949-abfb23c1dbba.png)

BBR의 loss에는 학습 데이터가 positive/negative sample 여부를 알려주는 index parameter $u$가 곱해져 있다. $u$는 위의 코드에서 mask 변수로 구현되어 있다. mask 변수는 labels에 저장된 배열을 bounding box에 대한 정보가 저장된 배열의 크기에 맞게 변환한 배열이다. mask를 예측 bounding box에 해당하는 bbox 변수와, ground truth box에 해당하는 gt_bbox 변수에 곱해준 후 Smooth L1 loss를 구한다. 그리고 최종적으로 두 loss 사이의 가중치를 조정하는 lambda에 해당하는 lmb 변수를 1로 설정하고 두 loss를 더해서 최종적으로 multi-task loss를 반환한다.


참고한 블로그의 글쓴이는 이 부분을 분석하는 것이 가장 어려웠다고 한다. 

### 4. Fast R-CNN 훈련하기
다음으로는 batch별로 Fast RCNN이 학습하는 과정을 구현한 코드를 살펴보자. 원본 이미지와 RoI를 위에서 정의한 R-CNN에 주입하고, loss를 구하는 과정이 구현되어 있다.


```python
def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val = False):
  sc, r_bbox = RCNN(img, rois, ridx) # class score, bbox를 구한다.
  loss, loss_sc, loss_loc = RCNN.calc_loss(sc, r_bbox, gt_cls, gt_tbbox) # loss 계산
  fl = loss.data.cpu().numpy()[0]
  fl_sc = loss_sc.data.cpu().numpy()[0]
  fl_loc = loss_loc.data.cpu().numpy()[0]

  if not is_val: # 검증 과정이 아닐 경우
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

다음으로는 epoch 별로 학습시키는 과정에서 Hierarchical sampling을 통해 적절한 학습 데이터를 sampling하는 과정을 구현한다. Hierarchical sampling은 간단히 말해서 featuring sharing을 가능하게 해서 연산을 공유할 수 없는 RCNN의 단점을 해결하는 기법이다. 

원본 이미지 중 2장을 sampling 한 후, 각 이미지에서 64장의 RoI를 sampling한다. 이처럼 같은 이미지에서 sampling한 RoI는 forward, back propagation시, **연산과 메모리를 공유할 수 있다.** 전체 RoI 중에서 positive sample은 25%, 나머지는 negative sample로 구성한다.


```python
from tqdm import trange

def train_epoch(run_set, is_val=False):
  I = 2 # number of image
  B = 64 # number of RoIs per image
  POS = int(B*0.25) # positive sample 비율 25%
  NEG = B - POS

  # shuffle images
  Nimg = len(run_set)
  perm = np.random.permutation(Nimg)
  perm = run_set[perm]  

  losses = []
  losses_sc = []
  losses_loc = []

  # 전체 이미지를 I(=2)개씩만큼 처리
  for i in trange(0, Nimg , I):
    lb = i 
    rb = min(i+I, Nimg)
    torch_seg = torch.from_numpy(perm[lb:rb]) # read 2 images
    img = Variable(train_imgs[torch_seg], volatile=is_val)
    ridx = []
    glo_ids = []
```

이어서 positive/negative sample의 index가 저장된 리스트에서 지정한 positive/negative 수에 맞게 sampling 한다. 그리고 이미지 2장에서 sampling 한 RoI를 glo_ids 변수에 저장해서 epoch당 학습 데이터로 사용한다.


```python
def train_epoch(run_set, is_val=False):
  I = 2 
  B = 64 
  POS = int(B*0.25) 
  NEG = B - POS

  Nimg = len(run_set)
  perm = np.random.permutation(Nimg)
  perm = run_set[perm]  

  losses = []
  losses_sc = []
  losses_loc = []

  for i in trange(0, Nimg , I):
    lb = i 
    rb = min(i+I, Nimg)
    torch_seg = torch.from_numpy(perm[lb:rb]) 
    img = Variable(train_imgs[torch_seg], volatile=is_val)
    ridx = []
    glo_ids = []

    # 이어서
    for j in range(lb,rb):
      info = train_img_info[perm[j]]

      # RoI의 positive, negative idx에 대한 리스트
      pos_idx = info['pos_idx']
      neg_idx = info['neg_idx']
      ids = []

      지정한 positive/negative 수에 맞게 sampling
      if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))
      if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
      if len(ids) == 0:
          continue
      ids = np.concatenate(ids, axis=0)

      # glo_ids : 두 이미지에 대한 positive, negative sample의 idx를 저장한 리스트
      glo_dis.append(ids)
      ridx += [j-lb] * ids.shape[0]

      if len(ridx) == 0 :
        continue
      glo_ids = np.concatenate(glo_ids, axis=0)
      ridx = np.array(ridx)

      # Hierarchical sampling을 통해 구성한 학습 데이터를 Fast RCNN에 주입해서 loss를 구한다.
      rois = train_roi[glo_ids]
      gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()
      gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()

      loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
      losses.append(loss)
      losses_sc.append(loss_sc)
      losses_loc.append(loss_loc)

  avg_loss = np.mean(losses)
  avg_loss_sc = np.mean(losses_sc)
  avg_loss_loc = np.mean(losses_loc)
  print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')
    
  return losses, losses_sc, losses_loc
```

is_val을 False로 설정해서 훈련 과정에서 나온 값들을 반환하고, is_val을 True로 설정해서 검증 과정을 거친다.


```python
def start_training(n_epoch=2):
    tl = []
    ts = []
    to = []
    vl = []
    vs = [] 
    vo = []
    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i+1}]')
        train_loss, train_sc, train_loc = train_epoch(train_set, False)
        print(f'[Validation Epoch {i+1}]')
        val_loss, val_sc, val_loc = train_epoch(val_set, True)
        
        tl.append(train_loss)
        ts.append(train_sc)
        to.append(train_loc)
        vl.append(val_loss)
        vs.append(val_sc)
        vo.append(val_loc)
        
    plot('loss','Train/Val : Loss', 'Train', 'Validation', tl, vl, n_epoch)
    plot('loss_sc','Train/Val : Loss_sc', 'Train', 'Validation', ts, vs, n_epoch)
    plot('loss_loc','Train/Val : Loss_loc', 'Train', 'Validation', to, vo, n_epoch)
```

데이터를 로드하는 과정과 검증 셋 데이터를 정의하는 과정과 테스트 셋을 돌리는 과정 등 코드의 모든 과정을 다루지는 못했지만 이렇게 해서 Fast RCNN 모델을 파이토치로 한번 구현해봤다. 

참고  : https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch,

https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work,

https://herbwood.tistory.com/9?category=867198
