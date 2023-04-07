---
title : '[CV/Pytorch] 파이토치로 YOLO v1 구현하기 🤟'
layout: single
toc: true
toc_sticky: true
categories:
  - pytorchOD
---

## Pytorch로 YOLO v1 구현하기

이번 글에서는 pytorch를 사용해서 YOLO v1을 구현해볼건데, [aladdinpersson님의 github repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO)에 올라온 코드를 리뷰해 볼 것이다. YOLO v1 모델에 대한 설명은 이전 글([**링크**](https://hamin-chang.github.io/cv-objectdetection/yolov1/))를 참고하면 되겠다. 

### 1. DarkNet 구현하기

![111](https://user-images.githubusercontent.com/77332628/214053977-e412dbe6-b054-40c5-82d5-02c9d7e9595c.png)

DarkNet은 위 이미지처럼 network의 최종 feature map의 크기가 7x7x30이 되도록 설계하면 되는데, 코드가 흥미로운 점은 network의 각 conv layer의 하이퍼파라미터값을 config 변수에 저장한 후 이를 불러와 사용한다는 것이다.

architecture_config 리스트의 각 요소는 conv layer의 하이퍼파라미터인 (kernel_size, num_filters, stride, padding)이 튜플 형식으로 저장되어 있고, 중간의 'M' 문자열은 max pooling을 의미한다. 또한 리스트 요소는 마지막 정수값만큼 layer를 반복한다.



```python
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
```

그리고 나서 architecture_config 리스트 요소의 type에 따라서 조건문으로 서로 다른 layer를 추가함으로써 모델을 설계한다. 만약 리스트 요소가 튜플이면 해당 하이퍼파라미터에 맞는 conv layer를, 문자열이면 max pooling을, 리스트면 마지막 정수값만큼 layer를 반복해서 전체적인 DarkNet 모델을 구성한다.


```python
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CNNBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.batchnorm = nn.BatchNorm2d(out_channels)
    self.leakyrelu = nn.LeakyReLU(0.1)

  def forward(self,x):
    return self.leakyrelu(self.batchnorm(self.conv(x)))
  
class Yolov1(nn.Module):
  def __init__(self, in_channels=3, **kwargs):
    super(Yolov1, self).__init__()
    self.architecture = architecture_config
    self.in_channels = in_channels
    self.darknet = self._create_conv_layers(self.architecture)
    self.fcs = self._create_fcs(**kwargs)
  
  def forward(self,x):
    x = self.darknet(x)
    return self.fcs(torch.flatten(x, start_dim=1))

  def _create_conv_layers(self, architecture):
    layers = []
    in_channels = self.in_channels

    for x in architecture:
      if type(x) == tuple:
        layers += [CNNBlock(in_channels, x[1], kernel_size = x[0], strid = x[2], padding = x[3])]
        in_channels = x[1]

      elif type(x) == str:
        layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

      elif type(x) == list:
        conv1 = x[0]
        conv2 = x[1]
        num_repeats = x[2]

        for _ in range(num_repeats):
          layers += [CNNBlock(in_channels, conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3])]
          layers += [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], stride = conv2[2], padding = conv2[3])]
          in_channels = conv2[1]

    return nn.Sequential(*layers)

  def _create_fcs(self, split_size, num_boxes, num_classes):
    S, B, C = split_size, num_boxes, num_classes

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * S * S, 496),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S*S*(C+B*5))
    )

```

### 2. YOLO v1 손실함수 구현

![222](https://user-images.githubusercontent.com/77332628/214053979-398764af-147d-4f14-9738-ec1c323dceec.png)

이번 코드 리뷰에서 중점적으로 살펴봐야할 부분이다. 구현하는 부분에서 최종 feature map에 대해서 처리해줘야 할 과정들이 몇가지 있다.

우선 YoloLoss 클래스로 정의해서 grid의 크기 S, grid cell 별 예측 bounding box의 수를 B, 예측하는 class의 수 C를 생성자로 받고 가중치 파라미터인 $λ_{coord}$, $λ_{noobj}$도 정의해준다.


```python
from utils import intersection_over_union

class YoloLoss(nn.Module):
  def __init__(self, S=7, B=2, C=20):
    super(YoloLoss,self).__init__()
    self.mse = nn.MSELoss(reduction='sum')
    self.S = S
    self.B = B
    self.C = C

    self.lambda_coord = 5
    self.lambda_noobj = 0.5

    # 이어서 
```

그리고 forward pass 시 처리할 과정을 정의한다. 각 grid cell마다 2개의 bounding box를 예측하고 그 중 confidence score가 높은 1개의 bounding box를 학습에 사용하는 과정을 구현한다. 



```python
  # 이어서
  def forward(self, predictions,target):
    # DarkNet이 최종적으로 출력하는 7x7x30 크기의 feature map을 flatten한 결과 [c1, c2, ..., c20, p_c1, x, y, w, h, p_c2, x, y, w, h]
    predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

    # 정답값인 target 좌표와 비교해서 IoU 계산
    iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # predictions[...,21:25] => 첫 번째 bounding box의 좌표값
    iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # predictions[...,26:30] => 두 번째 bounding box의 좌표값
    
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

    iou_maxes, bestbox = torch.max(ious, dim=0
                                   )
    # target[...,20]를 통해서 해당 grid cell에 ground truth box의 중심이 존재하는지 여부를 확인
    # 약 존재한다면 exist_box = 1, 아니면 exist_box = 0
    exists_box = target[...,20].unsqueeze(3)

    # 이어서

```

이제 먼저 **Localization Loss**를 구현해보자. best_box 변수를 활용해서 실제 bounding box 예측 중 IoU 값이 더 큰 box를 최종 예측으로 사용한다. 그리고 width, height 값에는 루트를 씌워주고 그 다음 bounding box 좌표값에 대하여 mse를 계산해준다.


```python
    # 이어서
    box_predictions = exists_box * ((bestbox * predictions[..., 26:30] + (1-bestbox) * predictions[..., 21:25]))
    box_targets = exists_box * target[..., 21:25]

    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
    box_loss = self.mse(torch.flatten(box_predictions, end_dim = -2),
                        torch.flatten(box_targets, end_dim = -2))
    
    # 이어서
    
```

다음으로 Confidence loss를 구현할건데, 먼저 object가 실제로 존재할 경우의 confidence loss부터 구한다. predictions[...,25:26]은 첫 번째 box의 confidence score를, prediction[..., 20:21]은 두 번째 box의 confidence score를 의미한다. exists_box 변수를 통해 grid cell에 할당된 ground truth box의 중심이 있는 경우에만 loss를 구한다.


```python
    # 이어서

    # 가장 높은 IoU를 가진 bbox에 대한 confidence score
    pred_box = (bestbox * predictions[...,25:26] + (1-bestbox) * predictions[...,20:21])

    object_loss = self.mse(torch.flatten(exists_box * pred_box),
                           torch.flatten(exists_box * target[..., 20:21]))
    
    # 이어서
```

다음으로는 object가 없을 경우의 confidence loss를 구현하는 과정을 살펴보자. 이 경우 두 bounding box를 모두 학습에 참여시킨다. 


```python
    # 이어서

    no_object_loss = self.mse(torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim = 1),
                              torch.flatten((1-exists_box) * target[..., 20:21], start_dim = 1))
    
    no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                               torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))

    # 이어서
```

마지막으로 Class loss를 구한다. predictions[..., :20]에 해당하는, 즉 20개의 class의 score를 target과 비교해서 mse loss를 구하고 이후 YoloLoss의 생성자에서 정의한 가중치 파라미터를 각각 곱해주고 localization, confidence, class loss를 모두 더해서 최종 loss를 구한다.


```python
    # 이어서

    class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim= -2),
                          torch.flatten(exists_box * target[..., 20], end_dim= -2))
    
    loss = (self.lambda_coord * box_loss 
            + object_loss 
            + self.lambda_noobj * no_object_loss
            + class_loss)
    
    return loss
```

### 3. Custom Dataset

마지막으로는 Dataset을 정의하는 부분이다. 이 부분에서 이미지의 각 grid cell에 ground truth box의 중심이 있는지 지정해준다.


```python
import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.S = S 
    self.B = B
    self.C = C

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
    boxes = []
    with open(label_path) as f:
      for label in f.readlines():
        class_label,x,y,width,height = [float(x) if float(x) != int(float(x)) else int(x)
                                        for x in label.replace("\n","").split()]
        boxes.append([class_label, x, y, width, height])

      img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
      image = Image.open(img_path)
      boxes = torch.tensor(boxes)

      if self.transform:
          image, boxes = self.transform(image, boxes)
        
      label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # grid cell과 같은 7x7x30 배열
      for box in boxes :
        # boxes 변수에 전체 ground truth box의 [x,y,w,h]가 저장 되어있어서
        class_label, x, y, width, height = box.tolist()
        class_label = int(class_label)

        i, j = int(self.S * y), int(self.S * x) # i,j는 cell의 행과 열 

        # 각각의 ground truth box를 순회하면서 ground truth box의 중심 좌표를 계산하고,
        x_cell, y_cell = self.S * x - j, self.S * y - i
        width_cell, height_cell = (width * self.S , height * self.S)

        # label_matrix에 confidence score와 bounding box 좌표 저장하는데,
        # ground truth box의 중심이 특정 cell에 존재하면 해당 cell의 20번째 index(confidence score) = 1 지정
        if label_matrix[i, j, 20] == 0:
          label_matrix[i, j, 20] = 1

          # bbox 좌표
          box_coordinates = torch.tensor(
              [x_cell, y_cell, width_cell, height_cell])
          label_matrix[i, j, 21:25] = box_coordinates

          # class_label에는 one hot encoding 설정
          label_matrix[i, j, class_label] = 1
      
      return image, label_matrix
```

지금까지는 YOLO v1을 구현하기 위해서 모델을 설계하고 손실함수를 정의하고 PASCAL VOC 데이터셋을 로드하기 위한 코드를 구현해봤다. 다음 글에서는 YOLO v1 모델을 실제로 훈련시켜보도록 하겠다.

출처 : 

[aladdinpersson님의 github repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO)

개인 블로그 (http://herbwood.tistory.com/14)
