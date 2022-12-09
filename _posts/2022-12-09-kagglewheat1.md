---
title: '[Kaggle/CV] Global Wheat - Faster RCNN(1) 🌾'
toc: true
toc_sticky: true
categories:
  - kaggle-objectdetection
---
## Faster RCNN으로 Global Wheat Detection 문제풀기 - Train

### 0. 데이터 준비하기
이번 글에서는 [**저번 글**](https://hamin-chang.github.io/cv-objectdetection/ffrcnn/)에서 알아본 Faster RCNN를 Pytorch로 구현해서 주어진 이미지들에서 보리의 머리들이 어디에 있는지 찾는 Object Detection을 진행한다. 먼저 필요한 라이브러리들을 로드하고, 데이터들의 경로를 정의한다. 


```python
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A # pytorch 버전 image augmentation
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
```


```python
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
print(train_df.head())
print(train_df.shape)
```

        image_id  width  height                         bbox   source
    0  b6ab77fd7   1024    1024   [834.0, 222.0, 56.0, 36.0]  usask_1
    1  b6ab77fd7   1024    1024  [226.0, 548.0, 130.0, 58.0]  usask_1
    2  b6ab77fd7   1024    1024  [377.0, 504.0, 74.0, 160.0]  usask_1
    3  b6ab77fd7   1024    1024  [834.0, 95.0, 109.0, 107.0]  usask_1
    4  b6ab77fd7   1024    1024  [26.0, 144.0, 124.0, 117.0]  usask_1
    (147793, 5)


train.csv의 상위 5개를 출력해보니 각 이미지에 보리 머리가 어디에 있는지 알려주는 bounding box 좌표와 이미지의 크기를 train.csv가 담고 있다는 것을 알 수 있고, 총 147793개의 이미지 정보를 담고 있다.

그리고 bbox에 있는 값들을 x,y,w,h 값으로 나눠서 train_df에 입력해준다.


```python
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)",x)) # 정규 표현식
    if len(r) == 0:
        r = [-1,-1,-1,-1]
    return r

train_df[['x','y','w','h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float32)
train_df['y'] = train_df['y'].astype(np.float32)
train_df['w'] = train_df['w'].astype(np.float32)
train_df['h'] = train_df['h'].astype(np.float32)
```


```python
train_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>width</th>
      <th>height</th>
      <th>source</th>
      <th>x</th>
      <th>y</th>
      <th>w</th>
      <th>h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>usask_1</td>
      <td>834.0</td>
      <td>222.0</td>
      <td>56.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>usask_1</td>
      <td>226.0</td>
      <td>548.0</td>
      <td>130.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>usask_1</td>
      <td>377.0</td>
      <td>504.0</td>
      <td>74.0</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>usask_1</td>
      <td>834.0</td>
      <td>95.0</td>
      <td>109.0</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b6ab77fd7</td>
      <td>1024</td>
      <td>1024</td>
      <td>usask_1</td>
      <td>26.0</td>
      <td>144.0</td>
      <td>124.0</td>
      <td>117.0</td>
    </tr>
  </tbody>
</table>
</div>



이제 전체 데이터를 훈련 데이터와 검증 데이터로 나눠준다.


```python
image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-665:] # 665개의 이미지를 valid_data로 사용
train_ids = image_ids[:-655] # 2718개의 이미지를 trian에 사용
```


```python
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape, train_df.shape
```




    ((25006, 8), (123025, 8))



### 1. 데이터 로드, Albumentation 클래스 정의하기
그 다음 데이터를 생성하는 WheatDataset이라는 클래스를 정의한다.


```python
class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index:int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg',cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        boxes = records[['x','y','w','h']].values
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        area = torch.as_tensor(area, dtype = torch.float32)
        
        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],),dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['istarget'] = iscrowd
        
        if self.transforms:
            sample = {
                'image':image,
                'bboxes' : target['boxes'],
                'labels' : labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)
        return image, target, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]
```

데이터를 생성하는 클래스를 생성한 다음, Albumentations를 하는 함수들을 정의해준다.


```python
# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format':'pascal_voc','label_fields':['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format':'pascal_voc','label_fields':['labels']})
```

### 2. 모델 구축하기
이제 모델을 구축해보자. 먼저 Faster RCNN에서 사용할 사전 훈련된 모델을 load한다. 사전 훈련된 모델은 ResNet50으로 사용한다.


```python
# model pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```

    Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth





      0%|          | 0.00/160M [00:00<?, ?B/s]



그 다음 사전 훈련된 모델을 fine-tuning 해준다.


```python
num_classes = 2 # 1 class(wheat) + 배경(아무 object도 없음)

# 분류기에 사용할 입력 특징의 차원 정보
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 사전 훈련된 모델의 머리 부분을 새로운 것으로 fine tuning
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
```


```python
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
        
    @property
    def value(self):
        if self.iterations == 0 :
            return 0
        else:
            return 1.0*self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations =0.0
```


```python
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_fn)

valid_data_loader = DataLoader(valid_dataset,
                              batch_size=8,
                              shuffle = False,
                              num_workers = 4,
                              collate_fn= collate_fn)
```

    /opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py:490: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))



```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

 ### 3. 샘플 데이터 출력하기


```python
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images) 
targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
```


```python
boxes = targets[4]['boxes'].cpu().numpy().astype(np.int32)
sample = images[4].permute(1,2,0).cpu().numpy()
```


```python
fig, ax = plt.subplots(1,1,figsize=(16,8))
for box in boxes:
    cv2.rectangle(sample,
                 (box[0],box[1]),
                 (box[2],box[3]),
                 (220,0,0),3)

ax.set_axis_off()
ax.imshow(sample)
```




    <matplotlib.image.AxesImage at 0x7fadeea71c10>




    
![111](https://user-images.githubusercontent.com/77332628/206717442-0e959229-6070-4675-952b-df9f4b6c140e.png)
    


### 4. 모델 훈련하기


```python
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

num_epochs=2
```


```python
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        
        if itr % 10 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")   
```

    Iteration #10 loss: 0.8767896085686229
    Iteration #20 loss: 1.0034717957629287
    Iteration #30 loss: 0.9212688329972594
    Iteration #40 loss: 0.8998036128129249
    Iteration #50 loss: 1.0090722438562278
    Iteration #60 loss: 0.9332446984702532
    Iteration #70 loss: 0.9520588092352517
    Iteration #80 loss: 0.8880465836542588
    Iteration #90 loss: 1.2082256340983488
    Iteration #100 loss: 0.7833284031450136
    Iteration #110 loss: 0.695484001848069
    Iteration #120 loss: 0.7146148842041496
    Iteration #130 loss: 0.7105493748761558
    Iteration #140 loss: 0.9121268667220603
    Iteration #150 loss: 0.7621172243502237
    Iteration #160 loss: 0.8594372129115371
    Iteration #170 loss: 0.7435640633778376
    Epoch #0 loss: 0.8754310350903002
    Iteration #180 loss: 0.8129185204232254
    Iteration #190 loss: 0.9815745474449904
    Iteration #200 loss: 0.8815886739787134
    Iteration #210 loss: 0.8623899764875205
    Iteration #220 loss: 0.9818613345877085
    Iteration #230 loss: 0.9182466013264382
    Iteration #240 loss: 0.9697364914691113
    Iteration #250 loss: 0.8432915679451991
    Iteration #260 loss: 1.1767832955492075
    Iteration #270 loss: 0.8194232906491191
    Iteration #280 loss: 0.6887506687183993
    Iteration #290 loss: 0.7078626006163129
    Iteration #300 loss: 0.7143421875285303
    Iteration #310 loss: 0.9085941208984142
    Iteration #320 loss: 0.7327133478059575
    Iteration #330 loss: 0.8352103870566991
    Iteration #340 loss: 0.7305171478843716
    Epoch #1 loss: 0.8581980241023012



```python
images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k : v.to(device) for k, v in t.items()} for t in targets]
```

    /opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py:490: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))



```python
boxes = targets[4]['boxes'].cpu().numpy().astype(np.int32)
sample = images[4].permute(1,2,0).cpu().numpy()
```


```python
model.eval()
cpu_device = torch.device('cpu')

outputs = model(images)
outputs = [{k : v.to(cpu_device) for k,v in t.items()} for t in outputs]
```


```python
fig,ax = plt.subplots(1,1,figsize=(16,8))

for box in boxes:
    cv2.rectangle(sample,
                 (box[0],box[1]),
                 (box[2],box[3]),
                 (220,0,0),3)
    

ax.set_axis_off()
ax.imshow(sample)
```




    <matplotlib.image.AxesImage at 0x7fadef73fe10>




    
![222](https://user-images.githubusercontent.com/77332628/206717450-a95f4a81-7d12-48a0-b86c-384db1731d9a.png)    



```python
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
```
