---
title: '[Kaggle/CV] Global Wheat - Faster RCNN (2) 🌾'
toc: true
toc_sticky: true
categories:
  - kaggle-objectdetection
---

## Pytorch로 구현한 Faster RCNN으로 Global Wheat Detection 문제풀기 - Inference(Test)

### 0. 데이터 준비하기
이번에는 저번 글([**링크**](https://hamin-chang.github.io/kaggle-objectdetection/kagglewheat1/))에서 구축한 Faster RCNN 모델로 추론을 해볼 거다. 저번과 같이 필요한 라이브러리들을 import하고 필요한 파일들의 경로를 정의할 건데, 이번에는 훈련 과정을 통해 얻은 모델의 가중치 파일도 필요하다. (이번 글은 이 캐글 코드([**링크**](https://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-inferencehttps://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-inference))를 참고했다.)


```python
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
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

# training을 통해 얻은 모델 가중치
DIR_WEIGHTS = '/kaggle/input/global-wheat-detection-public'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/weights/fasterrcnn_resnet50_fpn_best.pth'
```

우리가 추론해야하는 데이터의 형태를 출력해보면 총 10개의 image_id가 있는데, 하나의 image_id 당 (score, 박스좌표)의 형식으로 Prediction String 값을 채우면 되는 것으로 보인다.


```python
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')
print(test_df.head(5),)
print('\n shape of test_df : ',test_df.shape)
```

        image_id PredictionString
    0  aac893a91    1.0 0 0 50 50
    1  51f1be19e    1.0 0 0 50 50
    2  f5a1f0358    1.0 0 0 50 50
    3  796707dd7    1.0 0 0 50 50
    4  51b3e36ab    1.0 0 0 50 50
    
     shape of test_df :  (10, 2)


### 1. 데이터 로드, Albumenatation 클래스 정의하기
train을 할 때와 같이 데이터를 로드하는 WheatDataset을 정의할건데, 이번에는 test를 할 것이기 때문에 저번에 정의했던 WheatDataset과는 달리 **boxes, area, label, iscrowd는 정의하지 않는다.**


```python
class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms = None):
        super().__init__()
        
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index : int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.transforms :
            sample = {
                'image' : image
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
        return image, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]
```

Albumentations를 위한 함수를 정의하는데, 이번에는 테스트 데이터를 준비하는 것이기 때문에 flip은 하지 않는다.


```python
def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ])
```

### 2. Test 하기

먼저 사전 훈련된 모델을 load한다.


```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
```

그 다음에는 사전 훈련된 모델의 가중치를 fine tuning하는 과정을 거친다.


```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2 # 1 wheat class + 1 background class

# 분류기에 사용할 입력 특징의 차원 정보
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 사전 훈련된 모델의 머리 부분을 새로운 것으로 fine tuning
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

# train을 통해 얻은 가중치 load
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)
```

위에서 정의한 WheatDataset과 DataLoader를 이용해서 test data를 로드한다.


```python
def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(test_dataset,
                             batch_size=4,
                             shuffle = False,
                             num_workers = 4,
                             drop_last=False,
                             collate_fn=collate_fn)
```

    /opt/conda/lib/python3.7/site-packages/torch/utils/data/dataloader.py:490: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      cpuset_checked))


제출할 파일인 prediction string에 값을 입력하는 함수를 정의한다.


```python
def format_prediction_string(boxes,scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append('{0:.4f} {1} {2} {3} {4} |'.format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
        
    return " ".join(pred_strings)
```

score가 0.5 이상인 box만 테스트 결과에 포함한다. 이제 본격적인 추론을 시작한다.


```python
detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)
    
    for i , image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:,2] = boxes[:,2]-boxes[:,0]
        boxes[:,3] = boxes[:,3]-boxes[:,1]
        
        result = {
            'image_id' : image_id,
            'PredictionString': format_prediction_string(boxes,scores)
        }
        
        results.append(result)
```

이제 결과를 출력해보자. test data의 첫번째 이미지에 대한 추론 결과를 출력해본다.


```python
results[0]
```




    {'image_id': 'aac893a91',
     'PredictionString': '0.9978 554 531 128 192 | 0.9947 616 920 77 103 | 0.9946 68 2 102 162 | 0.9931 592 780 92 119 | 0.9917 326 666 124 153 | 0.9890 25 457 109 158 | 0.9875 816 703 105 204 | 0.9870 177 566 113 188 | 0.9858 741 774 80 114 | 0.9850 692 392 124 180 | 0.9825 236 842 159 111 | 0.9745 359 533 94 81 | 0.9734 458 861 82 98 | 0.9664 240 85 136 148 | 0.9085 90 621 117 73 | 0.9045 67 859 111 68 | 0.8843 483 8 214 251 | 0.8804 306 0 71 67 | 0.7252 821 632 86 112 | 0.7134 819 918 127 105 | 0.6647 360 268 95 142 |'}




```python
test_df = pd.DataFrame(results, columns = ['image_id','PredictionString'])
test_df.head(10)
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
      <th>PredictionString</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aac893a91</td>
      <td>0.9978 554 531 128 192 | 0.9947 616 920 77 103...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51f1be19e</td>
      <td>0.9973 607 93 162 177 | 0.9950 843 268 133 201...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f5a1f0358</td>
      <td>0.9958 540 276 110 114 | 0.9952 939 435 84 185...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>796707dd7</td>
      <td>0.9927 940 75 84 100 | 0.9922 895 333 113 92 |...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51b3e36ab</td>
      <td>0.9988 836 456 187 146 | 0.9981 544 34 246 130...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>348a992bb</td>
      <td>0.9963 917 569 85 89 | 0.9955 599 448 120 95 |...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cc3532ff6</td>
      <td>0.9989 771 834 167 162 | 0.9981 74 812 140 168...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2fd875eaa</td>
      <td>0.9974 457 502 83 131 | 0.9969 465 357 125 96 ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cb8d261a3</td>
      <td>0.9970 25 562 179 105 | 0.9943 21 867 82 145 |...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>53f253011</td>
      <td>0.9987 929 204 95 137 | 0.9982 464 470 159 205...</td>
    </tr>
  </tbody>
</table>
</div>



출력 결과를 보니 추론이 잘 이루어진것으로 보인다! 이제는 추론 결과를 이미지에 시각화 해보자.


```python
sample = images[0].permute(1,2,0).cpu().numpy()
boxes = outputs[0]['boxes'].data.cpu().numpy()
scores = outputs[0]['scores'].data.cpu().numpy()

boxes = boxes[scores>=detection_threshold].astype(np.int32)
```


```python
fig,ax = plt.subplots(1,1,figsize=(16,8))

for box in boxes:
    cv2.rectangle(sample,
                 (box[0], box[1]),
                 (box[2], box[3]),
                 (220,0,0),2)

ax.set_axis_off()
ax.imshow(sample)
```




    <matplotlib.image.AxesImage at 0x7f405d634a90>




    
![png](20221210_files/20221210_23_1.png)
    


이제 제출할 형태로 저장해주면 끝이다.


```python
test_df.to_csv('submission.csv',index=False)
```
