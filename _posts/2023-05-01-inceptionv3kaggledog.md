---
title: '[IC/Kaggle] Dog Breed Classification - 강아지 종 분류하기 🐶 '
toc: true
toc_sticky: true
categories:
  - kaggle-imageclassification
---

## Inception-v3로 강아지 종류 분류하기

이번 글에서는 이미지의 강아지의 종을 Inception-v3 모델을 이용해서 분류한다. 입력 이미지에서 YOLOv3를 이용해서 강아지가 있는 bounding box를 찾아서 이미지를 강아지 bounding box에 맞게 crop하고 나서 Inception-v3를 적용한다. 이번 글에서는 YOLOv3 모델을 이용해서 bbox 좌표를 찾는 과정은 생략한다. YOLOv3 부분은 [**<U>gabrielloye의 repository</U>**](https://github.com/gabrielloye/yolov3-stanford-dogs/blob/master/main.ipynb)에 나와있다. (YOLOv3 모델에 대한 설명은 [**<U>YOLOv3 논문 리뷰</U>**](https://hamin-chang.github.io/cv-objectdetection/yolov3/)를 참고하기 바란다.

이번 글의 전체 코드는 [**<U>GABRIEL의 notebook</U>**](https://www.kaggle.com/code/gabrielloye/dogs-inception-pytorch-implementation)의 코드를 사용했고, Inception-v3 모델에 대한 설명은 [**<U>Inception-v2,v3 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/inceptionv3/)를 참고하길 바란다.

### 1. 데이터 준비

먼저 필요한 라이브러리를 import 한다.


```python
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn

from PIL import Image
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import xml.etree.ElementTree as ET
```

이제 강아지가 있는 bounding box의 좌표를 이용해서 이미지를 강아지에 맞게 crop 해준다.


```python
def crop_image(breed, dog, data_dir):
    img = plt.imread(data_dir + 'images/Images/' + breed + '/' + dog + '.jpg')
    tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + dog)
    xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
    xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
    ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
    ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
    img = img[ymin:ymax, xmin:xmax, :]
    return img
```

crop_img 함수를 잘 정의했는지 랜덤으로 4개의 이미지를 불러와서 crop해보자.


```python
data_dir = '../input/stanford-dogs-dataset/'
breed_list = os.listdir(data_dir + 'images/Images/')

plt.figure(figsize=(20,20))
for i in range(4):
    plt.subplot(421 + (i*2))
    breed = np.random.choice(breed_list)
    dog = np.random.choice(os.listdir(data_dir+'annotations/Annotation/'+breed))
    img = plt.imread(data_dir + 'images/Images/'+breed+'/'+dog+'.jpg')
    plt.imshow(img)
    
    tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + dog)
    xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
    xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
    ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
    ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    crop_img = crop_image(breed,dog,data_dir)
    plt.subplot(422+(i*2))
    plt.imshow(crop_img)
```




    
![0](https://user-images.githubusercontent.com/77332628/235424561-68983948-33f2-463f-852b-815cfd29da0c.png)





```python
# crop된 이미지 따로 저장
if 'data' not in os.listdir():
    os.mkdir('data')

for breed in breed_list:
    os.mkdir('data/' + breed)
print('crop image를 breed별로 저장할 {}개의 폴더 생성완료!'.format(len(os.listdir('data'))))
```

    crop image를 breed별로 저장할 120개의 폴더 생성완료!



```python
for breed in os.listdir('data'):
    for file in os.listdir(data_dir + 'annotations/Annotation/' + breed):
        img = Image.open(data_dir + 'images/Images/' + breed + '/' + file + '.jpg')
        tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + file)
        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
        img = img.crop((xmin,ymin,xmax,ymax))
        img = img.convert('RGB')
        img.save('data/' + breed + '/' + file + '.jpg')
```


```python
img_count = 0
for folder in os.listdir('data'):
    for _ in os.listdir('data/'+folder):
        img_count +=1

print('이미지 수 : {}'.format(img_count))
```

    이미지 수 : 20580


이제 crop된 이미지 데이터들이 다 준비되었기 때문에 augmentation과 normalization을 적용해준다. 이때 train과 test 데이터 모두 299x299로 resize하는데 이는 Inceptionv3 모델에 입력 데이터 사이즈에 맞추기 위해서이다.


```python
image_transforms = {
    # data augmentation은 train에만 사용
    'train' : 
    transforms.Compose([
        transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # ImageNet standard
    ]),
    
    'test':
    transforms.Compose([
        transforms.Resize(size=299),
        transforms.CenterCrop(size=299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # ImageNet standard
    ])
}
```

이제 batch size를 128로 정하고 전체 데이터를 train, validation, test로 나눠준다.


```python
batch_size = 128

all_data = datasets.ImageFolder(root='data')
train_data_len = int(len(all_data)*0.8)
valid_data_len = int((len(all_data) - train_data_len)/2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['test']
test_data.dataset.transform = image_transforms['test']
print(len(train_data), len(val_data), len(test_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
```

    16464 2058 2058
    
### 2. Inception-v3 fine tuning

이제 사전 훈련된 inception-v3를 불러오자.

![1](https://user-images.githubusercontent.com/77332628/235423939-a5895072-fc42-4f28-8534-3134b3685923.png)

```python
model = models.inception_v3(pretrained=True)
model.aux_logits = False
model
```

    Downloading: "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth" to /root/.torch/models/inception_v3_google-1a9a5a14.pth
    108857766it [00:03, 34279419.12it/s]





    Inception3(
      (Conv2d_1a_3x3): BasicConv2d(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (Conv2d_2a_3x3): BasicConv2d(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (Conv2d_2b_3x3): BasicConv2d(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (Conv2d_3b_1x1): BasicConv2d(
        (conv): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (Conv2d_4a_3x3): BasicConv2d(
        (conv): Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (Mixed_5b): InceptionA(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_1): BasicConv2d(
          (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_2): BasicConv2d(
          (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3): BasicConv2d(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_5c): InceptionA(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_1): BasicConv2d(
          (conv): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_2): BasicConv2d(
          (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3): BasicConv2d(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_5d): InceptionA(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_1): BasicConv2d(
          (conv): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch5x5_2): BasicConv2d(
          (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3): BasicConv2d(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_6a): InceptionB(
        (branch3x3): BasicConv2d(
          (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3): BasicConv2d(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_6b): InceptionC(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_1): BasicConv2d(
          (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_2): BasicConv2d(
          (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_3): BasicConv2d(
          (conv): Conv2d(128, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_1): BasicConv2d(
          (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_2): BasicConv2d(
          (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_3): BasicConv2d(
          (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_4): BasicConv2d(
          (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_5): BasicConv2d(
          (conv): Conv2d(128, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_6c): InceptionC(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_1): BasicConv2d(
          (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_2): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_3): BasicConv2d(
          (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_1): BasicConv2d(
          (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_2): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_3): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_4): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_5): BasicConv2d(
          (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_6d): InceptionC(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_1): BasicConv2d(
          (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_2): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_3): BasicConv2d(
          (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_1): BasicConv2d(
          (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_2): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_3): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_4): BasicConv2d(
          (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_5): BasicConv2d(
          (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_6e): InceptionC(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_2): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7_3): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_2): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_3): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_4): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7dbl_5): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (AuxLogits): InceptionAux(
        (conv0): BasicConv2d(
          (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv1): BasicConv2d(
          (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (fc): Linear(in_features=768, out_features=1000, bias=True)
      )
      (Mixed_7a): InceptionD(
        (branch3x3_1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_2): BasicConv2d(
          (conv): Conv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7x3_1): BasicConv2d(
          (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7x3_2): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7x3_3): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch7x7x3_4): BasicConv2d(
          (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_7b): InceptionE(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_1): BasicConv2d(
          (conv): Conv2d(1280, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_2a): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_2b): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(1280, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3a): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3b): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(1280, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (Mixed_7c): InceptionE(
        (branch1x1): BasicConv2d(
          (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_1): BasicConv2d(
          (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_2a): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3_2b): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_1): BasicConv2d(
          (conv): Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_2): BasicConv2d(
          (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3a): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch3x3dbl_3b): BasicConv2d(
          (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
          (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch_pool): BasicConv2d(
          (conv): Conv2d(2048, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )



사전 훈련된 inception-v3에서 가중치를 훈련시키지 않을 layer(classifier를 제외한 layer)를 동결시킨다.


```python
for param in model.parameters():
    param.requires_grad = False
```

이제 fc layer를 120개의 output(class 개수)를 도출하도록 다시 정의해준다.


```python
n_classses = 120
n_inputs = model.fc.in_features # 4096 in this case
model.fc = nn.Sequential(nn.Linear(n_inputs, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(1024, n_classses),
                        nn.LogSoftmax(dim=1))
model.fc
```




    Sequential(
      (0): Linear(in_features=2048, out_features=1024, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.4)
      (3): Linear(in_features=1024, out_features=120, bias=True)
      (4): LogSoftmax()
    )

### 3. Model Training & Validation

모델을 GPU로 옮기고 손실함수와 optimizer를 정의해주고, 120개의 class에 index를 지정해준다.


```python
# GPU로 모델 이동
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
```


```python
model.class_to_idx = all_data.class_to_idx
model.idx_to_class = {
    idx : class_
    for class_, idx in model.class_to_idx.items()
}
list(model.idx_to_class.items())
```




    [(0, 'n02085620-Chihuahua'),
     (1, 'n02085782-Japanese_spaniel'),
     (2, 'n02085936-Maltese_dog'),
     (3, 'n02086079-Pekinese'),
     (4, 'n02086240-Shih-Tzu'),
     (5, 'n02086646-Blenheim_spaniel'),
     (6, 'n02086910-papillon'),
     (7, 'n02087046-toy_terrier'),
     (8, 'n02087394-Rhodesian_ridgeback'),
     (9, 'n02088094-Afghan_hound'),
     (10, 'n02088238-basset'),
     (11, 'n02088364-beagle'),
     (12, 'n02088466-bloodhound'),
     (13, 'n02088632-bluetick'),
     (14, 'n02089078-black-and-tan_coonhound'),
     (15, 'n02089867-Walker_hound'),
     (16, 'n02089973-English_foxhound'),
     (17, 'n02090379-redbone'),
     (18, 'n02090622-borzoi'),
     (19, 'n02090721-Irish_wolfhound'),
     (20, 'n02091032-Italian_greyhound'),
     (21, 'n02091134-whippet'),
     (22, 'n02091244-Ibizan_hound'),
     (23, 'n02091467-Norwegian_elkhound'),
     (24, 'n02091635-otterhound'),
     (25, 'n02091831-Saluki'),
     (26, 'n02092002-Scottish_deerhound'),
     (27, 'n02092339-Weimaraner'),
     (28, 'n02093256-Staffordshire_bullterrier'),
     (29, 'n02093428-American_Staffordshire_terrier'),
     (30, 'n02093647-Bedlington_terrier'),
     (31, 'n02093754-Border_terrier'),
     (32, 'n02093859-Kerry_blue_terrier'),
     (33, 'n02093991-Irish_terrier'),
     (34, 'n02094114-Norfolk_terrier'),
     (35, 'n02094258-Norwich_terrier'),
     (36, 'n02094433-Yorkshire_terrier'),
     (37, 'n02095314-wire-haired_fox_terrier'),
     (38, 'n02095570-Lakeland_terrier'),
     (39, 'n02095889-Sealyham_terrier'),
     (40, 'n02096051-Airedale'),
     (41, 'n02096177-cairn'),
     (42, 'n02096294-Australian_terrier'),
     (43, 'n02096437-Dandie_Dinmont'),
     (44, 'n02096585-Boston_bull'),
     (45, 'n02097047-miniature_schnauzer'),
     (46, 'n02097130-giant_schnauzer'),
     (47, 'n02097209-standard_schnauzer'),
     (48, 'n02097298-Scotch_terrier'),
     (49, 'n02097474-Tibetan_terrier'),
     (50, 'n02097658-silky_terrier'),
     (51, 'n02098105-soft-coated_wheaten_terrier'),
     (52, 'n02098286-West_Highland_white_terrier'),
     (53, 'n02098413-Lhasa'),
     (54, 'n02099267-flat-coated_retriever'),
     (55, 'n02099429-curly-coated_retriever'),
     (56, 'n02099601-golden_retriever'),
     (57, 'n02099712-Labrador_retriever'),
     (58, 'n02099849-Chesapeake_Bay_retriever'),
     (59, 'n02100236-German_short-haired_pointer'),
     (60, 'n02100583-vizsla'),
     (61, 'n02100735-English_setter'),
     (62, 'n02100877-Irish_setter'),
     (63, 'n02101006-Gordon_setter'),
     (64, 'n02101388-Brittany_spaniel'),
     (65, 'n02101556-clumber'),
     (66, 'n02102040-English_springer'),
     (67, 'n02102177-Welsh_springer_spaniel'),
     (68, 'n02102318-cocker_spaniel'),
     (69, 'n02102480-Sussex_spaniel'),
     (70, 'n02102973-Irish_water_spaniel'),
     (71, 'n02104029-kuvasz'),
     (72, 'n02104365-schipperke'),
     (73, 'n02105056-groenendael'),
     (74, 'n02105162-malinois'),
     (75, 'n02105251-briard'),
     (76, 'n02105412-kelpie'),
     (77, 'n02105505-komondor'),
     (78, 'n02105641-Old_English_sheepdog'),
     (79, 'n02105855-Shetland_sheepdog'),
     (80, 'n02106030-collie'),
     (81, 'n02106166-Border_collie'),
     (82, 'n02106382-Bouvier_des_Flandres'),
     (83, 'n02106550-Rottweiler'),
     (84, 'n02106662-German_shepherd'),
     (85, 'n02107142-Doberman'),
     (86, 'n02107312-miniature_pinscher'),
     (87, 'n02107574-Greater_Swiss_Mountain_dog'),
     (88, 'n02107683-Bernese_mountain_dog'),
     (89, 'n02107908-Appenzeller'),
     (90, 'n02108000-EntleBucher'),
     (91, 'n02108089-boxer'),
     (92, 'n02108422-bull_mastiff'),
     (93, 'n02108551-Tibetan_mastiff'),
     (94, 'n02108915-French_bulldog'),
     (95, 'n02109047-Great_Dane'),
     (96, 'n02109525-Saint_Bernard'),
     (97, 'n02109961-Eskimo_dog'),
     (98, 'n02110063-malamute'),
     (99, 'n02110185-Siberian_husky'),
     (100, 'n02110627-affenpinscher'),
     (101, 'n02110806-basenji'),
     (102, 'n02110958-pug'),
     (103, 'n02111129-Leonberg'),
     (104, 'n02111277-Newfoundland'),
     (105, 'n02111500-Great_Pyrenees'),
     (106, 'n02111889-Samoyed'),
     (107, 'n02112018-Pomeranian'),
     (108, 'n02112137-chow'),
     (109, 'n02112350-keeshond'),
     (110, 'n02112706-Brabancon_griffon'),
     (111, 'n02113023-Pembroke'),
     (112, 'n02113186-Cardigan'),
     (113, 'n02113624-toy_poodle'),
     (114, 'n02113712-miniature_poodle'),
     (115, 'n02113799-standard_poodle'),
     (116, 'n02113978-Mexican_hairless'),
     (117, 'n02115641-dingo'),
     (118, 'n02115913-dhole'),
     (119, 'n02116738-African_hunting_dog')]



훈련을 위한 함수를 정의하자. 2 epochs마다 validation 과정에서 loss가 감소하고 있는지 확인하고, loss가 3 epochs 만큼 지나도 감소하지 않으면 훈련을 중단하고 loss가 가장 낮을 때의 가중치를 저장한다.


```python
def train(model, criterion, optimizer, train_loader, val_loader,
         save_location, early_stop=3, n_epochs=20, print_every=2):
    valid_loss_min = np.Inf
    stop_count = 0
    valid_max_acc = 0
    history = []
    model.epochs = 0
    
    # 훈련 루프 시작
    for epoch in range(n_epochs):
        train_loss = 0
        valid_loss = 0
        
        train_acc = 0
        valid_acc = 0
        
        model.train()
        ii = 0
        
        for data, label in train_loader:
            ii += 1
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # average loss * number of sample in batch로 train loss 추적
            train_loss += loss.item() * data.size(0)
            
            # accuracy 계산
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)
            if ii%10 == 0:
                print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.')
        
        # validation
        model.epochs += 1 
        with torch.no_grad():
            model.eval()
            
            for data, label in val_loader:
                data, label = data.cuda(), label.cuda()
                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)
                
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(label.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_acc += accuracy.item() * data.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(val_loader.dataset)
            
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(val_loader.dataset)
            
            history.append([train_loss, valid_loss, train_acc, valid_acc])
            
            if (epoch + 1) & print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                
            if valid_loss < valid_loss_min:
                torch.save({
                    'state_dict' : model.state_dict(),
                    'idx_to_class' : model.idx_to_class
                }, save_location)
                stop_count = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch
            
            else:
                stop_count += 1
                
                # early stopping 사용할 시
                if stop_count >= early_stop :
                    print(f'\nEarly Stopping Total epochs : {epoch}. Best epoch : {best_epoch} with loss : {valid_loss_min:.2f} and acc : {100*valid_acc:.2f}%')
                    model.load_state_dict(torch.load(save_location)['state_dict'])
                    model.optimizer = optimizer
                    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
                    return model, history
    
    model.optimizer = optimizer
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
    
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history
```

이제 모델을 훈련시키자.


```python
model, history = train(model, criterion, optimizer, train_loader,
                      val_loader, save_location='./dog_inception.pt', early_stop=3,
                      n_epochs = 20, print_every=2)
```

    Epoch: 0	8.53% complete.
    Epoch: 0	16.28% complete.
    Epoch: 0	24.03% complete.
    Epoch: 0	31.78% complete.
    Epoch: 0	39.53% complete.
    Epoch: 0	47.29% complete.
    Epoch: 0	55.04% complete.
    Epoch: 0	62.79% complete.
    Epoch: 0	70.54% complete.
    Epoch: 0	78.29% complete.
    Epoch: 0	86.05% complete.
    Epoch: 0	93.80% complete.
    
    Epoch: 0 	Training Loss: 0.6162 	Validation Loss: 0.7186
    		Training Accuracy: 81.01%	 Validation Accuracy: 78.28%
    Epoch: 1	8.53% complete.
    Epoch: 1	16.28% complete.
    Epoch: 1	24.03% complete.
    Epoch: 1	31.78% complete.
    Epoch: 1	39.53% complete.
    Epoch: 1	47.29% complete.
    Epoch: 1	55.04% complete.
    Epoch: 1	62.79% complete.
    Epoch: 1	70.54% complete.
    Epoch: 1	78.29% complete.
    Epoch: 1	86.05% complete.
    Epoch: 1	93.80% complete.
    Epoch: 2	8.53% complete.
    Epoch: 2	16.28% complete.
    Epoch: 2	24.03% complete.
    Epoch: 2	31.78% complete.
    Epoch: 2	39.53% complete.
    Epoch: 2	47.29% complete.
    Epoch: 2	55.04% complete.
    Epoch: 2	62.79% complete.
    Epoch: 2	70.54% complete.
    Epoch: 2	78.29% complete.
    Epoch: 2	86.05% complete.
    Epoch: 2	93.80% complete.
    Epoch: 3	8.53% complete.
    Epoch: 3	16.28% complete.
    Epoch: 3	24.03% complete.
    Epoch: 3	31.78% complete.
    Epoch: 3	39.53% complete.
    Epoch: 3	47.29% complete.
    Epoch: 3	55.04% complete.
    Epoch: 3	62.79% complete.
    Epoch: 3	70.54% complete.
    Epoch: 3	78.29% complete.
    Epoch: 3	86.05% complete.
    Epoch: 3	93.80% complete.
    
    Epoch: 3 	Training Loss: 0.5522 	Validation Loss: 0.6994
    		Training Accuracy: 82.40%	 Validation Accuracy: 78.23%
    Epoch: 4	8.53% complete.
    Epoch: 4	16.28% complete.
    Epoch: 4	24.03% complete.
    Epoch: 4	31.78% complete.
    Epoch: 4	39.53% complete.
    Epoch: 4	47.29% complete.
    Epoch: 4	55.04% complete.
    Epoch: 4	62.79% complete.
    Epoch: 4	70.54% complete.
    Epoch: 4	78.29% complete.
    Epoch: 4	86.05% complete.
    Epoch: 4	93.80% complete.
    
    Epoch: 4 	Training Loss: 0.5624 	Validation Loss: 0.7211
    		Training Accuracy: 82.09%	 Validation Accuracy: 78.18%
    Epoch: 5	8.53% complete.
    Epoch: 5	16.28% complete.
    Epoch: 5	24.03% complete.
    Epoch: 5	31.78% complete.
    Epoch: 5	39.53% complete.
    Epoch: 5	47.29% complete.
    Epoch: 5	55.04% complete.
    Epoch: 5	62.79% complete.
    Epoch: 5	70.54% complete.
    Epoch: 5	78.29% complete.
    Epoch: 5	86.05% complete.
    Epoch: 5	93.80% complete.
    Epoch: 6	8.53% complete.
    Epoch: 6	16.28% complete.
    Epoch: 6	24.03% complete.
    Epoch: 6	31.78% complete.
    Epoch: 6	39.53% complete.
    Epoch: 6	47.29% complete.
    Epoch: 6	55.04% complete.
    Epoch: 6	62.79% complete.
    Epoch: 6	70.54% complete.
    Epoch: 6	78.29% complete.
    Epoch: 6	86.05% complete.
    Epoch: 6	93.80% complete.
    
    Early Stopping Total epochs : 6. Best epoch : 3 with loss : 0.70 and acc : 77.50%



```python
history
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
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>train_acc</th>
      <th>valid_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.616175</td>
      <td>0.718555</td>
      <td>0.810131</td>
      <td>0.782799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.598356</td>
      <td>0.722010</td>
      <td>0.814261</td>
      <td>0.785714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.590265</td>
      <td>0.710458</td>
      <td>0.816266</td>
      <td>0.780369</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.552151</td>
      <td>0.699411</td>
      <td>0.823980</td>
      <td>0.782313</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.562443</td>
      <td>0.721098</td>
      <td>0.820882</td>
      <td>0.781827</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.537756</td>
      <td>0.704002</td>
      <td>0.832847</td>
      <td>0.782799</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.534673</td>
      <td>0.740823</td>
      <td>0.830904</td>
      <td>0.775024</td>
    </tr>
  </tbody>
</table>
</div>

### 4. Model Testing

6번의 epoch 이후에 early stopping을 적용한 결과 validation 대략 78%로 overfitting은 안된것으로 보인다. 이제 test dataset으로 test하자.


```python
def test(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        test_acc = 0
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            test_acc += accuracy.item() * data.size(0)
        
        test_acc = test_acc / len(test_loader.dataset)
        return test_acc
```

훈련 과정에서 저장한 가중치를 이용해서 test를 한다.


```python
model.load_state_dict(torch.load('./dog_inception.pt')['state_dict'])
test_acc = test(model.cuda(), test_loader, criterion)
print(f'test data에서 {100* test_acc:.2f}%의 정확도를 보임!')
```

    test data에서 89.26%의 정확도를 보임!


test data에서 accuracy가 89.26%으로 나쁘지 않은 성능의 모델을 얻어냈다. 이제 각 종 별로 정확도를 도출해보자. 정확도가 낮은 종도 있는데, 이는 우리 사람의 눈으로도 분류하기 비슷할 정도로 비슷하게 생긴 종들이 있기 때문이다.


```python
def evaluate(model, test_loader, criterion):
    classes = []
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            output = model(data)
            
            for pred, true in zip(output, labels):
                _, pred = pred.unsqueeze(0).topk(1)
                correct = pred.eq(true.unsqueeze(0))
                acc_results[i] = correct.cpu()
                classes.append(model.idx_to_class[true.item()][10:])
                i+=1
    
    results = pd.DataFrame({
        'class':classes,
        'results': acc_results
    })
    results = results.groupby(classes).mean()
    return results
```


```python
print(evaluate(model, test_loader, criterion))
```

                                     results
    Afghan_hound                    1.000000
    African_hunting_dog             1.000000
    Airedale                        0.882353
    American_Staffordshire_terrier  0.600000
    Appenzeller                     0.687500
    Australian_terrier              0.800000
    Bedlington_terrier              1.000000
    Bernese_mountain_dog            0.875000
    Blenheim_spaniel                0.937500
    Border_collie                   0.777778
    Border_terrier                  1.000000
    Boston_bull                     0.882353
    Bouvier_des_Flandres            0.928571
    Brabancon_griffon               0.900000
    Brittany_spaniel                0.923077
    Cardigan                        1.000000
    Chesapeake_Bay_retriever        0.941176
    Chihuahua                       0.863636
    Dandie_Dinmont                  1.000000
    Doberman                        0.833333
    English_foxhound                0.782609
    English_setter                  1.000000
    English_springer                1.000000
    EntleBucher                     0.920000
    Eskimo_dog                      0.166667
    French_bulldog                  1.000000
    German_shepherd                 1.000000
    German_short-haired_pointer     0.909091
    Gordon_setter                   0.928571
    Great_Dane                      0.750000
    ...                                  ...
    curly-coated_retriever          0.941176
    dhole                           0.888889
    dingo                           1.000000
    flat-coated_retriever           0.916667
    giant_schnauzer                 0.833333
    golden_retriever                0.866667
    groenendael                     0.923077
    keeshond                        1.000000
    kelpie                          0.777778
    komondor                        1.000000
    kuvasz                          0.866667
    malamute                        0.947368
    malinois                        0.944444
    miniature_pinscher              0.900000
    miniature_poodle                0.588235
    miniature_schnauzer             0.888889
    otterhound                      0.916667
    papillon                        1.000000
    pug                             0.909091
    redbone                         0.952381
    schipperke                      1.000000
    silky_terrier                   0.652174
    soft-coated_wheaten_terrier     0.866667
    standard_poodle                 0.800000
    standard_schnauzer              0.769231
    toy_poodle                      0.846154
    toy_terrier                     0.842105
    vizsla                          0.777778
    whippet                         0.875000
    wire-haired_fox_terrier         0.642857
    
    [120 rows x 1 columns]


출처 및 참고문헌 :

1. https://www.kaggle.com/code/gabrielloye/dogs-inception-pytorch-implementation
2. https://arxiv.org/pdf/1512.00567.pdf
