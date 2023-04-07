---
title : '[CV/Pytorch] 파이토치로 Faster R-CNN 구현하기 📦'
layout: single
toc: true
toc_sticky: true
categories:
  - pytorchOD
---

## Pytorch로 Faster R-CNN 구현하기

이번 글에서는 [How FasterRCNN works and step-by-step PyTorch implementation](https://www.google.com/search?q=How+FasterRCNN+works+and+step-by-step+PyTorch+implementation&oq=How+FasterRCNN+works+and+step-by-step+PyTorch+implementation&aqs=chrome..69i57j69i61l3&client=ubuntu&sourceid=chrome&ie=UTF-8)에 나온 파이토치로 구현한 Faster RCNN 코드를 분석해본다. Faster RCNN에 대한 설명은 이전 글 ([**링크**](https://hamin-chang.github.io/cv-objectdetection/ffrcnn/#faster-rcnn-%EB%85%BC%EB%AC%B8-%EC%9D%BD%EC%96%B4%EB%B3%B4%EA%B8%B0))을 참고하길 바란다. 먼저 사용할 라이브러리들을 로드하고 GPU를 사용하기 위한 코드를 입력한다.




```python
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
  print(DEVICE, torch.cuda.get_device_name(0))
else :
  DEVICE = torch.device('cpu')
  print(DEVICE)
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    cpu


참고한 코드에서는 입력 이미지로 다음의 얼룩말 이미지를 사용했다.


```python
img0 = cv2.imread("/content/drive/MyDrive/zebras.jpg")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
print(img0.shape)
plt.imshow(img0)
plt.show()
```

    (1333, 2000, 3)



    
![1](https://user-images.githubusercontent.com/77332628/212281527-ce922d2e-541d-408c-a1dd-8f98d973722b.png)
    


얼룩말이 있는 박스를 시각화해보자. 


```python
bbox0 = np.array([[223, 782, 623, 1074], [597, 695, 1038, 1050], 
                  [1088, 699, 1452, 1057], [1544, 771, 1914, 1063]]) 
labels = np.array([1, 1, 1, 1]) # 0: background, 1: zebra
```


```python
img0_clone = np.copy(img0)
for i in range(len(bbox0)):
    cv2.rectangle(img0_clone, (bbox0[i][0], bbox0[i][1]), 
                              (bbox0[i][2], bbox0[i][3]),
                 color=(0, 255, 0), thickness=10)
plt.imshow(img0_clone)
plt.show()
```


    
![2](https://user-images.githubusercontent.com/77332628/212281536-90220e9e-1394-4d3c-8710-c6360312eabe.png)
    


편의를 위해서 원본 이미지를 800x800 크기로 resize 해준다. 실제 sub-sampling ratio = 1/16으로 지정해서 feature extractor를 거친 feature map의 크기는 50x50이 된다.


```python
img = cv2.resize(img0, dsize=(800,800), interpolation=cv2.INTER_CUBIC)
plt.figure(figsize=(7,7))
plt.imshow(img)
plt.show()
```


    
![3](https://user-images.githubusercontent.com/77332628/212281538-82d5fac2-4c3b-41c2-a86c-88dcb46bed66.png)


수정한 이미지의 크기에 맞게 얼룩말이 있는 박스를 다시 시각화 해보자. 이 이미지에서의 박스들의 좌표가 ground truth 값이 될 것이다. 


```python
Wratio = 800/img0.shape[1]
Hratio = 800/img0.shape[0]

ratioList = [Wratio,Hratio,Wratio,Hratio]
bbox = []

for box in bbox0:
  box = [int(a*b) for a, b in zip(box,ratioList)]
  bbox.append(box)

bbox = np.array(bbox) 
print('coordinates of ground truth boxes:')
print(bbox)

img_clone = np.copy(img)
for i in range(len(bbox)):
    cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color=(0, 255, 0), thickness=5)
plt.imshow(img_clone)
plt.show()
```

    coordinates of ground truth boxes:
    [[ 89 469 249 644]
     [238 417 415 630]
     [435 419 580 634]
     [617 462 765 637]]



    
![4](https://user-images.githubusercontent.com/77332628/212281543-bf5658f8-5da7-4a48-a008-4e9f3796accf.png)
    


### 1. Feature extraction by pre-trained VGG16
원본 이미지에 대해서 feature extraction을 수행할 사전 훈련된 VGG16 모델을 정의한다. 그 다음 전체 모델에서 sub-sampling ratio에 맞게 50x50 크기가 되는 layer까지만 feature extractor로 사용한다. 이를 위해 원본 이미지와 크기가 같은 800x800 크기의 dummy 배열을 입력해서 50x50 크기의 feature map을 출력하는 layer를 찾는다. 이후 **faster_rcnn_feature_extractor** 변수에 전체 모델에서 해당 layer까지만 저장하고 원본 이미지를 **faster_rcnn_feature_extractor**에 입력해서 50x50x512 크기의 feature map을 얻는다. 


```python
# 사전 훈련된 VGG16 모델 로드
model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
features = list(model.features)

dummy_img = torch.zeros((1,3,800,800)).float() # 입력 이미지와 같은 크기의 dummy 배열

req_features = []
output = dummy_img.clone().to(DEVICE)

for feature in features:
  output = feature(output)

  if output.size()[2] < 800//16:
    break
  req_features.append(feature)
  out_channels = output.size()[1]

# 해당 layer까지만 Sequential model로 변환
faster_rcnn_feature_extractor = nn.Sequential(*req_features)
```

    /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


위에서 정의한 feature extractor에 입력 이미지를 주입해서 Feature extraction을 수행해보자.


```python
transform = transforms.Compose([transforms.ToTensor()])
imgTensor = transform(img).to(DEVICE)
imgTensor = imgTensor.unsqueeze(0)
output_map = faster_rcnn_feature_extractor(imgTensor) # 출력값

print(output_map.size())
```

    torch.Size([1, 512, 50, 50])


출력한 50x50x512 feature maps의 첫 5개의 채널을 시각화해보자.


```python
imgArray = output_map.data.cpu().numpy().squeeze(0)
fig = plt.figure(figsize=(12,4))
figNo = 1 

for i in range(5):
  fig.add_subplot(1,5,figNo)
  plt.imshow(imgArray[i], cmap='gray')
  figNo += 1

plt.show()
```


    
![5](https://user-images.githubusercontent.com/77332628/212281547-c0754078-005c-4882-a576-d49ec359b032.png)
    


### 2. Anchor Generation Layer
Anchor Generation Layer에서는 anchor box를 생성하는 역할을 합니다. 먼저 이해를 돕기 위해 입력 이미지에 anchor의 중심 좌표를 시각화해보자.


```python
'''이미지 크기는 800x800이고 sub-sampling rate = 1/16이기 때문에
총 50x50 = 2500개의 anchor이 만들어지고 anchor 하나당 9개의 anchor boxes가
만들어지기 때문에 총 9x2500 = 22500개의 anchor boxes 생성'''

feature_size = 800 // 16
ctr_x = np.arange(16, (feature_size+1) * 16, 16 ) 
ctr_y = np.arange(16, (feature_size+1) * 16, 16 )
print(ctr_x)
```

    [ 16  32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272 288
     304 320 336 352 368 384 400 416 432 448 464 480 496 512 528 544 560 576
     592 608 624 640 656 672 688 704 720 736 752 768 784 800]



```python
# anchor의 중심 좌표 구하기

index = 0
ctr = np.zeros((2500,2))

for i in range(len(ctr_x)):
  for j in range(len(ctr_y)):
    ctr[index, 1] = ctr_x[i] - 8
    ctr[index, 0] = ctr_y[j] - 8
    index += 1
```

Anchor의 중심 좌표를 빨간색 점으로 이미지에 표시해보자.


```python
img_clone2 = np.copy(img)
ctr_int = ctr.astype('int32')

plt.figure(figsize=(7,7))
for i in range(ctr.shape[0]):
  cv2.circle(img_clone2, (ctr_int[i][0], ctr_int[i][1]),
             radius = 1 , color = (255,0,0), thickness=3)
plt.imshow(img_clone2)
plt.show()
```


    
![6](https://user-images.githubusercontent.com/77332628/212281552-09b92547-530f-4cf8-8584-7fff42656449.png)
    


이제 본격적인 Anchor generation layer를 구현할텐데, 이를 위해서 16x16 간격의 grid마다 anchor를 생선하고, anchor를 기준으로 서로 다른 크기와 비율을 가지는 9개의 anchor box를 생성한다. anchor_boxes 변수에 전체 anchor box의 좌표 (x1,y1,x2,y2)를 저장한다. 


```python
ratios = [0.5,1,2]
scales = [8,16,32]
sub_sample = 16

anchor_boxes = np.zeros(((feature_size * feature_size * 9),4))
index = 0
for c in ctr :  # per anchors
  ctr_y , ctr_x = c 
  for i in range(len(ratios)): # per ratios
    for j in range(len(scales)): # per scales

      # anchor box의 height, width
      h = sub_sample * scales[j] * np.sqrt(ratios[i])
      w = sub_sample * scales[j] * np.sqrt(1./ratios[i])

      # anchor boxes 변수에 전체 anchor box의 좌표 저장
      anchor_boxes[index,1] = ctr_y - h/2
      anchor_boxes[index, 0] = ctr_x - w / 2.
      anchor_boxes[index, 3] = ctr_y + h / 2.
      anchor_boxes[index, 2] = ctr_x + w / 2.
      index += 1

```

생성한 anchor box들을 원본 이미지에 시각화 해볼 건데, 이때 이미지의 경계를 벗어나는 anchor box들도 있기 때문에 원본 이미지에 padding을 추가한다.


```python
# padding 추가
img_clone3 = np.copy(img)
img_clone4 = cv2.copyMakeBorder(img_clone3,400,400,400,400,cv2.BORDER_CONSTANT,value=(255,255,255))
img_clone5 = np.copy(img_clone4)

# 모든 anchor boxes 그리기
for i in range(len(anchor_boxes)):
  x1 = int(anchor_boxes[i][0])
  y1 = int(anchor_boxes[i][1])
  x2 = int(anchor_boxes[i][2])
  y2 = int(anchor_boxes[i][3])

  cv2.rectangle(img_clone5, (x1+400,y1+400),(x2+400,y2+400),color=(255,0,0), thickness=3)

plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(img_clone4)
plt.subplot(122), plt.imshow(img_clone5)
plt.show()
```


    
![7](https://user-images.githubusercontent.com/77332628/212281559-44b3f290-81a3-4097-a592-8fad861bcf7d.png)
    


### 3. Anchor Target Layer
Anchor Target Layer에서는 RPN을 학습시키기 위해서 적절한 anchor box를 선택하는 작업을 수행한다. 먼저 이미지 경계 내부에 있는 anchor box 만을 선택하자.


```python
# anchor box의 좌표가 (x1,y1) >= 0 이고 (x2,y2) <= 800인 경우만 선택

index_inside = np.where((anchor_boxes[:,0] >= 0)&
                        (anchor_boxes[:,1] >= 0)&
                        (anchor_boxes[:,2] <= 800)&
                        (anchor_boxes[:,3] <= 800))[0]

valid_anchor_boxes = anchor_boxes[index_inside]
print('number of valid anchor boxes : ',valid_anchor_boxes.shape[0])
```

    number of valid anchor boxes :  8940


그 다음 전체 anchor box에 대해서 ground truth box와 IoU값을 구한다. 위의 코드에서 구했듯이 유효한 anchor box는 8940개이고 ground truth 값은 4개이기 때문에 한 8940개의 [IoU with gt box1,IoU with gt box2,IoU with gt box3,IoU with gt box4]의 값이 나와야 한다.


```python
ious = np.empty((len(valid_anchor_boxes),4),dtype=np.float32)
ious.fill(0) # 일단 더미 배열 만들어놓기

# anchor box 영역 구하기
for i, anchor_box in enumerate(valid_anchor_boxes):
  xa1, ya1, xa2, ya2 = anchor_box
  anchor_area = (xa2-xa1) * (ya2-ya1)
  # gt box 영역 구하기
  for j, gt_box in enumerate(bbox):
    xb1, yb1, xb2, yb2 = gt_box
    box_area = (xb2-xb1) * (yb2 - yb1)

    # 겹치는 영역의 좌표
    inter_x1 = max([xb1,xa1])
    inter_y1 = max([yb1,ya1])
    inter_x2 = min([xb2,xa2])
    inter_y2 = min([yb2,ya2])

    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
      inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
      iou = inter_area / (anchor_area + box_area - inter_area)
    
    else : 
      iou = 0
    
    ious[i,j] = iou

print(ious.shape)
print(ious[8930:8940,:]) # 마지막 10개 예시로 출력
```

    (8940, 4)
    [[0.         0.         0.         0.37780452]
     [0.         0.         0.         0.33321926]
     [0.         0.         0.         0.29009855]
     [0.         0.         0.         0.24967977]
     [0.         0.         0.         0.2117167 ]
     [0.         0.         0.         0.17599213]
     [0.         0.         0.         0.14231375]
     [0.         0.         0.         0.11051063]
     [0.         0.         0.         0.08043041]
     [0.         0.         0.         0.05193678]]


이제 각 gt box들과 최대 iou를 가지는 anchor box가 무엇인지 알아보자.


```python
gt_argmax_ious = ious.argmax(axis=0)

gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print(gt_argmax_ious)
```

    [1008 1013 1018 1226 1232 1238 2862 2869 2876 3108 3115 3122 3336 3343
     3350 3354 3357 3361 3364 3368 3371 3377 3383 3389 3600 3607 3614 3846
     3853 3860 5935 5942 6164 6171 6178 6181 6185 6188 6192 6198 6427 6434
     8699 8703 8707]


각 anchor box마다의 어떤 gt box와의 iou가 가장 높은지 알아보고, 가장 높은 iou 값을 출력해보자.


```python
argmax_ious = ious.argmax(axis=1)
print(argmax_ious)

max_ious = ious[np.arange(len(index_inside)), argmax_ious]
print(max_ious)
```

    [0 0 0 ... 3 3 3]
    [0.         0.         0.         ... 0.11051063 0.08043041 0.05193678]


그 다음 각 gt box와 IoU가 가장 큰 anchor box와 IoU 값이 0.7 이상인 anchor box는 positive sample로, IoU 값이 0.3 미만인 anchor box는 negative samplefh 저장한다. 먼저 더미 label 배열을 만들고 label 변수에 positive일 경우 1, negative일 경우 0으로 저장한다.


```python
# 더미 label 배열 만들기
label = np.empty((len(index_inside),), dtype=np.int32)
label.fill(-1)

# positive, negative sample 기준 IoU
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1
label[max_ious < neg_iou_threshold] = 0
```

이제 mini-batch를 구성할건데, 크기는 256, positive/negative sample의 비율은 1:1로 구성한다. 만약 positive sample이 128개 이상이면 남는 postive sample에 해당하는 label 변수는 -1로 지정한다. negative sample에 대해서도 마찬가지로 수행하지만, 일반적으로 positive samle의 수가 128개 미만일 경우, 부족한 만큼의 sample을 negative sample에서 추출한다.


```python
n_sample = 256
pos_ratio = 0.5
n_pos = pos_ratio * n_sample

pos_index = np.where(label==1)[0]

if len(pos_index) > n_pos :
  disable_index = np.random.choice(pos_index,
                                   size = (len(pos_index) - n_pos),
                                   replace = False)
  label[disable_index] = -1

n_neg = n_sample * np.sum(label == 1)
neg_index = np.where(label == 0)[0]

if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, 
                                    size = (len(neg_index) - n_neg), 
                                    replace = False)
    label[disable_index] = -1
```

그 다음은 각 valid anchor box가 최대 IoU를 가지는 gt objection의 좌표를 알아내고, valid anchor box의 좌표를 [x1,y1,x2,y2] 형식으로 변환한다.


```python
# convert the format of valid anchor boxes [x1, y1, x2, y2]

# For each valid anchor box, find the groundtruth object which has max_iou 
max_iou_bbox = bbox[argmax_ious]
print(max_iou_bbox)

height = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
width = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
ctr_y = valid_anchor_boxes[:, 1] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 0] + 0.5 * width

base_height = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_width = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_ctr_y = max_iou_bbox[:, 1] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 0] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()
print(anchor_locs)
```

    [[ 89 469 249 644]
     [ 89 469 249 644]
     [ 89 469 249 644]
     ...
     [617 462 765 637]
     [617 462 765 637]
     [617 462 765 637]]
    [[ 1.24848541  2.49973296  0.56971714 -0.03381788]
     [ 1.24848541  2.41134461  0.56971714 -0.03381788]
     [ 1.24848541  2.32295626  0.56971714 -0.03381788]
     ...
     [-0.5855728  -0.63252911  0.4917556  -0.03381788]
     [-0.5855728  -0.72091746  0.4917556  -0.03381788]
     [-0.5855728  -0.80930581  0.4917556  -0.03381788]]



```python
# First set the label=-1 and locations=0 of the 22500 anchor boxes, 
# and then fill in the locations and labels of the 8940 valid anchor boxes
# NOTICE: For each training epoch, we randomly select 128 positive + 128 negative 
# from 8940 valid anchor boxes, and the others are marked with -1

anchor_labels = np.empty((len(anchor_boxes),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label
print(anchor_labels.shape)
print(anchor_labels)

anchor_locations = np.empty((len(anchor_boxes),) + anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs
print(anchor_locations.shape)
print(anchor_locations[:10, :])
```

    (22500,)
    [-1 -1 -1 ... -1 -1 -1]
    (22500, 4)
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]


### 4. RPN (Region Proposal Network)

![11111](https://user-images.githubusercontent.com/77332628/212281576-d088d853-2f95-4215-b473-5ce0addb5626.png)

이제 RPN을 정의할건데, 1.Feature extraction을 통해 생성된 feature map에 3x3 conv 연산을 적용하는 layer를 정의하고, 1x1 conv 연산을 적용해서 9x4(anchor box 종류 수 x bounding box 좌표)개의 channel을 가지는 feature map을 반환하는 Bounding box regressor를 정의한다. 그리고 1x1 conv 연산을 적용해서 9x2(anchor box 수 x object 존재 여부)개의 channel을 가지는 feature map을 반환하는 Classifier를 정의한다.


```python
in_channels = 512
mid_channels = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(DEVICE)
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# bounding box regressor
reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0).to(DEVICE)
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classifier(object or not)
cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0).to(DEVICE)
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



이제 모델을 구축할건데, feature extraction을 통해서 얻은 feature map을 3x3 conv layer에 입력한다. 이를 통해서 얻은 50x50x512 크기의 feature map을 BBR, Classifier에 주입해서 bounding box coefficients (pred_anchor_locs 변수)와 objectness score (pred_cls_score 변수)를 얻는다. 이를 target 값과 비교하기 위해서 적절하게 resize 해준다.


```python
x = conv1(output_map.to(DEVICE)) # output_map : feature extraction에서 얻은 feature map
pred_anchor_locs = reg_layer(x) # BBR 출력
pred_cls_scores = cls_layer(x) # Classifier 출력

print(pred_anchor_locs.shape, pred_cls_scores.shape)

```

    torch.Size([1, 36, 50, 50]) torch.Size([1, 18, 50, 50])


위 코드의 출력값의 형식을 바꿔주자.
* BBR의 위치 : [1,36,50,50] => [1,22500(=50x50x9),4] (dy,dx,dh,dw)
* Classification : [1,18,50,50] => [1,22500,2] (1,0)



```python
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
print(pred_cls_scores.shape)

objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)

pred_cls_scores = pred_cls_scores.view(1, -1, 2)
print(pred_cls_scores.shape)
```

    torch.Size([1, 22500, 4])
    torch.Size([1, 50, 50, 18])
    torch.Size([1, 22500, 2])


이제 손실값을 구하기 위해 BBR과 Classification에 대한 ground truth 값을 정의하자.


```python
# RPN의 출력값
rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]

# ground truth 값
gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_labels)

print(rpn_loc.shape)
print(rpn_score.shape)
print(gt_rpn_loc.shape)
print(gt_rpn_score.shape)
```

    torch.Size([22500, 4])
    torch.Size([22500, 2])
    torch.Size([22500, 4])
    torch.Size([22500])


#### 4.1 Multi-task Loss
이제 RPN의 손실값을 계산하는 Multi-taks loss를 구현해보자. Classification loss는 cross entropy loss를 활용해서 구한다. BBR loss는 오직 positive에 해당하는 sample에 대해서만 loss를 계산하므로, positive/negative 여부를 저장하는 배열인 mask를 생성하고 이를 활용해서 Smooth L1 loss를 계산한다. Classification loss와 BBR loss 사이를 조정하는 balancing parameter λ = 10으로 지정하고 두 loss를 더해서 multi-task loss를 구현한다.


```python
rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long().to(DEVICE), ignore_index = -1)
print(rpn_cls_loss)
```

    tensor(0.6945, grad_fn=<NllLossBackward0>)



```python
# only positive samples
pos = gt_rpn_score > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)

# positive labels를 갖는 bounding box만 사용
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
print(rpn_loc_loss.sum())
```

    tensor(5.5637, dtype=torch.float64, grad_fn=<SumBackward0>)



```python
# combine rpn_cls_loss and rpn_reg_loss

rpn_lambda = 10
N_reg = (gt_rpn_score > 0).float().sum()
rpn_loc_loss = rpn_loc_loss.sum() / N_reg
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print(rpn_loss)
```

    tensor(1.9309, dtype=torch.float64, grad_fn=<AddBackward0>)


### 5. Proposal Layer
Proposal Layer에서는 Anchor generation layer에서 생성된 anchor boxes와 RPN에서 반환된 class scores와 bounding box regressor를 사용해서 region proposals를 추출하는 작업을 수행한다. 먼저 score 변수에 저장된 objectness score를 내림차순으로 정렬한 후 objectness score 상위 N(n_train_pre_nms=1200)개의 anchor box에 대하여 Non Maximum Suppression 알고리즘을 수행한다. 남은 anchor box 중 상위 N(n_train_post_nms=2000)개의 region proposals를 모델 학습에 사용한다. 

먼저 anchor box를 RPN에서 구한 값들을 이용해서 변환시켜서 RoI를 구하자.


```python
nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# anchor box를 [x1,y1,x2,y2]에서 [ctrx,ctry,w,h]로 변환
anc_height = anchor_boxes[:,3] - anchor_boxes[:,1]
anc_width = anchor_boxes[:,2] - anchor_boxes[:,0]
anc_ctrx = anchor_boxes[:,1] + 0.5 * anc_height
anc_ctry = anchor_boxes[:,0] + 0.5 * anc_width

# 22500개의 anchor boxes location과 objectness score를 numpy로 변환
pred_anchor_locs_numpy = pred_anchor_locs[0].cpu().data.numpy()
objectness_score_numpy = objectness_score[0].cpu().data.numpy()

dy = pred_anchor_locs_numpy[:,1::4]
dx = pred_anchor_locs_numpy[:,0::4]
dh = pred_anchor_locs_numpy[:,3::4]
dw = pred_anchor_locs_numpy[:,2::4]

# BBR을 이용해서 RoI의 ctr_x, ctr_y, h, w 구하기
ctr_x = dx * anc_height[:, np.newaxis] + anc_ctrx[:, np.newaxis]
ctr_y = dy * anc_width[:, np.newaxis] + anc_ctry[:, np.newaxis]
h = np.exp(dh) * anc_height[:, np.newaxis]
w = np.exp(dw) * anc_width[:, np.newaxis]

# RoI의 좌표
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype = anchor_locs.dtype)
roi[:, 0::4] = ctr_x - 0.5 * w 
roi[:, 1::4] = ctr_y - 0.5 * h
roi[:, 2::4] = ctr_x + 0.5 * w
roi[:, 3::4] = ctr_y + 0.5 * h
```

그리고 구한 RoI들을 이미지 크기에 맞춰서 최소 최댓값을 제한해주자.


```python
img_size = (800,800)
roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)],0,img_size[0]) # x1,x2값
roi[:,slice(1,4,2)] = np.clip(roi[:,slice(1,4,2)],0,img_size[1]) # y1,y2값

# height나 width가 min_size = 16 보다 작은 predicted box는 버린다.
hs = roi[:,3] - roi[:,1]
ws = roi[:,2] - roi[:,0]

keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep,:]
score = objectness_score_numpy[keep]
print(keep.shape, roi.shape, score.shape)
```

    (22490,) (22490, 4) (22490,)


상위 12000개의 objectness score인 anchor boxes만 사용한다.


```python
# 내림차순
order = score.ravel().argsort()[::-1]

order = order[:n_train_pre_nms] # 상위 12000개
roi = roi[order, :]
```

이제 2000개의 bounding box를 선택하는 Non maximun suppression을 수행하자.


```python
x1 = roi[:,0]
y1 = roi[:,1]
x2 = roi[:,2]
y2 = roi[:,3]

areas = (x2-x1 +1) * (y2-y1 +1)
order = order.argsort()[::-1]
keep = []

while (order.size>0):
  i = order[0]
  keep.append(i)

  xx1 = np.maximum(x1[i], x1[order[1:]])
  yy1 = np.maximum(y1[i], y1[order[1:]])
  xx2 = np.minimum(x2[i], x2[order[1:]])
  yy2 = np.minimum(y2[i], y2[order[1:]])

  w = np.maximum(0.0, xx2 - xx1 + 1)
  h = np.maximum(0.0, yy2 - yy1 + 1)  

  inter = w * h
  ovr = inter / (areas[i] + areas[order[1:]] - inter)
  inds = np.where(ovr <= nms_thresh)[0]
  order = order[inds + 1]

keep = keep[:n_train_post_nms]
roi = roi[keep]
print(len(keep), roi.shape)
```

    2000 (2000, 4)


### 6. Proposal Target Layer
Proposal target layer의 목표는 proposal layer에서 나온 region proposals 중에서 Fast RCNN 모델을 학습시키기 위한 유용한 sample을 선택하는 것이다. 학습을 위해서 128개의 sample을 mini-batch로 구성한다. 이때 Proposal layer에서 얻은 anchor box 중 gt box와의 IoU 값이 0.5 이상인 box를 positive sample로, 0.5 미만인 box를 negative sample로 지정한다. 전체 mini-batch sample 중 25%, 즉 32개가 positive sample이 되도록 구성한다. positive sample이 32개 미만인 경우에는 부족한 sample을 negative sample에서 구한다.




```python
n_sample = 128 
pos_ratio = 0.25 
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5 
neg_iou_thresh_lo = 0.0
```

먼저 IoU를 계산한다.


```python
ious = np.empty((len(roi), bbox.shape[0]), dtype = np.float32)
ious.fill(0)

for num1, i in enumerate(roi):
  ya1, xa1, ya2, xa2 = i
  anchor_area = (ya2-ya1) * (xa2-xa1)

  for num2 , j in enumerate(bbox):
    yb1, xb1, yb2, xb2 = j
    box_area = (yb2-yb1) * (xb2-xb1)
    inter_x1 = max([xb1,xa1])
    inter_y1 = max([yb1,ya1])
    inter_x2 = min([xb2,xa2])
    inter_y2 = min([yb2,ya2])

    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
      inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
      iou = inter_area / (anchor_area + box_area - inter_area)
    else :
      iou = 0
    ious[num1, num2] = iou

print(ious.shape)
```

    (2000, 4)


그리고 각 region proposal에 어떤 gt box가 높은 IoU를 가지는지 찾고 이를 사용해서 각 region proposal에 레이블을 달아준다.


```python
gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)

print(gt_assignment)

gt_roi_label = labels[gt_assignment]
print(gt_roi_label)
```

    [0 3 0 ... 0 0 0]
    [1 1 1 ... 1 1 1]


이제 positive sample과 negative sample을 분류해보자.


```python
pos_roi_per_image = 32
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))

if pos_index.size > 0:
  pos_index = np.random.choice(
      pos_index, size=pos_roi_per_this_image, replace=False)
  
print(pos_roi_per_this_image)
print(pos_index)
```

    32
    [ 712  662  821  851  917  805  656  803  701  964  704  732  774  740
      779  822 1151  651  713  691  784  847  752  690  900  857  710  693
      620  836  835  716]



```python
neg_index = np.where((max_iou < neg_iou_thresh_hi) &
                     (max_iou >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

if neg_index.size > 0:
  neg_index = np.random.choice(
    neg_index, size = neg_roi_per_this_image, replace=False)
  
print(neg_roi_per_this_image)
print(neg_index)
```

    96
    [1858 1782 1339  388 1503  589 1090 1046  738  441 1053 1067 1989 1289
     1588 1110 1777  433  954 1292 1960 1159 1123  166  966  175 1387   38
     1196 1486  161 1526 1412 1183 1555  322  177 1611 1126 1332 1710 1089
     1810 1504 1038  543  977  658 1263  485  190 1482 1621 1803  168  508
     1180  232 1568 1887 1644  213  812   96 1282 1811 1661  898 1417 1351
       52 1873 1235 1169 1331  689 1688  367 1844 1634  444  121 1466  965
     1115  220 1998 1484  563  807 1676 1897 1264  141  275 1522]


이제 positive sample과 negative sample을 원본 이미지에 각각 시각화해보자. 


```python
# display RoI samples with positive

img_clone = np.copy(img)

for i in range(pos_roi_per_this_image):
  x1, y1, x2, y2 = roi[pos_index[i]].astype(int)
  cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255,255,255),
                thickness=3)
  
for i in range(len(bbox)):
  cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), 
                color = (0, 255, 0), thickness=3)

plt.imshow(img_clone)
plt.show()
```


    
![8](https://user-images.githubusercontent.com/77332628/212281563-ad109df8-5135-4b86-a1e8-349c71ace133.png)
    



```python
# display RoI samples with negative

img_clone = np.copy(img)

plt.figure(figsize=(9, 6))

for i in range(neg_roi_per_this_image):
  x1, y1, x2, y2 = roi[neg_index[i]].astype(int)
  cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255, 255, 255),
                thickness=3)
  
for i in range(len(bbox)):
  cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), 
                color = (0, 255, 0), thickness=3)
  
plt.imshow(img_clone)
plt.show()
```


    
![9](https://user-images.githubusercontent.com/77332628/212281566-54a50258-f073-4346-a785-a34394dc3732.png)
    


이제 마지막으로 positive와 negative sample을 합쳐서 미니 배치를 만들자.


```python
keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0 # negative label은 0으로 만든다.
sample_roi = roi[keep_index]
print(sample_roi.shape)
```

    (128, 4)



```python
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]

width = sample_roi[:, 2] - sample_roi[:, 0]
height = sample_roi[:, 3] - sample_roi[:, 1]
ctr_x = sample_roi[:, 0] + 0.5 * width
ctr_y = sample_roi[:, 1] + 0.5 * height

base_width = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_height = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_x = bbox_for_sampled_roi[:, 0] + 0.5 * base_width
base_ctr_y = bbox_for_sampled_roi[:, 1] + 0.5 * base_height 

# transform anchor boxes

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dx = (base_ctr_x - ctr_x) / width
dy = (base_ctr_y - ctr_y) / height
dw = np.log(base_width / width)
dh = np.log(base_height / height)

gt_roi_locs = np.vstack((dx, dy, dw, dh)).transpose()
print(gt_roi_locs.shape)
```

    (128, 4)


### 7. RoI Pooling
Feature extractor를 통해서 얻은 feature map과 Proposal Target Layer에서 얻은 region proposals를 이용해서 RoI Pooling을 수행하는데, output feature map의 크기가 7x7이 되도록 한다.

먼저 labels와 bbox 좌표를 합치자.


```python
rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)
```

    torch.Size([128, 4]) torch.Size([128])



```python
indices_rois = torch.cat([roi_indices[:,None],rois],dim=1)
xy_indices_rois = indices_rois[:,[0,2,1,4,3]]
indices_rois = xy_indices_rois.contiguous()
print(xy_indices_rois.shape)
```

    torch.Size([128, 5])


그리고 RoI Pooling을 구현해보자.


```python
size = (7,7)
adaptive_max_pool = nn.AdaptiveMaxPool2d(size[0],size[1])

output = []
rois = indices_rois.data.float()
rois[:, 1:].mul_(1/16.0) # sub-sampling ratio
rois = rois.long()
num_rois = rois.size(0)

for i in range(num_rois):
  roi = rois[i]
  im_idx = roi[0]
  im = output_map.narrow(0, im_idx, 1)[..., roi[1]:(roi[3]+1), roi[2]:(roi[4]+1)]
  tmp = adaptive_max_pool(im)
  output.append(tmp[0])

output = torch.cat(output, 0)

print(output.size())
```

    torch.Size([128, 512, 7, 7])


### 8. Fast R-CNN

![2222](https://user-images.githubusercontent.com/77332628/212281573-4a37c71a-cb0e-4da7-bef5-077bd264203d.png)


마지막 단계로 RoI Pooling을 통해서 얻은 7x7 크기의 feature map을 받을 fc layer를 정의한다. 그리고 class 별로 bounding box coefficients를 예측하는 Bounding Box Regreesor와 class score를 예측하는 Classifier를 정의한다.


```python
# feed forward layer에 전달할 수 있도록 output을 reshape해준다.
k = output.view(output.size(0),-1)

# 7x7x512 크기의 feature map이 fc layer에 전달된다.
roi_head_classifier = nn.Sequential(*[nn.Linear(25088,4096), nn.Linear(4096,4096)])
cls_loc = nn.Linear(4096, 2*4) # 1 class, 1 배경, 4 좌표
cls_loc.weight.data.normal_(0,0.01)
cls_loc.bias.data.zero_()

score = nn.Linear(4096,2) # 1 class, 1 배경

k = roi_head_classifier(k.to(DEVICE)) 
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)

```

다음으로 Fast RCNN에서의 classification loss와 Regression loss를 구하고 결합해서 Multi-task loss를 정의한다.


```python
# 예측값
print(roi_cls_loc.shape)
print(roi_cls_score.shape)

# 정답값
print(gt_roi_locs.shape)
print(gt_roi_labels.shape)

gt_roi_labels
```

    torch.Size([128, 8])
    torch.Size([128, 2])
    (128, 4)
    (128,)





    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
# Classification Loss

# Converting ground truth to torch variable
gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
print(gt_roi_loc.shape, gt_roi_label.shape)

#Classification loss
roi_cls_loss = F.cross_entropy(roi_cls_score.cpu(), gt_roi_label.cpu(), ignore_index=-1)
```

    torch.Size([128, 4]) torch.Size([128])



```python
# regression loss

n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]

# for regression we use smooth l1 loss as defined in the Fast R-CNN paper
pos = gt_roi_label > 0
mask = pos.unsqueeze(1).expand_as(roi_loc)

# take those bounding boxes which have positive labels
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_loc[mask].view(-1, 4)

x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
print(roi_loc_loss.sum())
```

    tensor(6.7210, dtype=torch.float64, grad_fn=<SumBackward0>)



```python
# Multi-task loss
roi_lambda = 10.
roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
print(roi_loss)

total_loss = rpn_loss + roi_loss
print(total_loss)
```

    tensor([[1.0936, 0.9071, 3.3268, 0.7143],
            [1.1238, 0.7561, 1.3800, 0.7753],
            [0.7255, 0.7369, 1.5322, 1.0111],
            [0.7884, 1.1029, 1.7753, 0.7045],
            [0.7651, 1.3447, 2.1244, 0.8515],
            [1.2455, 0.6956, 0.7916, 1.5698],
            [0.7303, 2.9759, 0.9521, 1.9188],
            [0.8895, 1.2778, 1.2332, 1.5378],
            [1.3119, 1.4879, 0.7367, 1.3184],
            [0.6888, 0.9747, 0.7605, 2.4373],
            [0.8053, 1.0531, 0.7241, 1.7428],
            [1.6073, 1.8577, 0.9956, 0.6962],
            [0.7959, 0.7438, 1.0073, 3.1008],
            [0.7297, 0.7043, 1.9854, 0.7268],
            [1.9234, 1.1130, 0.7196, 0.7216],
            [1.0215, 0.9468, 1.7787, 0.8716],
            [1.1643, 0.7128, 1.1576, 1.2314],
            [0.8628, 0.8774, 0.6940, 1.9987],
            [0.7526, 0.7147, 1.5985, 0.6927],
            [0.7461, 0.9348, 1.0082, 1.9449],
            [0.7094, 0.8767, 1.0298, 1.3036],
            [0.9790, 0.8008, 2.4844, 0.8239],
            [0.6896, 0.7068, 3.3081, 0.6885],
            [0.7091, 0.8563, 0.7847, 1.6985],
            [0.6904, 0.7272, 3.3711, 0.8233],
            [0.9068, 0.6919, 0.7404, 0.8501],
            [0.8393, 0.7360, 0.7358, 0.7666],
            [0.8143, 3.5515, 1.8522, 1.0700],
            [0.7370, 2.3878, 1.1776, 2.2519],
            [1.0304, 0.8789, 0.7123, 0.7005],
            [1.1016, 0.7949, 2.5251, 4.2220],
            [1.1834, 0.7261, 1.5839, 1.1725]], dtype=torch.float64,
           grad_fn=<AddBackward0>)
    tensor([[3.0245, 2.8380, 5.2578, 2.6453],
            [3.0547, 2.6870, 3.3109, 2.7063],
            [2.6565, 2.6678, 3.4631, 2.9420],
            [2.7194, 3.0338, 3.7062, 2.6354],
            [2.6960, 3.2756, 4.0553, 2.7824],
            [3.1764, 2.6265, 2.7225, 3.5007],
            [2.6612, 4.9068, 2.8831, 3.8497],
            [2.8204, 3.2087, 3.1641, 3.4687],
            [3.2428, 3.4188, 2.6676, 3.2493],
            [2.6197, 2.9056, 2.6915, 4.3682],
            [2.7362, 2.9840, 2.6550, 3.6738],
            [3.5382, 3.7887, 2.9266, 2.6272],
            [2.7268, 2.6748, 2.9383, 5.0317],
            [2.6606, 2.6352, 3.9163, 2.6577],
            [3.8543, 3.0439, 2.6505, 2.6525],
            [2.9524, 2.8777, 3.7096, 2.8025],
            [3.0952, 2.6438, 3.0886, 3.1623],
            [2.7937, 2.8083, 2.6250, 3.9296],
            [2.6835, 2.6456, 3.5294, 2.6236],
            [2.6771, 2.8657, 2.9391, 3.8758],
            [2.6403, 2.8076, 2.9607, 3.2345],
            [2.9099, 2.7318, 4.4153, 2.7548],
            [2.6205, 2.6377, 5.2390, 2.6195],
            [2.6400, 2.7872, 2.7156, 3.6294],
            [2.6213, 2.6581, 5.3020, 2.7542],
            [2.8377, 2.6228, 2.6714, 2.7810],
            [2.7702, 2.6670, 2.6668, 2.6976],
            [2.7452, 5.4824, 3.7831, 3.0009],
            [2.6679, 4.3187, 3.1085, 4.1828],
            [2.9613, 2.8099, 2.6432, 2.6314],
            [3.0325, 2.7259, 4.4561, 6.1529],
            [3.1143, 2.6570, 3.5148, 3.1034]], dtype=torch.float64,
           grad_fn=<AddBackward0>)


이렇게 해서 Faster RCNN을 실제로 구현하는 방법에 대해 알아봤다. 굉장히 코드도 길고, 설명도 있었기 때문에 긴 글이 되었지만 그래도 Faster RCNN이 동작하는 방법에 대해서 조금은 자세하게 알게 된것 같아서 의미 있는 실습이었다.

출처

* https://www.google.com/search?q=How+FasterRCNN+works+and+step-by-step+PyTorch+implementation&oq=How+FasterRCNN+works+and+step-by-step+PyTorch+implementation&aqs=chrome..69i57j69i61l3&client=ubuntu&sourceid=chrome&ie=UTF-8 원본 코드 영상

* https://herbwood.tistory.com/11 코드의 설명 참고
