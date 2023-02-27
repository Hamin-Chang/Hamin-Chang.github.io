---
title: '[Kaggle] Pneumonia Detection - Mask R-CNN을 이용해서 폐렴 탐지하기 🤢'
toc: true
toc_sticky: true
categories:
  - kaggle-objectdetection
---

## Mask R-CNN으로 Instance segmantation 구현해보기

이번 글에서는 Mask R-CNN 모델을 이용해서 Airbus Ship Detection challenge를 풀어볼 것이다. Mask R-CNN을 처음부터 끝까지 구현하는 것이 아니라 모델을 불러와서 문제를 풀건데, 이때 COCO dataset이 사용된 pre-trained weights를 사용할 것이다.

HENRIQUE MENDONÇA의 노트북 (https://www.kaggle.com/code/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155)을 참고해서 코드를 작성했다.

### 0. 사전 준비


```python
import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold
```


```python
DATA_DIR = '/kaggle/input'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'
```

이제 Matterport의 Mask R-CNN 모델을 [<U>github</U>](https://github.com/matterport/Mask_RCNN/tree/master/mrcnn)에서 가져와서 설치할건데, 구조에 대한 자세한 설명은 [<U>파이토치로 구현한 Mask R-CNN</U>](https://hamin-chang.github.io/pytorchcv/maskrcnnpytorch/)을 참고하면 된다.


```python
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN')
```

    Cloning into 'Mask_RCNN'...
    remote: Enumerating objects: 956, done.[K
    remote: Total 956 (delta 0), reused 0 (delta 0), pack-reused 956[K
    Receiving objects: 100% (956/956), 137.67 MiB | 37.39 MiB/s, done.
    Resolving deltas: 100% (558/558), done.



```python
# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
```

    Using TensorFlow backend.



```python
train_dicom_dir = os.path.join(DATA_DIR, 'stage_2_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')
```

COCO pre-trained weights 다운로드


```python
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"
```

    -rw-r--r-- 1 root root 246M Dec  6  2021 mask_rcnn_coco.h5



```python
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/'+'*.dcm') # list of dicom image path and filenames
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp : [] for fp in image_fps} # dictionary of annotations keyed by filenames
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations
```

먼저 base Config보다 더 좋은 성능을 낼 수 있는 DetectorConfig를 다시 정의해준다. 


```python
class DetectorConfig(Config):
    NAME = 'pneumonia'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2 # background + 1 pneumonia class
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01
    
    STEPS_PER_EPOCH = 200

config = DetectorConfig()

# 수정한 config를 살펴보자.
config.display()
```

    
    Configurations:
    BACKBONE                       resnet50
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     8
    BBOX_STD_DEV                   [0.1 0.1 0.2 0.2]
    COMPUTE_BACKBONE_SHAPE         None
    DETECTION_MAX_INSTANCES        3
    DETECTION_MIN_CONFIDENCE       0.78
    DETECTION_NMS_THRESHOLD        0.01
    FPN_CLASSIF_FC_LAYERS_SIZE     1024
    GPU_COUNT                      1
    GRADIENT_CLIP_NORM             5.0
    IMAGES_PER_GPU                 8
    IMAGE_CHANNEL_COUNT            3
    IMAGE_MAX_DIM                  256
    IMAGE_META_SIZE                14
    IMAGE_MIN_DIM                  256
    IMAGE_MIN_SCALE                0
    IMAGE_RESIZE_MODE              square
    IMAGE_SHAPE                    [256 256   3]
    LEARNING_MOMENTUM              0.9
    LEARNING_RATE                  0.001
    LOSS_WEIGHTS                   {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
    MASK_POOL_SIZE                 14
    MASK_SHAPE                     [28, 28]
    MAX_GT_INSTANCES               4
    MEAN_PIXEL                     [123.7 116.8 103.9]
    MINI_MASK_SHAPE                (56, 56)
    NAME                           pneumonia
    NUM_CLASSES                    2
    POOL_SIZE                      7
    POST_NMS_ROIS_INFERENCE        1000
    POST_NMS_ROIS_TRAINING         2000
    PRE_NMS_LIMIT                  6000
    ROI_POSITIVE_RATIO             0.33
    RPN_ANCHOR_RATIOS              [0.5, 1, 2]
    RPN_ANCHOR_SCALES              (16, 32, 64, 128)
    RPN_ANCHOR_STRIDE              1
    RPN_BBOX_STD_DEV               [0.1 0.1 0.2 0.2]
    RPN_NMS_THRESHOLD              0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE    256
    STEPS_PER_EPOCH                200
    TOP_DOWN_PYRAMID_SIZE          256
    TRAIN_BN                       False
    TRAIN_ROIS_PER_IMAGE           32
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001
    
    


### 1. 데이터 준비
그리고 이제 pneumonia detection 훈련을 위한 Dataset class를 정의한다.


```python
class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                          annotations=annotations, orig_height=orig_height, orig_width=orig_width)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # 이미지가 grayscale이라면 RGB로 변환
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x,y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)
```

Examine the annotation data, parse the dataset, and view dicom fields


```python
# annotation data
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train_labels.csv'))
anns.head()
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
      <th>patientId</th>
      <th>x</th>
      <th>y</th>
      <th>width</th>
      <th>height</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0004cfab-14fd-4e49-80ba-63a80b6bddd6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00313ee0-9eaa-42f4-b0ab-c148ed3241cd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00322d4d-1c29-4943-afc9-b6754be640eb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>003d8fa0-6bf1-40ed-b54c-ac657f8495c5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00436515-870c-4b36-a041-de91049b9ab4</td>
      <td>264.0</td>
      <td>152.0</td>
      <td>213.0</td>
      <td>379.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# parse dataset, and..
image_fps , image_annotations = parse_dataset(train_dicom_dir, anns=anns)

ds = pydicom.read_file(image_fps[0]) # filepath에서 dicom image 불러오기
image = ds.pixel_array # get image array
```


```python
# DICOM 원본 이미지 사이즈 : 1024 x 1024
ORIG_SIZE = 1024
```

데이터셋을 훈련 데이터셋과 검증 데이터셋으로 분리하자.


```python
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)
val_size = 1500
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print(len(image_fps_train), len(image_fps_val))
```

    25184 1500


훈련 데이터셋은 25184개, 검증 데이터셋은 1500개가 되도록 전체 데이터셋을 나눈 것을 볼 수 있다.

그리고 이제 DetectorDataset class를 이용해서 훈련 데이터셋을 준비한다.


```python
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()
```

훈련 데이터셋을 준비했으니 검증 데이터셋도 준비해주자.


```python
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE,ORIG_SIZE)
dataset_val.prepare()
```

일단 예시로 하나의 이미지를 bounding box와 함께 출력해보자.


```python
class_ids = [0]
while class_ids[0] == 0: # look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1,2,2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:,:,0] * mask[:,:,i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)
```

    (1024, 1024, 3)
    /kaggle/input/stage_2_train_images/620ef67c-0975-48e0-88cd-170226a62e8b.dcm
    [1]





    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_25_1.png)
    



image augmentation을 적용시키는데, 이때 몇개의 값을 좀 바꿔서 finetuning 해준다.


```python
augmentation = iaa.Sequential([
    iaa.OneOf([ # geometric transform
        iaa.Affine(
            scale={'x':(0.98, 1.02), 'y': (0.98, 1.04)},
            translate_percent = {"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate = (-2,2),
            shear=(-1, 1)
        ),
        iaa.PiecewiseAffine(scale=(0.001,0.025))
    ]),
    iaa.OneOf([ # brightness or contranst
        iaa.Multiply((0.9,1.1)),
        iaa.ContrastNormalization((0.9,1.1))
    ]),
    iaa.OneOf([ # blur or sharpen
        iaa.GaussianBlur(sigma=(0.0,0.1)),
        iaa.Sharpen(alpha=(0.0,0.1))
    ])
])

# 위의 예시 이미지에 적용해보기
imggrid = augmentation.draw_grid(image[:,:,0], cols=5, rows=2)
plt.figure(figsize=(30,12))
_ = plt.imshow(imggrid[:,:,0], cmap='gray')
```




    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_27_0.png)
    



원본 이미지랑 별 차이 없어보이지만.. 잘 보면 미세하게 다르다! 이제 데이터가 준비되었으니 모델을 본격적으로 훈련시킨다.

### 2. 모델 훈련시키기

이제 모델을 훈련시킬건데, baseline 모델이어도 훈련하는데 몇시간은 걸린다고 한다... 그래서 이번 코드에서는 훈련시간을 줄이기 위해서 one epoch만 훈련을 시킨다. 


```python
model = modellib.MaskRCNN(mode='training', config=config, model_dir = ROOT_DIR)

# 모델의 마지막 층은 number of class를 필요로하기 때문에 제외한다.
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

LEARNING_RATE = 0.006

import warnings
warnings.filterwarnings('ignore')
```


```python
%%time
model.train(dataset_train, dataset_val,
           learning_rate=LEARNING_RATE,
           epochs = 2,
           layers='heads',
           augmentation=None) # 이 단계에서는 augmentation이 필요없다고 한다.

history = model.keras_model.history.history
```

    
    Starting at epoch 0. LR=0.006
    
    Checkpoint Path: /kaggle/working/pneumonia20230227T0808/mask_rcnn_pneumonia_{epoch:04d}.h5
    Selecting layers to train
    fpn_c5p5               (Conv2D)
    fpn_c4p4               (Conv2D)
    fpn_c3p3               (Conv2D)
    fpn_c2p2               (Conv2D)
    fpn_p5                 (Conv2D)
    fpn_p2                 (Conv2D)
    fpn_p3                 (Conv2D)
    fpn_p4                 (Conv2D)
    In model:  rpn_model
        rpn_conv_shared        (Conv2D)
        rpn_class_raw          (Conv2D)
        rpn_bbox_pred          (Conv2D)
    mrcnn_mask_conv1       (TimeDistributed)
    mrcnn_mask_bn1         (TimeDistributed)
    mrcnn_mask_conv2       (TimeDistributed)
    mrcnn_mask_bn2         (TimeDistributed)
    mrcnn_class_conv1      (TimeDistributed)
    mrcnn_class_bn1        (TimeDistributed)
    mrcnn_mask_conv3       (TimeDistributed)
    mrcnn_mask_bn3         (TimeDistributed)
    mrcnn_class_conv2      (TimeDistributed)
    mrcnn_class_bn2        (TimeDistributed)
    mrcnn_mask_conv4       (TimeDistributed)
    mrcnn_mask_bn4         (TimeDistributed)
    mrcnn_bbox_fc          (TimeDistributed)
    mrcnn_mask_deconv      (TimeDistributed)
    mrcnn_class_logits     (TimeDistributed)
    mrcnn_mask             (TimeDistributed)
    Epoch 1/2
    200/200 [==============================] - 754s 4s/step - loss: 1.7812 - rpn_class_loss: 0.0229 - rpn_bbox_loss: 0.5409 - mrcnn_class_loss: 0.2524 - mrcnn_bbox_loss: 0.5438 - mrcnn_mask_loss: 0.4212 - val_loss: 1.6542 - val_rpn_class_loss: 0.0167 - val_rpn_bbox_loss: 0.5182 - val_mrcnn_class_loss: 0.2651 - val_mrcnn_bbox_loss: 0.4779 - val_mrcnn_mask_loss: 0.3762
    Epoch 2/2
    200/200 [==============================] - 378s 2s/step - loss: 1.6780 - rpn_class_loss: 0.0188 - rpn_bbox_loss: 0.5729 - mrcnn_class_loss: 0.2432 - mrcnn_bbox_loss: 0.4597 - mrcnn_mask_loss: 0.3833 - val_loss: 1.6547 - val_rpn_class_loss: 0.0190 - val_rpn_bbox_loss: 0.5912 - val_mrcnn_class_loss: 0.2252 - val_mrcnn_bbox_loss: 0.4342 - val_mrcnn_mask_loss: 0.3851
    CPU times: user 7min 4s, sys: 12.4 s, total: 7min 16s
    Wall time: 22min 18s


그리고 이제 모델의 모든 layer에 대해서 learning_rate = 0.006으로 설정하고 훈련을 6epoch만큼 시킨다.


```python
%%time
model.train(dataset_train, dataset_val,
           learning_rate=LEARNING_RATE,
           epochs = 6, layers='all',
           augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history : history[k]  = history[k] + new_history[k]
```

    
    Starting at epoch 2. LR=0.006
    
    Checkpoint Path: /kaggle/working/pneumonia20230227T0808/mask_rcnn_pneumonia_{epoch:04d}.h5
    Selecting layers to train
    conv1                  (Conv2D)
    bn_conv1               (BatchNorm)
    res2a_branch2a         (Conv2D)
    bn2a_branch2a          (BatchNorm)
    res2a_branch2b         (Conv2D)
    bn2a_branch2b          (BatchNorm)
    res2a_branch2c         (Conv2D)
    res2a_branch1          (Conv2D)
    bn2a_branch2c          (BatchNorm)
    bn2a_branch1           (BatchNorm)
    res2b_branch2a         (Conv2D)
    bn2b_branch2a          (BatchNorm)
    res2b_branch2b         (Conv2D)
    bn2b_branch2b          (BatchNorm)
    res2b_branch2c         (Conv2D)
    bn2b_branch2c          (BatchNorm)
    res2c_branch2a         (Conv2D)
    bn2c_branch2a          (BatchNorm)
    res2c_branch2b         (Conv2D)
    bn2c_branch2b          (BatchNorm)
    res2c_branch2c         (Conv2D)
    bn2c_branch2c          (BatchNorm)
    res3a_branch2a         (Conv2D)
    bn3a_branch2a          (BatchNorm)
    res3a_branch2b         (Conv2D)
    bn3a_branch2b          (BatchNorm)
    res3a_branch2c         (Conv2D)
    res3a_branch1          (Conv2D)
    bn3a_branch2c          (BatchNorm)
    bn3a_branch1           (BatchNorm)
    res3b_branch2a         (Conv2D)
    bn3b_branch2a          (BatchNorm)
    res3b_branch2b         (Conv2D)
    bn3b_branch2b          (BatchNorm)
    res3b_branch2c         (Conv2D)
    bn3b_branch2c          (BatchNorm)
    res3c_branch2a         (Conv2D)
    bn3c_branch2a          (BatchNorm)
    res3c_branch2b         (Conv2D)
    bn3c_branch2b          (BatchNorm)
    res3c_branch2c         (Conv2D)
    bn3c_branch2c          (BatchNorm)
    res3d_branch2a         (Conv2D)
    bn3d_branch2a          (BatchNorm)
    res3d_branch2b         (Conv2D)
    bn3d_branch2b          (BatchNorm)
    res3d_branch2c         (Conv2D)
    bn3d_branch2c          (BatchNorm)
    res4a_branch2a         (Conv2D)
    bn4a_branch2a          (BatchNorm)
    res4a_branch2b         (Conv2D)
    bn4a_branch2b          (BatchNorm)
    res4a_branch2c         (Conv2D)
    res4a_branch1          (Conv2D)
    bn4a_branch2c          (BatchNorm)
    bn4a_branch1           (BatchNorm)
    res4b_branch2a         (Conv2D)
    bn4b_branch2a          (BatchNorm)
    res4b_branch2b         (Conv2D)
    bn4b_branch2b          (BatchNorm)
    res4b_branch2c         (Conv2D)
    bn4b_branch2c          (BatchNorm)
    res4c_branch2a         (Conv2D)
    bn4c_branch2a          (BatchNorm)
    res4c_branch2b         (Conv2D)
    bn4c_branch2b          (BatchNorm)
    res4c_branch2c         (Conv2D)
    bn4c_branch2c          (BatchNorm)
    res4d_branch2a         (Conv2D)
    bn4d_branch2a          (BatchNorm)
    res4d_branch2b         (Conv2D)
    bn4d_branch2b          (BatchNorm)
    res4d_branch2c         (Conv2D)
    bn4d_branch2c          (BatchNorm)
    res4e_branch2a         (Conv2D)
    bn4e_branch2a          (BatchNorm)
    res4e_branch2b         (Conv2D)
    bn4e_branch2b          (BatchNorm)
    res4e_branch2c         (Conv2D)
    bn4e_branch2c          (BatchNorm)
    res4f_branch2a         (Conv2D)
    bn4f_branch2a          (BatchNorm)
    res4f_branch2b         (Conv2D)
    bn4f_branch2b          (BatchNorm)
    res4f_branch2c         (Conv2D)
    bn4f_branch2c          (BatchNorm)
    res5a_branch2a         (Conv2D)
    bn5a_branch2a          (BatchNorm)
    res5a_branch2b         (Conv2D)
    bn5a_branch2b          (BatchNorm)
    res5a_branch2c         (Conv2D)
    res5a_branch1          (Conv2D)
    bn5a_branch2c          (BatchNorm)
    bn5a_branch1           (BatchNorm)
    res5b_branch2a         (Conv2D)
    bn5b_branch2a          (BatchNorm)
    res5b_branch2b         (Conv2D)
    bn5b_branch2b          (BatchNorm)
    res5b_branch2c         (Conv2D)
    bn5b_branch2c          (BatchNorm)
    res5c_branch2a         (Conv2D)
    bn5c_branch2a          (BatchNorm)
    res5c_branch2b         (Conv2D)
    bn5c_branch2b          (BatchNorm)
    res5c_branch2c         (Conv2D)
    bn5c_branch2c          (BatchNorm)
    fpn_c5p5               (Conv2D)
    fpn_c4p4               (Conv2D)
    fpn_c3p3               (Conv2D)
    fpn_c2p2               (Conv2D)
    fpn_p5                 (Conv2D)
    fpn_p2                 (Conv2D)
    fpn_p3                 (Conv2D)
    fpn_p4                 (Conv2D)
    In model:  rpn_model
        rpn_conv_shared        (Conv2D)
        rpn_class_raw          (Conv2D)
        rpn_bbox_pred          (Conv2D)
    mrcnn_mask_conv1       (TimeDistributed)
    mrcnn_mask_bn1         (TimeDistributed)
    mrcnn_mask_conv2       (TimeDistributed)
    mrcnn_mask_bn2         (TimeDistributed)
    mrcnn_class_conv1      (TimeDistributed)
    mrcnn_class_bn1        (TimeDistributed)
    mrcnn_mask_conv3       (TimeDistributed)
    mrcnn_mask_bn3         (TimeDistributed)
    mrcnn_class_conv2      (TimeDistributed)
    mrcnn_class_bn2        (TimeDistributed)
    mrcnn_mask_conv4       (TimeDistributed)
    mrcnn_mask_bn4         (TimeDistributed)
    mrcnn_bbox_fc          (TimeDistributed)
    mrcnn_mask_deconv      (TimeDistributed)
    mrcnn_class_logits     (TimeDistributed)
    mrcnn_mask             (TimeDistributed)
    Epoch 3/6
    200/200 [==============================] - 1456s 7s/step - loss: 1.6418 - rpn_class_loss: 0.0178 - rpn_bbox_loss: 0.5111 - mrcnn_class_loss: 0.2552 - mrcnn_bbox_loss: 0.4507 - mrcnn_mask_loss: 0.4069 - val_loss: 1.5492 - val_rpn_class_loss: 0.0167 - val_rpn_bbox_loss: 0.4147 - val_mrcnn_class_loss: 0.2863 - val_mrcnn_bbox_loss: 0.4417 - val_mrcnn_mask_loss: 0.3900
    Epoch 4/6
    200/200 [==============================] - 1148s 6s/step - loss: 1.6337 - rpn_class_loss: 0.0173 - rpn_bbox_loss: 0.5100 - mrcnn_class_loss: 0.2649 - mrcnn_bbox_loss: 0.4407 - mrcnn_mask_loss: 0.4008 - val_loss: 1.5859 - val_rpn_class_loss: 0.0165 - val_rpn_bbox_loss: 0.4976 - val_mrcnn_class_loss: 0.2400 - val_mrcnn_bbox_loss: 0.4569 - val_mrcnn_mask_loss: 0.3749
    Epoch 5/6
    200/200 [==============================] - 1176s 6s/step - loss: 1.5655 - rpn_class_loss: 0.0157 - rpn_bbox_loss: 0.4899 - mrcnn_class_loss: 0.2257 - mrcnn_bbox_loss: 0.4321 - mrcnn_mask_loss: 0.4021 - val_loss: 1.7024 - val_rpn_class_loss: 0.0193 - val_rpn_bbox_loss: 0.6065 - val_mrcnn_class_loss: 0.2381 - val_mrcnn_bbox_loss: 0.4459 - val_mrcnn_mask_loss: 0.3925
    Epoch 6/6
    200/200 [==============================] - 1220s 6s/step - loss: 1.5146 - rpn_class_loss: 0.0165 - rpn_bbox_loss: 0.4548 - mrcnn_class_loss: 0.2289 - mrcnn_bbox_loss: 0.4221 - mrcnn_mask_loss: 0.3922 - val_loss: 1.5759 - val_rpn_class_loss: 0.0145 - val_rpn_bbox_loss: 0.5160 - val_mrcnn_class_loss: 0.2215 - val_mrcnn_bbox_loss: 0.4372 - val_mrcnn_mask_loss: 0.3866
    CPU times: user 12min 36s, sys: 29.2 s, total: 13min 6s
    Wall time: 1h 26min 18s


그리고 더 정확한 훈련을 위해서 속도는 더 느리지만 learning_rate를 직전 훈련의 1/5로 설정하고 모델의 전체 layer에 대해 훈련시킨다. 16epoch동안 훈련을 시키는데, 노트북의 저자는 6시간이라는 훈련시간에 맞추기 위해 16epoch로 설정했다고 한다.


```python
%%time
model.train(dataset_train, dataset_val, 
           learning_rate=LEARNING_RATE/5,
           epochs=16, layers='all',
           augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history : history[k] = history[k] + new_history[k]
```

    
    Starting at epoch 6. LR=0.0012000000000000001
    
    Checkpoint Path: /kaggle/working/pneumonia20230227T0808/mask_rcnn_pneumonia_{epoch:04d}.h5
    Selecting layers to train
    conv1                  (Conv2D)
    bn_conv1               (BatchNorm)
    res2a_branch2a         (Conv2D)
    bn2a_branch2a          (BatchNorm)
    res2a_branch2b         (Conv2D)
    bn2a_branch2b          (BatchNorm)
    res2a_branch2c         (Conv2D)
    res2a_branch1          (Conv2D)
    bn2a_branch2c          (BatchNorm)
    bn2a_branch1           (BatchNorm)
    res2b_branch2a         (Conv2D)
    bn2b_branch2a          (BatchNorm)
    res2b_branch2b         (Conv2D)
    bn2b_branch2b          (BatchNorm)
    res2b_branch2c         (Conv2D)
    bn2b_branch2c          (BatchNorm)
    res2c_branch2a         (Conv2D)
    bn2c_branch2a          (BatchNorm)
    res2c_branch2b         (Conv2D)
    bn2c_branch2b          (BatchNorm)
    res2c_branch2c         (Conv2D)
    bn2c_branch2c          (BatchNorm)
    res3a_branch2a         (Conv2D)
    bn3a_branch2a          (BatchNorm)
    res3a_branch2b         (Conv2D)
    bn3a_branch2b          (BatchNorm)
    res3a_branch2c         (Conv2D)
    res3a_branch1          (Conv2D)
    bn3a_branch2c          (BatchNorm)
    bn3a_branch1           (BatchNorm)
    res3b_branch2a         (Conv2D)
    bn3b_branch2a          (BatchNorm)
    res3b_branch2b         (Conv2D)
    bn3b_branch2b          (BatchNorm)
    res3b_branch2c         (Conv2D)
    bn3b_branch2c          (BatchNorm)
    res3c_branch2a         (Conv2D)
    bn3c_branch2a          (BatchNorm)
    res3c_branch2b         (Conv2D)
    bn3c_branch2b          (BatchNorm)
    res3c_branch2c         (Conv2D)
    bn3c_branch2c          (BatchNorm)
    res3d_branch2a         (Conv2D)
    bn3d_branch2a          (BatchNorm)
    res3d_branch2b         (Conv2D)
    bn3d_branch2b          (BatchNorm)
    res3d_branch2c         (Conv2D)
    bn3d_branch2c          (BatchNorm)
    res4a_branch2a         (Conv2D)
    bn4a_branch2a          (BatchNorm)
    res4a_branch2b         (Conv2D)
    bn4a_branch2b          (BatchNorm)
    res4a_branch2c         (Conv2D)
    res4a_branch1          (Conv2D)
    bn4a_branch2c          (BatchNorm)
    bn4a_branch1           (BatchNorm)
    res4b_branch2a         (Conv2D)
    bn4b_branch2a          (BatchNorm)
    res4b_branch2b         (Conv2D)
    bn4b_branch2b          (BatchNorm)
    res4b_branch2c         (Conv2D)
    bn4b_branch2c          (BatchNorm)
    res4c_branch2a         (Conv2D)
    bn4c_branch2a          (BatchNorm)
    res4c_branch2b         (Conv2D)
    bn4c_branch2b          (BatchNorm)
    res4c_branch2c         (Conv2D)
    bn4c_branch2c          (BatchNorm)
    res4d_branch2a         (Conv2D)
    bn4d_branch2a          (BatchNorm)
    res4d_branch2b         (Conv2D)
    bn4d_branch2b          (BatchNorm)
    res4d_branch2c         (Conv2D)
    bn4d_branch2c          (BatchNorm)
    res4e_branch2a         (Conv2D)
    bn4e_branch2a          (BatchNorm)
    res4e_branch2b         (Conv2D)
    bn4e_branch2b          (BatchNorm)
    res4e_branch2c         (Conv2D)
    bn4e_branch2c          (BatchNorm)
    res4f_branch2a         (Conv2D)
    bn4f_branch2a          (BatchNorm)
    res4f_branch2b         (Conv2D)
    bn4f_branch2b          (BatchNorm)
    res4f_branch2c         (Conv2D)
    bn4f_branch2c          (BatchNorm)
    res5a_branch2a         (Conv2D)
    bn5a_branch2a          (BatchNorm)
    res5a_branch2b         (Conv2D)
    bn5a_branch2b          (BatchNorm)
    res5a_branch2c         (Conv2D)
    res5a_branch1          (Conv2D)
    bn5a_branch2c          (BatchNorm)
    bn5a_branch1           (BatchNorm)
    res5b_branch2a         (Conv2D)
    bn5b_branch2a          (BatchNorm)
    res5b_branch2b         (Conv2D)
    bn5b_branch2b          (BatchNorm)
    res5b_branch2c         (Conv2D)
    bn5b_branch2c          (BatchNorm)
    res5c_branch2a         (Conv2D)
    bn5c_branch2a          (BatchNorm)
    res5c_branch2b         (Conv2D)
    bn5c_branch2b          (BatchNorm)
    res5c_branch2c         (Conv2D)
    bn5c_branch2c          (BatchNorm)
    fpn_c5p5               (Conv2D)
    fpn_c4p4               (Conv2D)
    fpn_c3p3               (Conv2D)
    fpn_c2p2               (Conv2D)
    fpn_p5                 (Conv2D)
    fpn_p2                 (Conv2D)
    fpn_p3                 (Conv2D)
    fpn_p4                 (Conv2D)
    In model:  rpn_model
        rpn_conv_shared        (Conv2D)
        rpn_class_raw          (Conv2D)
        rpn_bbox_pred          (Conv2D)
    mrcnn_mask_conv1       (TimeDistributed)
    mrcnn_mask_bn1         (TimeDistributed)
    mrcnn_mask_conv2       (TimeDistributed)
    mrcnn_mask_bn2         (TimeDistributed)
    mrcnn_class_conv1      (TimeDistributed)
    mrcnn_class_bn1        (TimeDistributed)
    mrcnn_mask_conv3       (TimeDistributed)
    mrcnn_mask_bn3         (TimeDistributed)
    mrcnn_class_conv2      (TimeDistributed)
    mrcnn_class_bn2        (TimeDistributed)
    mrcnn_mask_conv4       (TimeDistributed)
    mrcnn_mask_bn4         (TimeDistributed)
    mrcnn_bbox_fc          (TimeDistributed)
    mrcnn_mask_deconv      (TimeDistributed)
    mrcnn_class_logits     (TimeDistributed)
    mrcnn_mask             (TimeDistributed)
    Epoch 7/16
    200/200 [==============================] - 1780s 9s/step - loss: 1.3329 - rpn_class_loss: 0.0129 - rpn_bbox_loss: 0.3849 - mrcnn_class_loss: 0.1770 - mrcnn_bbox_loss: 0.3759 - mrcnn_mask_loss: 0.3821 - val_loss: 1.3017 - val_rpn_class_loss: 0.0114 - val_rpn_bbox_loss: 0.3586 - val_mrcnn_class_loss: 0.1742 - val_mrcnn_bbox_loss: 0.3900 - val_mrcnn_mask_loss: 0.3675
    Epoch 8/16
    200/200 [==============================] - 1238s 6s/step - loss: 1.2999 - rpn_class_loss: 0.0116 - rpn_bbox_loss: 0.3699 - mrcnn_class_loss: 0.1688 - mrcnn_bbox_loss: 0.3708 - mrcnn_mask_loss: 0.3789 - val_loss: 1.3831 - val_rpn_class_loss: 0.0132 - val_rpn_bbox_loss: 0.4584 - val_mrcnn_class_loss: 0.1521 - val_mrcnn_bbox_loss: 0.3877 - val_mrcnn_mask_loss: 0.3717
    Epoch 9/16
    200/200 [==============================] - 1267s 6s/step - loss: 1.2382 - rpn_class_loss: 0.0098 - rpn_bbox_loss: 0.3462 - mrcnn_class_loss: 0.1478 - mrcnn_bbox_loss: 0.3585 - mrcnn_mask_loss: 0.3759 - val_loss: 1.3381 - val_rpn_class_loss: 0.0118 - val_rpn_bbox_loss: 0.4280 - val_mrcnn_class_loss: 0.1472 - val_mrcnn_bbox_loss: 0.3822 - val_mrcnn_mask_loss: 0.3690
    Epoch 10/16
    200/200 [==============================] - 1220s 6s/step - loss: 1.1937 - rpn_class_loss: 0.0099 - rpn_bbox_loss: 0.3072 - mrcnn_class_loss: 0.1531 - mrcnn_bbox_loss: 0.3504 - mrcnn_mask_loss: 0.3731 - val_loss: 1.4476 - val_rpn_class_loss: 0.0110 - val_rpn_bbox_loss: 0.5468 - val_mrcnn_class_loss: 0.1236 - val_mrcnn_bbox_loss: 0.3893 - val_mrcnn_mask_loss: 0.3769
    Epoch 11/16
    200/200 [==============================] - 1225s 6s/step - loss: 1.2339 - rpn_class_loss: 0.0106 - rpn_bbox_loss: 0.3509 - mrcnn_class_loss: 0.1512 - mrcnn_bbox_loss: 0.3487 - mrcnn_mask_loss: 0.3725 - val_loss: 1.3305 - val_rpn_class_loss: 0.0112 - val_rpn_bbox_loss: 0.4320 - val_mrcnn_class_loss: 0.1486 - val_mrcnn_bbox_loss: 0.3706 - val_mrcnn_mask_loss: 0.3681
    Epoch 12/16
    200/200 [==============================] - 1174s 6s/step - loss: 1.2235 - rpn_class_loss: 0.0098 - rpn_bbox_loss: 0.3356 - mrcnn_class_loss: 0.1525 - mrcnn_bbox_loss: 0.3519 - mrcnn_mask_loss: 0.3736 - val_loss: 1.3307 - val_rpn_class_loss: 0.0110 - val_rpn_bbox_loss: 0.4556 - val_mrcnn_class_loss: 0.1347 - val_mrcnn_bbox_loss: 0.3658 - val_mrcnn_mask_loss: 0.3635
    Epoch 13/16
    200/200 [==============================] - 1245s 6s/step - loss: 1.2246 - rpn_class_loss: 0.0093 - rpn_bbox_loss: 0.3443 - mrcnn_class_loss: 0.1475 - mrcnn_bbox_loss: 0.3487 - mrcnn_mask_loss: 0.3747 - val_loss: 1.2735 - val_rpn_class_loss: 0.0098 - val_rpn_bbox_loss: 0.3672 - val_mrcnn_class_loss: 0.1571 - val_mrcnn_bbox_loss: 0.3773 - val_mrcnn_mask_loss: 0.3620
    Epoch 14/16
    200/200 [==============================] - 1265s 6s/step - loss: 1.2082 - rpn_class_loss: 0.0089 - rpn_bbox_loss: 0.3266 - mrcnn_class_loss: 0.1543 - mrcnn_bbox_loss: 0.3477 - mrcnn_mask_loss: 0.3706 - val_loss: 1.3368 - val_rpn_class_loss: 0.0109 - val_rpn_bbox_loss: 0.3986 - val_mrcnn_class_loss: 0.1706 - val_mrcnn_bbox_loss: 0.3895 - val_mrcnn_mask_loss: 0.3671
    Epoch 15/16
    200/200 [==============================] - 1248s 6s/step - loss: 1.2080 - rpn_class_loss: 0.0092 - rpn_bbox_loss: 0.3424 - mrcnn_class_loss: 0.1523 - mrcnn_bbox_loss: 0.3375 - mrcnn_mask_loss: 0.3667 - val_loss: 1.3304 - val_rpn_class_loss: 0.0096 - val_rpn_bbox_loss: 0.4429 - val_mrcnn_class_loss: 0.1479 - val_mrcnn_bbox_loss: 0.3678 - val_mrcnn_mask_loss: 0.3621
    Epoch 16/16
    200/200 [==============================] - 1263s 6s/step - loss: 1.1735 - rpn_class_loss: 0.0085 - rpn_bbox_loss: 0.3124 - mrcnn_class_loss: 0.1512 - mrcnn_bbox_loss: 0.3346 - mrcnn_mask_loss: 0.3668 - val_loss: 1.2860 - val_rpn_class_loss: 0.0088 - val_rpn_bbox_loss: 0.3939 - val_mrcnn_class_loss: 0.1437 - val_mrcnn_bbox_loss: 0.3704 - val_mrcnn_mask_loss: 0.3693
    CPU times: user 24min 31s, sys: 1min 6s, total: 25min 37s
    Wall time: 3h 39min 1s
    


훈련을 마쳤으니 이제 16번의 epoch동안 훈련된 값의 결과값들을 출력해보자.


```python
epochs = range(1,len(next(iter(history.values())))+1)
pd.DataFrame(history, index=epochs)
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
      <th>val_loss</th>
      <th>val_rpn_class_loss</th>
      <th>val_rpn_bbox_loss</th>
      <th>val_mrcnn_class_loss</th>
      <th>val_mrcnn_bbox_loss</th>
      <th>val_mrcnn_mask_loss</th>
      <th>loss</th>
      <th>rpn_class_loss</th>
      <th>rpn_bbox_loss</th>
      <th>mrcnn_class_loss</th>
      <th>mrcnn_bbox_loss</th>
      <th>mrcnn_mask_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.654170</td>
      <td>0.016743</td>
      <td>0.518162</td>
      <td>0.265114</td>
      <td>0.477941</td>
      <td>0.376209</td>
      <td>1.781201</td>
      <td>0.022894</td>
      <td>0.540945</td>
      <td>0.252429</td>
      <td>0.543758</td>
      <td>0.421174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.654725</td>
      <td>0.019027</td>
      <td>0.591201</td>
      <td>0.225225</td>
      <td>0.434199</td>
      <td>0.385070</td>
      <td>1.677987</td>
      <td>0.018838</td>
      <td>0.572944</td>
      <td>0.243216</td>
      <td>0.459668</td>
      <td>0.383319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.549238</td>
      <td>0.016653</td>
      <td>0.414658</td>
      <td>0.286284</td>
      <td>0.441671</td>
      <td>0.389963</td>
      <td>1.641752</td>
      <td>0.017812</td>
      <td>0.511102</td>
      <td>0.255239</td>
      <td>0.450726</td>
      <td>0.406866</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.585936</td>
      <td>0.016501</td>
      <td>0.497574</td>
      <td>0.239968</td>
      <td>0.456937</td>
      <td>0.374947</td>
      <td>1.633669</td>
      <td>0.017301</td>
      <td>0.509994</td>
      <td>0.264871</td>
      <td>0.440703</td>
      <td>0.400791</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.702443</td>
      <td>0.019349</td>
      <td>0.606495</td>
      <td>0.238133</td>
      <td>0.445919</td>
      <td>0.392537</td>
      <td>1.565518</td>
      <td>0.015700</td>
      <td>0.489866</td>
      <td>0.225673</td>
      <td>0.432149</td>
      <td>0.402120</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.575887</td>
      <td>0.014525</td>
      <td>0.515989</td>
      <td>0.221512</td>
      <td>0.437231</td>
      <td>0.386619</td>
      <td>1.514558</td>
      <td>0.016532</td>
      <td>0.454803</td>
      <td>0.228915</td>
      <td>0.422057</td>
      <td>0.392242</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.301677</td>
      <td>0.011423</td>
      <td>0.358645</td>
      <td>0.174153</td>
      <td>0.389951</td>
      <td>0.367494</td>
      <td>1.332906</td>
      <td>0.012941</td>
      <td>0.384914</td>
      <td>0.177003</td>
      <td>0.375935</td>
      <td>0.382104</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.383140</td>
      <td>0.013219</td>
      <td>0.458378</td>
      <td>0.152056</td>
      <td>0.387747</td>
      <td>0.371730</td>
      <td>1.299860</td>
      <td>0.011560</td>
      <td>0.369892</td>
      <td>0.168775</td>
      <td>0.370762</td>
      <td>0.378861</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.338125</td>
      <td>0.011777</td>
      <td>0.427969</td>
      <td>0.147162</td>
      <td>0.382206</td>
      <td>0.369000</td>
      <td>1.238165</td>
      <td>0.009769</td>
      <td>0.346227</td>
      <td>0.147761</td>
      <td>0.358520</td>
      <td>0.375877</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.447643</td>
      <td>0.010978</td>
      <td>0.546836</td>
      <td>0.123612</td>
      <td>0.389313</td>
      <td>0.376893</td>
      <td>1.193691</td>
      <td>0.009927</td>
      <td>0.307183</td>
      <td>0.153120</td>
      <td>0.350388</td>
      <td>0.373063</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.330518</td>
      <td>0.011200</td>
      <td>0.432014</td>
      <td>0.148578</td>
      <td>0.370608</td>
      <td>0.368107</td>
      <td>1.233905</td>
      <td>0.010587</td>
      <td>0.350943</td>
      <td>0.151194</td>
      <td>0.348691</td>
      <td>0.372478</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.330692</td>
      <td>0.010990</td>
      <td>0.455630</td>
      <td>0.134703</td>
      <td>0.365823</td>
      <td>0.363535</td>
      <td>1.223465</td>
      <td>0.009831</td>
      <td>0.335611</td>
      <td>0.152531</td>
      <td>0.351922</td>
      <td>0.373558</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.273452</td>
      <td>0.009775</td>
      <td>0.367214</td>
      <td>0.157146</td>
      <td>0.377296</td>
      <td>0.362011</td>
      <td>1.224559</td>
      <td>0.009309</td>
      <td>0.344316</td>
      <td>0.147495</td>
      <td>0.348707</td>
      <td>0.374721</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.336795</td>
      <td>0.010947</td>
      <td>0.398649</td>
      <td>0.170555</td>
      <td>0.389505</td>
      <td>0.367127</td>
      <td>1.208151</td>
      <td>0.008933</td>
      <td>0.326599</td>
      <td>0.154292</td>
      <td>0.347717</td>
      <td>0.370598</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.330449</td>
      <td>0.009649</td>
      <td>0.442947</td>
      <td>0.147874</td>
      <td>0.367833</td>
      <td>0.362136</td>
      <td>1.208050</td>
      <td>0.009159</td>
      <td>0.342355</td>
      <td>0.152298</td>
      <td>0.337492</td>
      <td>0.366735</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.286039</td>
      <td>0.008812</td>
      <td>0.393891</td>
      <td>0.143686</td>
      <td>0.370377</td>
      <td>0.369262</td>
      <td>1.173484</td>
      <td>0.008466</td>
      <td>0.312371</td>
      <td>0.151195</td>
      <td>0.334610</td>
      <td>0.366831</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(17,5))

plt.subplot(131)
plt.plot(epochs, history['loss'], label='Train loss')
plt.plot(epochs, history['val_loss'], label='Valid loss')
plt.legend()

plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label='Train class loss')
plt.plot(epochs, history['val_mrcnn_class_loss'], label='Valid class loss')
plt.legend()

plt.subplot(133)
plt.plot(epochs, history['mrcnn_bbox_loss'], label='Train box loss')
plt.plot(epochs, history['val_mrcnn_bbox_loss'], label='Valid box loss')
plt.legend()

plt.show()
```




    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_37_0.png)
    



validation loss를 기준으로 최상의 결과를 낸 epoch를 찾아보자.


```python
best_epoch = np.argmin(history['val_loss'])
print('Best epoch:', best_epoch +1 , history['val_loss'][best_epoch])
```

    Best epoch: 13 1.2734521102905274



```python
# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))
    
fps = []
# Pick last directory
for d in dir_names: 
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[best_epoch])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))
```

    Found model /kaggle/working/pneumonia20230227T0808/mask_rcnn_pneumonia_0013.h5


### 3. 훈련된 모델 사용해서 Inference 수행


```python
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
inference_config = InferenceConfig()

# inference를 위한 모델 새롭게 정의
model = modellib.MaskRCNN(mode='inference', 
                         config = inference_config,
                         model_dir = ROOT_DIR)

# trained weights 로드
assert model_path != "", 'Provide path to trained weights'
print('Loading weights from ', model_path)
model.load_weights(model_path, by_name=True)
```

    Loading weights from  /kaggle/working/pneumonia20230227T0808/mask_rcnn_pneumonia_0013.h5
    Re-starting from epoch 13



```python
# set color for class
def get_colors_for_class_ids(class_ids):
    colors=[]
    for class_id in class_ids:
        if class_id == 1 :
            colors.append((.941, .204, .204))
    return colors
```

predicted box와 expected box value는 어떻게 비교되는 것일까? validation dataset을 이용해서 몇개의 이미지를 시각화해서 ground truth 값과 validation dataset에서의 예측값을 비교해보자.


```python
dataset = dataset_val
fig = plt.figure(figsize=(10, 30))

for i in range(6):

    image_id = random.choice(dataset.image_ids)
    
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
    print(original_image.shape)
    plt.subplot(6, 2, 2*i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    
    plt.subplot(6, 2, 2*i + 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
```

    (256, 256, 3)
    
    *** No instances to display *** 
    
    (256, 256, 3)
    
    *** No instances to display *** 
    
    (256, 256, 3)
    
    *** No instances to display *** 
    
    (256, 256, 3)
    
    *** No instances to display *** 
    
    (256, 256, 3)
    (256, 256, 3)
    
    *** No instances to display *** 
    





    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_45_1.png)
    




```python
# get filenames of test datset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)
```

이제 캐글에 제출할 inference 결과를 submission file로 만들어준다.


```python
def predict(image_fps, filepath='submission.csv', min_conf=0.95):
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write('patientId,PredictionString\n')
        
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # convert grey -> RGB
            if len(image.shape) !=3 or image.shape[2] !=3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim = config.IMAGE_MIN_DIM,
                max_dim = config.IMAGE_MAX_DIM,
                min_scale = config.IMAGE_MIN_SCALE,
                mode= config.IMAGE_RESIZE_MODE)
            patient_id = os.path.splitext(os.path.basename(image_id))[0]
            
            results = model.detect([image])
            r = results[0]
            
            out_str = ""
            out_str += patient_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str

            file.write(out_str+"\n")
```


```python
submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
predict(test_image_fps, filepath=submission_fp)
print(submission_fp)
```

    100%|██████████| 3000/3000 [13:28<00:00,  5.98it/s]

    /kaggle/working/submission.csv


    



```python
output = pd.read_csv(submission_fp)
output.head(60)
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
      <th>patientId</th>
      <th>PredictionString</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0e9709fd-a769-4d60-8a06-29a41c7c8297</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0107871b-1095-4cd5-a197-bc7715873cbf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>017cebf7-65c6-4508-bf78-14eb93001ab7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2cfda55a-6021-4740-a2cf-89abb2909762</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>252e33a2-395e-4abd-bc4b-b67e6e8e7fdf</td>
      <td>0.97 176.0 376.0 156.0 136.0 0.95 604.0 292.0...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>04dfc89d-0cdd-43d9-8147-e70ed709adf1</td>
      <td>0.95 248.0 388.0 156.0 104.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1bae5ce0-3a82-4c56-9c6b-c9d601fb8308</td>
      <td>0.96 244.0 540.0 208.0 208.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>24845f34-a0db-4bd7-939b-417364bd0285</td>
      <td>0.99 240.0 328.0 180.0 204.0 0.97 536.0 236.0...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2699ba79-7383-46e2-bf24-53b6bda87dfc</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11e44920-87d0-477b-ab1f-9d21a533595b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>22d4c0ce-db48-4a4d-9653-a9e362c1f812</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2a266b8b-cd95-4d58-a018-f26f12d34f56</td>
      <td>0.97 572.0 196.0 248.0 392.0 0.96 216.0 560.0...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c121b434-a3cf-415b-9229-0ec10a66d6be</td>
      <td>0.97 300.0 460.0 172.0 148.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>22bc8cc0-b7eb-487e-94ec-cdec71aff3f3</td>
      <td>0.96 192.0 340.0 260.0 224.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1ad5f416-0e7e-4f08-9e8d-abd19bbe25f5</td>
      <td>0.96 608.0 448.0 248.0 340.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1114d11a-201b-4105-8d7f-032dc2143e61</td>
      <td>0.99 268.0 412.0 164.0 140.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>290c3c3c-8864-43a7-894b-52edb4b21957</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>22a5a7ed-41dc-47ad-bf9a-660eeee035bd</td>
      <td>0.95 548.0 572.0 184.0 236.0 0.95 272.0 632.0...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2775f13e-cbd7-4216-b89f-4c74126af11b</td>
      <td>0.96 312.0 408.0 132.0 176.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>11cc923f-ec9a-49c5-b4a3-ceb3c6da3273</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>117d67ee-2bb5-43a9-8cc8-d32ec1cf6b53</td>
      <td>0.98 204.0 520.0 200.0 276.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>116e1a13-ac55-499d-bcc9-ae9e0bd450d2</td>
      <td>0.98 544.0 520.0 196.0 232.0 0.97 224.0 456.0...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>251ecdcd-70e7-4aa2-91f6-208d61a6b41e</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2d70844f-b516-43f7-8bdc-a6d56b5c81de</td>
      <td>0.96 172.0 312.0 212.0 232.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>04627a72-3f26-4b2e-864a-01da3abc0cee</td>
      <td>0.98 236.0 192.0 208.0 400.0 0.98 592.0 260.0...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2479b752-baae-4fbb-8b35-5a43464f1188</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0ff9da5a-f294-4566-87e2-924899770072</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3052367c-ccde-434d-9ed0-2d5f364e4b04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0f96b5e6-6714-4613-af33-edfbdbc1e0bd</td>
      <td>0.98 512.0 304.0 228.0 444.0 0.97 136.0 460.0...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2ea32c58-de2a-4085-bd7c-7b158bd681c8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>265dd221-9049-4bca-b5c0-4118dafa55c5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>19fe1b8d-e93e-4810-b8cb-19c59141c90c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2069e8aa-9fe5-442a-9c28-e3f12d3addf6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>13c66d0d-32ab-4a06-8c35-5abd5d1e4bd1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0dcf2fda-af28-4473-8ac3-47bb30d332a8</td>
      <td>0.96 208.0 484.0 204.0 300.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2d1c672f-5583-4dea-aa29-fd0f79d579d2</td>
      <td>0.95 252.0 468.0 208.0 188.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1ac1dc25-bf30-45be-9705-b983eb92974d</td>
      <td>0.98 172.0 464.0 256.0 184.0 0.96 600.0 400.0...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1270a024-bebe-4b8f-8702-f029f51d532e</td>
      <td>0.97 132.0 460.0 188.0 192.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0de94683-fb0c-444d-be13-7617a09a3247</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>03b16b8f-a03c-409c-8740-7e694b7d58e5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0e85807d-4f11-43eb-9bf8-deecfcf79961</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>03dfd775-0cb4-4ede-9eba-e01df8a84965</td>
      <td>0.96 260.0 452.0 168.0 284.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>10b73213-154e-470b-b4c8-8f8ecbc5d756</td>
      <td>0.98 284.0 584.0 216.0 196.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>302d8a19-5868-4286-be5d-a9be0b7310c1</td>
      <td>0.96 192.0 424.0 216.0 252.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1e62c8e9-ce2f-4958-b8a0-f426e09c1381</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>27e2e08f-207c-4534-8da7-412fb7868a8a</td>
      <td>0.95 628.0 432.0 252.0 304.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>038cd87f-d57f-4db6-9548-879dc697d656</td>
      <td>0.96 168.0 456.0 264.0 244.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>20702407-dfbc-49b8-9324-311961fb1ee2</td>
      <td>0.99 176.0 308.0 204.0 280.0 0.96 544.0 328.0...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1f042317-fe9a-4ac4-b5af-6771e6be78f0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2696d094-0bf0-47b5-bf0a-13bbb16a03be</td>
      <td>0.99 236.0 344.0 208.0 252.0 0.98 624.0 328.0...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>24db06a7-86e0-4690-8d57-088ab93d3e04</td>
      <td>0.96 676.0 520.0 252.0 328.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>27d3a0fe-120f-488e-8cb1-4dd73afa755c</td>
      <td>0.96 120.0 552.0 192.0 304.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2fd17ea9-5654-4632-9dd3-a9ec6c81aaf4</td>
      <td>0.97 576.0 292.0 208.0 428.0 0.96 204.0 516.0...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>308d1be1-7a4a-460f-8429-0e02f1ca51fd</td>
      <td>0.96 232.0 388.0 188.0 208.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1a0309e4-762b-4b64-9e2c-0f16dc4d5381</td>
      <td>0.96 276.0 396.0 136.0 168.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>25f6cad5-c636-48af-bea6-7bc4263ae253</td>
      <td>0.98 164.0 488.0 268.0 284.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>14b22d6c-7cb3-4331-bdfe-b67a20a623d4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1378f79c-6ea0-41e9-98a9-1b15c2c54a7d</td>
      <td>0.96 512.0 388.0 224.0 240.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>14a5a858-39e6-47b2-b0ed-35023536b3ff</td>
      <td>0.96 236.0 568.0 176.0 120.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>30451f0f-6692-4633-b234-94ae2b5960de</td>
      <td>0.97 580.0 212.0 256.0 472.0 0.96 184.0 608.0...</td>
    </tr>
  </tbody>
</table>
</div>



최종 예측값을 시각화해보는 것으로 마치겠다.


```python
def visualize():
    image_id = random.choice(test_image_fps)
    ds = pydicom.read_file(image_id)
    
    # original image
    image = ds.pixel_array
    
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 
    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    patient_id = os.path.splitext(os.path.basename(image_id))[0]
    print(patient_id)
    
    results = model.detect([resized_image])
    r = results[0]
    for bbox in r['rois']: 
        print(bbox)
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2]  * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width = x2 - x1 
        height = y2 - y1 
        print("x {} y {} h {} w {}".format(x1, y1, width, height))
    plt.figure() 
    plt.imshow(image, cmap=plt.cm.gist_gray)

visualize()
visualize()
visualize()
visualize()
```

    26cd4988-ff71-4d30-b36a-99645c022f90
    [131 143 180 196]
    x 572 y 524 h 212 w 196
    [118  37 160  81]
    x 148 y 472 h 176 w 168
    1277f37b-d591-48ad-8d7d-fcf85c5cf40f
    [ 94  73 139 114]
    x 292 y 376 h 164 w 180
    [ 93 160 154 205]
    x 640 y 372 h 180 w 244
    0f08ceaf-0815-4e90-bbf0-5252bdd05d29
    [ 95 144 217 199]
    x 576 y 380 h 220 w 488
    [ 83  47 215 107]
    x 188 y 332 h 240 w 528
    2d77734e-9ccc-47e8-a3bc-eee7c719a271
    [122  68 182 123]
    x 272 y 488 h 220 w 240
    [130 160 199 218]
    x 640 y 520 h 232 w 276





    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_52_1.png)
    






    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_52_2.png)
    






    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_52_3.png)
    






    
![png](mask-rcnn-and-coco-transfer-learning-lb-0-155_files/mask-rcnn-and-coco-transfer-learning-lb-0-155_52_4.png)
    


