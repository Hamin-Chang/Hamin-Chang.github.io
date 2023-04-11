---
title: '[IC/Kaggle] Mask Detection - 마스크 착용 유무 이미지 분류 😷 '
toc: true
toc_sticky: true
categories:
  - kaggle-imageclassification
---

## VGG19로 마스크 착용 유무 판단하기

이번 글에서는 이미지 내의 사람들의 마스크 착용 유무를 판단하는 문제를 VGG19를 사용해서 해결한다. [<U>NAGESH SINGH CHAUHAN의 캐글 노트북</U>](https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19)의 코드를 사용했는데, 참고한 노트북에서는 이미지 내의 사람들끼리 사회적 거리두기를 지켰는지에 대한 문제도 해결하는데, 이는 참고만 하면 될 것 같다.

또한 이미지 내에서 사람의 얼굴이 어디에 있는지 찾기 위해서 object detection 또한 적용하는데, 이 부분도 이 글에서 다루는 주된 내용이 아니기 때문에 참고하고 넘어가면 된다.




```python
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import distance

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


### 0.1 Haar cascade 사용해서 얼굴 탐지하기 & 사회적 거리두기 판단
Haar cascade를 사용해서 마스크 착용 유무를 판단할 얼굴들을 이미지에서 탐지하기 위해 객체 탐지를 사용하는 부분이기 때문에 코드만 작성하고 넘어간다.


```python
face_model = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
```


```python
# 샘플 이미지 출력
import matplotlib.pyplot as plt

img = cv2.imread('../input/face-mask-detection/images/maksssksksss296.png')
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

# 얼굴 좌표 (x,y,w,h)
faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for (x,y,w,h) in faces:
    cv2.rectangle(out_img, (x,y), (x+w, y+h), (0,0,255),1)
plt.figure(figsize=(12,12))
plt.imshow(out_img)
```




    <matplotlib.image.AxesImage at 0x7b28313ff2d0>






    
![1](https://user-images.githubusercontent.com/77332628/231181461-2f4d37be-58b4-41c9-a3a8-ddc88910dc8f.png)
    



### 0.2 사회적 거리두기 판단하기

위 코드에서 찾은 얼굴들의 좌표들끼리의 거리를 계산해서 MIN_DISTANCE보다 거리가 짧으면 빨간 박스로 처리하고, 거리가 멀면 초록 박스로 처리한다.


```python
MIN_DISTANCE = 130

if len(faces)>=2:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            if dist < MIN_DISTANCE: # 거리가 MIN_DISTANCE보다 짧을 경우
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        if label[i]==1:
            cv2.rectangle(new_img, (x,y), (x+w,y+h), (255,0,0),1)
        else :
            cv2.rectangle(new_img, (x,y), (x+w,y+h), (0,255,0),1)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)

else:
    print("No. of faces detected is less than 2")
                
```




    
![2](https://user-images.githubusercontent.com/77332628/231181472-de5bf493-7edc-4236-87ef-dd39451a4dc7.png)
    



### 1. mask detection 위한 dataset 준비

먼저 vgg19를 사용하기 위한 라이브러리를 import하고, train data와 test data의 파일 경로를 설정한 후, data augmentation을 적용해서 overfit를 최대한 방지해준다.


```python
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#Load train and test set
train_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'
test_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test'
val_dir = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128,128), class_mode='categorical', batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128,128), class_mode='categorical', batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128,128), class_mode='categorical', batch_size=32)
```

    Found 10000 images belonging to 2 classes.
    Found 800 images belonging to 2 classes.
    Found 800 images belonging to 2 classes.


### 2. VGG19 transfer learning model 구축

기존에 있는 VGG19 모델을 불러와서 마지막의 Flatten과 Dense부분만 추가해준다.


```python
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
    80142336/80134624 [==============================] - 4s 0us/step
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg19 (Functional)           (None, 4, 4, 512)         20024384  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 2)                 16386     
    =================================================================
    Total params: 20,040,770
    Trainable params: 16,386
    Non-trainable params: 20,024,384
    _________________________________________________________________



```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

history = model.fit_generator(generator=train_generator, 
                             steps_per_epoch = len(train_generator)//32,
                             epochs=20,
                             validation_data = val_generator,
                             validation_steps = len(val_generator)//32)
```

    /opt/conda/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py:1844: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      warnings.warn('`Model.fit_generator` is deprecated and '


    Epoch 1/20
    9/9 [==============================] - 6s 280ms/step - loss: 0.8958 - accuracy: 0.5523
    Epoch 2/20
    9/9 [==============================] - 2s 208ms/step - loss: 0.3052 - accuracy: 0.8746
    Epoch 3/20
    9/9 [==============================] - 2s 206ms/step - loss: 0.2156 - accuracy: 0.9160
    Epoch 4/20
    9/9 [==============================] - 2s 213ms/step - loss: 0.1700 - accuracy: 0.9472
    Epoch 5/20
    9/9 [==============================] - 2s 207ms/step - loss: 0.1603 - accuracy: 0.9441
    Epoch 6/20
    9/9 [==============================] - 2s 211ms/step - loss: 0.0993 - accuracy: 0.9666
    Epoch 7/20
    9/9 [==============================] - 2s 199ms/step - loss: 0.1094 - accuracy: 0.9661
    Epoch 8/20
    9/9 [==============================] - 2s 207ms/step - loss: 0.1376 - accuracy: 0.9470
    Epoch 9/20
    9/9 [==============================] - 2s 207ms/step - loss: 0.0723 - accuracy: 0.9905
    Epoch 10/20
    9/9 [==============================] - 2s 202ms/step - loss: 0.0754 - accuracy: 0.9811
    Epoch 11/20
    9/9 [==============================] - 2s 240ms/step - loss: 0.0954 - accuracy: 0.9653
    Epoch 12/20
    9/9 [==============================] - 2s 196ms/step - loss: 0.0440 - accuracy: 0.9918
    Epoch 13/20
    9/9 [==============================] - 2s 206ms/step - loss: 0.0977 - accuracy: 0.9599
    Epoch 14/20
    9/9 [==============================] - 2s 199ms/step - loss: 0.1198 - accuracy: 0.9590
    Epoch 15/20
    9/9 [==============================] - 2s 192ms/step - loss: 0.0663 - accuracy: 0.9720
    Epoch 16/20
    9/9 [==============================] - 2s 196ms/step - loss: 0.0808 - accuracy: 0.9726
    Epoch 17/20
    9/9 [==============================] - 2s 190ms/step - loss: 0.1152 - accuracy: 0.9568
    Epoch 18/20
    9/9 [==============================] - 2s 190ms/step - loss: 0.0818 - accuracy: 0.9738
    Epoch 19/20
    9/9 [==============================] - 2s 209ms/step - loss: 0.0447 - accuracy: 0.9828
    Epoch 20/20
    9/9 [==============================] - 2s 184ms/step - loss: 0.0629 - accuracy: 0.9847



```python
model.evaluate_generator(test_generator)
```

    /opt/conda/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py:1877: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.
      warnings.warn('`Model.evaluate_generator` is deprecated and '





    [0.054718971252441406, 0.9825000166893005]



test data로 test 했을 때 98.2%의 높은 정확도를 갖는 모델을 만들었다.

### 3. test data 출력

마지막으로 test data중 이미지 하나를 모델에 입력하여 결과를 출력해본다.


```python
sample_mask_img = cv2.imread('../input/face-mask-12k-images-dataset/Face Mask Dataset/Test/WithMask/768.png')
sample_mask_img = cv2.resize(sample_mask_img, (128,128))
plt.imshow(sample_mask_img)
sample_mask_img = np.reshape(sample_mask_img, [1,128,128,3])
sample_mask_img = sample_mask_img / 255.0
```




![3](https://user-images.githubusercontent.com/77332628/231181474-e6fc660d-cf74-44db-b809-24cfd2e1ba3a.png)
    




```python
model.predict(sample_mask_img)
```




    array([[1.000000e+00, 1.039873e-34]], dtype=float32)



모델이 이미지가 마스크를 쓰고 있을 확률을 93.7%라고 잘 예측하고 있다.

이제 모델을 저장하고 haar cascade와 mask detection모델을 합치자.


```python
model.save('masknet.h5')
```


```python
mask_label = {0:'Mask', 1:'No Mask'}
dist_label = {0:(0,255,0), 1:(255,0,0)}
```


```python
if len(faces)>=2:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
    
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img, mask_label[mask_result.argmax()], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_label[label[i]],2)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[label[i]],1)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)

else:
    print("No. of faces detected is less than 2")
```




![4](https://user-images.githubusercontent.com/77332628/231181475-05f0581f-b4ba-4dde-a60d-03902a54a391.png)

    



출처 :

[<U>NAGESH SINGH CHAUHAN의 캐글 노트북</U>](https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19)

