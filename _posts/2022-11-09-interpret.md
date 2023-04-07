---
toc : true
title : '[CV/KERAS] 컨브넷 해석 - 모델이 무엇을 학습했는가? 🔍'
layout : single
toc: true
toc_sticky: true
categories:
  - CVBasic
---


## 컨브넷이 학습한 것 해석하기


### 6.0 들어가며

사람들은 딥러닝 모델을 '블랙박스' 같다고 자주 이야기한다. 모델이 학습한 표현을 사람이 이해하기 쉬운 형태로 뽑아내는 것이 어렵기 때문이다. 하지만 이는 컨브넷 모델에서는 틀린말이다. 컨브넷의 표현은 시각적인 개념을 학습한 것이기 때문에 시각화에 용이하다. 이번 글에서는 컨브넷의 표현들을 시각화하고 해석하는 다음의 세가지 기법을 알아보겠다. 


*   **컨브넷 중간층의 출력 시각화하기** : 연속된 컨브넷 층이 입력을 어떻게 변형시키는지 이해하고 개별적인 필터의 의미를 파악하는데 도움이 된다.
*   **컨브넷 필터 시각화하기** : 컨브넷 필터가 찾으려는 시각적인 패턴과 개념이 무엇인지 상세하게 이해하는데 도움이 된다.
*   **클래스 활성화에 대한 히트맵(heatmap)을 이미지에 시각화하기** : 어떤 클래스에 속하는 데 이미지의 어느부분이 기여했는지 이해하고 이미지에서 객체의 위치를 추정하는 데 도움이 된다.

첫번째 항목만 [이전 글](https://hamin-chang.github.io/small/)의 강아지 vs 고양이 분류 문제에서 밑바닥부터 훈련시킨 작은 컨브넷을 사용하고 , 다른 두 가지 항목은 사전 훈련된 Xception 모델을 사용하겠다.

### 6.1 중간 활성화 시각화

중간층의 활성화 시각화는 어떤 입력이 주어졌을 때 모델에 있는 여러 합성곱과 풀링 층이 반환하는 값을 그리는 것이다. (층의 출력을 종종 활성화라고 부른다.) 이 방법은 네트워크에 의해 학습된 필터들이 어떻게 입력을 분해하는지 보여준다. 너비, 높이, 깊이(채널) 3개의 차워에 대해 특성 맵을 시각화하는 것이 좋다. 각 채널은 비교적 독립적인 특성을 인코딩하기 때문에 특성맵의 각 채널 내용을 독립적인 2D 이미지로 그리는 것이 좋은 방법이다. [이전 글](https://hamin-chang.github.io/small/)에서 저장했던 모델을 로드해서 시작한다.


```python

# 이전 글에서 모델 저장 안했다면 이 코드 사용해서 모델 다운로드
!wget https://github.com/rickiepark/deep-learning-with-python-2nd/raw/main/convnet_from_scratch_with_augmentation.keras

from tensorflow import keras
model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")
```

    --2022-11-07 03:06:02--  https://github.com/rickiepark/deep-learning-with-python-2nd/raw/main/convnet_from_scratch_with_augmentation.keras
    Resolving github.com (github.com)... 20.205.243.166
    Connecting to github.com (github.com)|20.205.243.166|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/rickiepark/deep-learning-with-python-2nd/main/convnet_from_scratch_with_augmentation.keras [following]
    --2022-11-07 03:06:03--  https://raw.githubusercontent.com/rickiepark/deep-learning-with-python-2nd/main/convnet_from_scratch_with_augmentation.keras
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7994528 (7.6M) [application/octet-stream]
    Saving to: ‘convnet_from_scratch_with_augmentation.keras’
    
    convnet_from_scratc 100%[===================>]   7.62M  --.-KB/s    in 0.04s   
    
    2022-11-07 03:06:04 (181 MB/s) - ‘convnet_from_scratch_with_augmentation.keras’ saved [7994528/7994528]
    



```python
model.summary()
```

    Model: "model_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_4 (InputLayer)        [(None, 180, 180, 3)]     0         
                                                                     
     sequential (Sequential)     (None, 180, 180, 3)       0         
                                                                     
     rescaling_1 (Rescaling)     (None, 180, 180, 3)       0         
                                                                     
     conv2d_11 (Conv2D)          (None, 178, 178, 32)      896       
                                                                     
     max_pooling2d_6 (MaxPooling  (None, 89, 89, 32)       0         
     2D)                                                             
                                                                     
     conv2d_12 (Conv2D)          (None, 87, 87, 64)        18496     
                                                                     
     max_pooling2d_7 (MaxPooling  (None, 43, 43, 64)       0         
     2D)                                                             
                                                                     
     conv2d_13 (Conv2D)          (None, 41, 41, 128)       73856     
                                                                     
     max_pooling2d_8 (MaxPooling  (None, 20, 20, 128)      0         
     2D)                                                             
                                                                     
     conv2d_14 (Conv2D)          (None, 18, 18, 256)       295168    
                                                                     
     max_pooling2d_9 (MaxPooling  (None, 9, 9, 256)        0         
     2D)                                                             
                                                                     
     conv2d_15 (Conv2D)          (None, 7, 7, 256)         590080    
                                                                     
     flatten_3 (Flatten)         (None, 12544)             0         
                                                                     
     dropout (Dropout)           (None, 12544)             0         
                                                                     
     dense_3 (Dense)             (None, 1)                 12545     
                                                                     
    =================================================================
    Total params: 991,041
    Trainable params: 991,041
    Non-trainable params: 0
    _________________________________________________________________


이 네트워크를 훈련할 때 사용했던 이미지가 아닌 다른 고양이 사진 하나를 입력 이미지로 선택한다.


```python
from tensorflow import keras
import numpy as np

img_path = keras.utils.get_file(  # 이미지 다운로드
    fname="cat.jpg",
    origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def get_img_array(img_path, target_size):
  img = keras.utils.load_img(   # 이미지 파일을 로드하고 크기를 변경
      img_path, target_size=target_size)
  array = keras.utils.img_to_array(img) # 이미지를 (180,180,3) 크기의 float32 넘파이 배열로 변환
  array = np.expand_dims(array,axis=0) # 배열을 단일 이미지 배치로 변환하기 위해 차원추가해서 (1,180,180,3)으로 변환
  return array

img_tensor = get_img_array(img_path,target_size=(180,180))
```

    Downloading data from https://img-datasets.s3.amazonaws.com/cat.jpg
    80329/80329 [==============================] - 0s 6us/step



```python
# 이미지 출력 코드
import matplotlib.pyplot as plt

plt.axis('off')
plt.imshow(img_tensor[0].astype('uint8'))
plt.show()
```




    
![int0](https://user-images.githubusercontent.com/77332628/200705472-758a0630-9c60-4fc8-9061-93b121d5b4f3.png)

시각화하고 싶은 특성 맵을 추출하기 위해 이미지 배치를 입력으로 받아서 모든 합성곱과 풀링 층의 활성화를 출력하는 케라스 모델을 구축한다.


```python
from tensorflow.keras import layers

layer_outputs = []
layer_names = []
for layer in model.layers:
  if isinstance(layer,(layers.Conv2D,layers.MaxPooling2D)):
    layer_outputs.append(layer.output) # 모든 Conv2D와 MaxPooling2D 층의 출력을 하나의 리스트에 추가
    layer_names.append(layer.name) # 나중을 위해 층 이름 저장

#모델 입력이 주어졌을 때 층의 출력을 반환하는 모델 구축
activation_model = keras.Model(inputs=model.input,outputs=layer_outputs)
```

입력 이미지가 주입되면 이 모델은 원본 모델의 활성화 값을 반환한다. 이 모델은 다중 출력 모델로 하나의 입력층과 층의 활성화(출력)마다 하나씩 총 9개의 출력을 가진다.


```python
activations = activation_model.predict(img_tensor)
# 층 활성화마다 배열 하나씩 총 9개의 넘파이 배열로 구성된 리스트 반환
```

    1/1 [==============================] - 8s 8s/step


예를 들어, 다음 코드는 고양이 이미지에 대한 첫 번째 합성곱 층의 활성화 값이다.


```python
first_layer_activation = activations[0]
print(first_layer_activation.shape)
```

    (1, 178, 178, 32)


이 활성화는 32개의 채널을 가진 178x178 크기의 특성 맵이다. 다음 코드는 원본 모델의 첫번째 층 활성화 중에서 다섯번째 채널을 그리는 코드다.


```python
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0,:,:,5], cmap ='viridis')
plt.show()
```




    
![int1](https://user-images.githubusercontent.com/77332628/200705468-0e2ba000-11a8-4465-b296-0b70fe2b2bef.png)


이제 네트워크의 모든 활성화를 시각화 해보겠다. 각 층의 활성화에 있는 모든 채널을 그리기 위해 하나의 큰 이미지 그리드에 추출한 결과를 나란히 쌓겠다.


```python
images_per_row = 16 # 그리드 행 개수

# 활성화와 해당 층 이름에 대해 루프를 순회한다. 
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] # 층 활성화 크기는 (1,size,size,n_features)
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row # 그리드 열 개수
    display_grid = np.zeros(((size + 1) * n_cols - 1,  # 빈 그리드 준비
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy() # 하나의 채널(또는 특성) 이미지

            # 모두 0인 채널은 그대로 두고,
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            
            # 채널 값을 [0,255] 범위로 정규화
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")

            # 빈 그리드에 채널 행렬 저장
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row] = channel_image
    # 그리드 출력
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")
```




    
![int2](https://user-images.githubusercontent.com/77332628/200705530-24311444-ff63-495a-93a3-873a6027c295.png)
![int3](https://user-images.githubusercontent.com/77332628/200705531-9cac506d-c4de-4173-8b1e-31b4a63f45bf.png)
![int4](https://user-images.githubusercontent.com/77332628/200705536-b5324b0a-c262-4ef8-9fd7-685828ebca96.png)
![int5](https://user-images.githubusercontent.com/77332628/200705539-74ec3541-4506-4bdf-8805-270efbf96b51.png)
![int6](https://user-images.githubusercontent.com/77332628/200705543-14b7aa3e-b90a-4626-81a4-529e63d21614.png)
![int7](https://user-images.githubusercontent.com/77332628/200705545-54ff0e7b-4f8b-4562-b2eb-3e29f794a1eb.png)
![int8](https://user-images.githubusercontent.com/77332628/200705546-5157e895-8796-4c52-a82b-c58ea4c5e94c.png)
![int9](https://user-images.githubusercontent.com/77332628/200705547-876ece46-e31a-46ab-85a4-22376e38fba0.png)
![int10](https://user-images.githubusercontent.com/77332628/200705550-6e16a768-c29d-404c-985e-9ba4cdf89e90.png)
    



출력된 이미지가 굉장히 많지만 다음의 몇가지 주목할 내용이 있다.


*   첫번째 층은 여러 종류의 에지 감지기를 모아 놓은 것 같다. 이 단계의 화렁화에서는 초기 이미지에 있는 거의 모든 정보가 유지된다.
*   층이 깊어질수록 활성화는 점점 더 추상적으로 되고 시각적으로 이해하기 어려워진다. '고양이 귀'와 '고양이 눈'과 같은 고수준 개념을 인코딩하기 시작하고, 깊은 층의 표현은 이미지의 시각적 콘텐츠에 관한 정보가 점점 줄어들고 이미지의 클래스에 관한 정보가 점점 증가한다.
*   층이 깊어질수록 비어 있는 활성화가 늘어난다. 첫 번째 층에서는 거의 모든 필터가 입력 이미지에 활성화 되었지만 층을 올라가면서 활성화되지 않는 필터들이 생긴다. 이는 필터에 인코딩된 패턴이 입력 이미지에 나타나지 않았다는 것을 의미한다.

깊은 층의 활성화는 특정 입력에 대한 시각적 정보는 점점 줄고, 타깃에 관한 정보는 점점 더 증가한다. 심층 신경망은 반복적인 변환을 통해 관계없는 정보를 걸러내고 유용한 정보는 강조되고 개선된다(여기에서는 이미지의 클래스). 사람과 동물이 세상을 인지하는 방식이 이와 비슷하다. 물체의 구체적인 모양을 기억하는 것이 아니라 우리 뇌는 시각적 입력에서 관련성이 적은 요소를 필터링해서 고수준 개념으로 변환한다.



### 6.2 컨브넷 필터 시각화하기

컨브넷이 학습한 필터를 조사하는 또 다른 방법은 각 필터가 반응하는 시각적 패턴을 그려보는 것이다. 빈 입력 이미지에서 시작해서 특정 필터의 응답을 최대화하기 위해 컨브넷 입력 이미지에 경사 상승법을 적용한다. 결과적으로 입력 이미지는 선택된 필터가 최대로 응답하는 이미지가 될 것이다.

이번에는 ImageNet에서 사전 훈련된 Xception 모델의 필터를 사용한다. 전체 과정은 간단하다. 특성 합성곱 층의 한 필터 값을 최대화하는 손실함수를 정의하고 이 활성화 값을 최대화하기 위해 입력 이미지를 변경하도록 확률적 경사 상승법을 사용한다. 이는 GradientTape 객체를 사용해서 저수준의 훈련 루프를 구현하는 방법이다. 먼저 ImageNet에서 사전 훈련된 가중치를 로드해서 Xception 모델을 만들어보자. 


```python
model = keras.applications.xception.Xception(
    weights = 'imagenet',
    include_top = False) # 분류 층이 필요없기 때문에 모델의 상단부 제외
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
    83683744/83683744 [==============================] - 4s 0us/step


우리는 이 모델의 합성곱 층인 Conv2D와 SeperableConv2D 층에 관심이 있기 때문에 이런 층의 출력을 얻으려면 이름을 알아야 한다. 깊이 순서대로 이름을 출력하자.


```python
for layer in model.layers:
  if isinstance(layer,(keras.layers.Conv2D, keras.layers.SeparableConv2D)):
    print(layer.name)
```

    block1_conv1
    block1_conv2
    block2_sepconv1
    block2_sepconv2
    conv2d
    block3_sepconv1
    block3_sepconv2
    conv2d_1
    block4_sepconv1
    block4_sepconv2
    conv2d_2
    block5_sepconv1
    block5_sepconv2
    block5_sepconv3
    block6_sepconv1
    block6_sepconv2
    block6_sepconv3
    block7_sepconv1
    block7_sepconv2
    block7_sepconv3
    block8_sepconv1
    block8_sepconv2
    block8_sepconv3
    block9_sepconv1
    block9_sepconv2
    block9_sepconv3
    block10_sepconv1
    block10_sepconv2
    block10_sepconv3
    block11_sepconv1
    block11_sepconv2
    block11_sepconv3
    block12_sepconv1
    block12_sepconv2
    block12_sepconv3
    block13_sepconv1
    block13_sepconv2
    conv2d_3
    block14_sepconv1
    block14_sepconv2


Xception 모델은 여러개의 합성곱 층을 담은 블록으로 구성되어 있다. 예를 들어 SeperableConv2D 층의 이름은 모두 block6_sepconv1, block7_sepconv2와 같은 식이다.

이제 특정 층의 출력을 반환하는 두 번째 보델, 즉 특성 추출 모델을 만들어보자. 함수형 API를 사용한 모델이기 때문에 Xception 전체 코드를 복사할 필요 없이 한 층의 output을 추출해서 새 모델에 재사용할 수 있다.


```python
layer_name = 'block3_sepconv1' # Xception 합성곱 기반에 있는 다른 층의 이름으로 교체 가능
layer = model.get_layer(name = layer_name) # 관심 대상인 층의 객체

# 입력 이미지가 주어졌을 때 해당 층의 출력을 반환하는 모델 구축
feature_extractor = keras.Model(inputs=model.input,outputs=layer.output)
```

이 모델을 사용하려면 어떤 입력 데이터에서 모델을 호출하면 된다.


```python
activation = feature_extractor(
    keras.applications.xception.preprocess_input(img_tensor)) 
```

특성 추출 모델을 사용해서 입력 이미지가 층의 필터를 얼마나 활성화하는지 정향화된 스칼라 값을 반환하는 함수를 정의해본다. 이 함수가 경사 상승법 과정도안 최대화할 '손실 함수'가 된다.


```python
import tensorflow as tf
# 이 손실 함수는 이미지 텐서와 필터 인덱스(정수)를 입력으로 받음 
def comput_loss(image,filter_index): 
  activation = feature_extractor(image)

  # 손실에 경계 픽셀 제외시켜서 경계에 나타는 부수효과 제외
  filter_activation = activation[:,2:-2,2:-2,filter_index]

  # 이 필터에 대한 활성화 값 평균 반환
  return tf.reduce_mean(filter_activation)
```

GradientTape을 사용해서 경사 상승법 단계를 구현한다. 

경사 상승법 과정을 부드럽게 하기 위해서 그레디언트 텐서를 L2 norm(텐서에 있는 값을 제곱한 합의 제곱근)으로 나눠서 정규화해서 입력 이미지에 적용할 수정량의 크기를 항상 일정 범위 안에 놓는다.


```python
@tf.function # 속도 향상 위해 데코레이터 사용
def gradient_ascent_step(image,filter_index,learning_rate):
  with tf.GradientTape() as tape:
    tape.watch(image) # 이미지는 텐서플로 변수가 아니기 때문에 명시적으로 지정

    # 현재 이미지가 필터를 얼마나 활성화하는지 나타내는 스칼라함수 계산
    loss = comput_loss(image,filter_index) 
  grads = tape.gradient(loss,image) # 이미지에 대한 손실의 그레디언트 계산
  grads = tf.math.l2_normalize(grads) # 그레디언트 정규화 트릭 적용
  image += learning_rate * grads # 필터를 더 강하게 활성화시키는 방향으로 이미지 이동
  return image # 반복 루프에서 이 스텝을 실행할 수 있도록 업데이트된 이미지 반환
  
```

이제 층 이름과 필터 인덱스를 입력으로 받고, 지정된 필터의 활성화를 최대화하는 패턴을 나타내는 텐서를 반환하는 파이썬 함수를 만든다.


```python
img_width = 200
img_height = 200
def generate_filter_pattern(filter_index):
  iterations = 30 # 경사 상승법 적용 횟수
  learning_rate = 10. # 학습률

  # 랜덤한 값으로 이미지 텐서 초기화
  image = tf.random.uniform(
      minval = 0.4,
      maxval = 0.6,
      shape = (1,img_width,img_height,3))
  
  # 손실함수를 최대화하도록 이미지 텐서 값을 반복적으로 업데이트
  for i in range(iterations):
    image = gradient_ascent_step(image,filter_index,learning_rate)
  
  return image[0].numpy()
```

결과 이미지 텐서는 (200,200,3) 크기의 부동 소수점 텐서이다. 이 텐서값은 [0,255] 사이의 정수가 아니기 때문에 출력 가능한 이미지로 변환하기 위해 후처리를 다음과 같이 해준다.


```python
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :] # 부수 효과 피하기 위해 경계 픽셀 제외
    return image
```

이제 함수를 실행해보자.


```python
plt.axis('off')
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))
plt.show()
```




    
![predown](https://user-images.githubusercontent.com/77332628/200705593-197c6283-ac36-4f6b-901c-d4649930b3a0.png)

block3_sepconv1 층에 있는 세 번째 필터는 물이나 털 같은 수평 패턴에 반응하는 것으로 보인다.

다음은 굉장히 흥미로운 부분이다. 층의 모든 필터를 시각화하거나 모델에 있는 모든 층의 필터를 시각화할 수 있다.


```python
# 층에 있는 처음 64개의 필터를 시각화하여 저장
all_images = [] 
for filter_index in range(64):
  print(f'{filter_index}번 필터 처리중')
  image = deprocess_image(
      generate_filter_pattern(filter_index))
  all_images.append(image)

# 필터 시각화를 출력한 빈 이미지 준비
margin = 5 
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# 저장된 필터로 이미지 채우기
for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = image

keras.utils.save_img(
    f"filters_for_layer_{layer_name}.png", stitched_filters)

```

    0번 필터 처리중
    1번 필터 처리중
    2번 필터 처리중
   .
   .
   .
    61번 필터 처리중
    62번 필터 처리중
    63번 필터 처리중


**block2 **
![block2](https://user-images.githubusercontent.com/77332628/200705633-004cb2d3-4287-471a-ba68-ef8f87c65e3e.png)

**block4 **
![block4](https://user-images.githubusercontent.com/77332628/200705640-6b75c942-501c-4e11-8b9c-427cadbb0147.png)

**block8 **
![block8](https://user-images.githubusercontent.com/77332628/200705643-0d7658ab-d387-4424-aa5c-c71edd3cf2f5.png)

이런 필터 시각화를 통해 컨브넷 층이 바라보는 방식을 이해할 수 있다. 컨브넷의 각 층은 필터의 조합으로 입력을 표현할 수 있는 일련의 패턴을 학습한다. 이 컨브넷 필터들은 모델의 층이 깊어질수록 점점 더 복잡해지고 개선된다.

위의 이미지들을 통해서 다음의 사실들을 알 수 있다.


*   모델에 있는 첫 번째 층의 필터는 간단한 대각선 방향의 에지와 색깔을 인코딩한다.
*   block4_sepconv1과 같이 조금 더 나중에 있는 층의 필터는 에지나 색깔의 조합으로 만들어진 간단한 질감을 인코딩한다.
*   더 뒤에 있는 층의 필터는 깃털, 눈 , 나뭇잎 등 자연적인 이미지에서 찾을법한 질감을 점점 닮아가기 시작한다.

### 6.3 클래스 출력의 히트맵 시각화하기

마지막으로 소개할 시각화 기법은 **클래스 활성화 맵(Class Activation Map, CAM) 시각화**이다. 이 방법은 입력 이미지에 대한 클래스 활성화의 히트맵을 만드는데, 클래스 활성화 히트맵은 특정 출력 클래스에 대해 입력 이미지의 모든 위치를 계산한 2D 점수 그리드이다. 클래스에 대해 각 위치가 얼마나 중요한지 알려준다. 이 방법은 이미지의 어느 부분이 컨브넷의 최종 분류 결정에 얼마나 기여하는지 알 수 있기 때문에 분류에 실수가 있는 경우 결정 과정을 **'디버깅'**하는 데 도움이 된다. 또한, 이미지에 특정 물체가 있는 위치를 파악하는 데 사용하기도 한다.

이 글에서 사용할 구체적인 구현은 "Grad-CAM : Visual Explanations from Deep Networks via Gradient-based Localization"에 기술되어 있는 Grad-CAM이다. Grad-CAM을 직관적으로 설명하면, **'입력 이미지가 각 채널을 활성화하는 정도'**에 대한 공간적인 맵을 **'클래스에 대한 각 채널의 중요도'**로 가중치를 부여하여 **'입력 이미지가 클래스를 활성화하는 정도'**에 대한 공간적인 맵을 만드는 것이라고 설명할 수 있다.

사전 훈련된 Xception 모델을 다시 사용해서 이 기법을 시연해보겠다.


```python
from tensorflow import keras
model = keras.applications.xception.Xception(weights='imagenet') # 최상위 밀집 연결 층 포함
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
    91884032/91884032 [==============================] - 5s 0us/step


![elephant](https://user-images.githubusercontent.com/77332628/200705747-4a85f82a-286d-4842-af96-c459fcefe77a.jpg)

위의 어미와 새끼 코끼리 이미지를 적용해본다. Xception 모델은 299x299 크기의 이미지에서 훈련되었고, keras.applications.xception.preprocess_input 함수에 따라 전처리 되었다. 그러므로 코끼리 이미지를 299x299 크기로 변경하고 넘파이 float32 텐서로 바꾼 후 전처리 함수를 적용해서 Xception 모델이 인식할 수 있도록 변환한다.


```python
import numpy as np
# 이미지 다운로드, 경로 설정
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")

def get_img_array(img_path,target_size):

  #299x299 크기의 PIL 이미지 반환
  img = keras.utils.load_img(img_path,target_size=target_size)

  array = keras.utils.img_to_array(img) # (299,299,3) 크기의 float32 넘파이 배열 변환
  array = np.expand_dims(array,axis=0) # (1,299,299,3)의 배치 변환 위해 차원 추가
  array = keras.applications.xception.preprocess_input(array) # 전처리 함수 적용
  return array

img_array = get_img_array(img_path,target_size=(299,299))
```

이제 변환한 이미지에서 사전 훈련된 네트워크를 실행하고 예측 벡터를 이해하기 쉽게 디코딩한다.


```python
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds,top=3)[0])
```

    1/1 [==============================] - 9s 9s/step
    Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
    35363/35363 [==============================] - 0s 0us/step
    [('n02504458', 'African_elephant', 0.8699264), ('n01871265', 'tusker', 0.076968774), ('n02504013', 'Indian_elephant', 0.023537323)]


이 이미지에 대한 상위 3개의 예측 클래스는 다음과 같다.


1.   아프리카 코끼리 : 87% 확률
2.   코끼리 : 8% 확률
3.   인도 코끼리 : 2% 확률

이 네트워크는 이미지가 아프리카 코끼리를 담고 있다고 인식했다.




```python
np.argmax(preds[0]) 
```




    386



예측 벡터에서 최대로 활성화된 항목은 '아프리카 코끼리' 클래스에 대한 것으로 386번 인덱스이다.

이미지에서 가장 아프리카 코끼리와 같은 부위를 시각화하기 위해 Grad-CAM 처리 과정을 구현한다.

먼저, 입력 이미지를 마지막 합성곱 층의 활성화에 매핑하는 모델을 구축한다.


```python
last_conv_layer_name = 'block14_sepconv2_act'
classifier_layer_names = ['avg_pool','predictions']
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs,last_conv_layer.output)
```

그다음 마지막 합성곱 층의 활성화를 최종 클래스 예측하는 모델을 구축한다.


```python
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
  x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input,x)
```

그다음 마지막 합성곱 층의 활성화에 대한 최상위 예측 클래스의 그레디언트를 계산한다.


```python
import tensorflow as tf

with tf.GradientTape() as tape:
  # 마지막 합성곱 층의 활성화를 계산하고 GradientTape로 감시
  last_conv_layer_output = last_conv_layer_model(img_array)
  tape.watch(last_conv_layer_output)

  # 최상위 예측 클래스에 해당하는 활성화 채널 추출
  preds = classifier_model(last_conv_layer_output)
  top_pred_index = tf.argmax(preds[0])
  top_class_channel = preds[:,top_pred_index]

# 마지막 합성곱 층의 출력 특성맵에 대한 최상위 예측 클래스의 그레디언트 계산
grads = tape.gradient(top_class_channel,last_conv_layer_output)
```

이제 그레디언트 텐서를 평균하고 중요도 가중치를 적용해서 클래스 활성화 히트맵을 만들어보자.


```python
'''이 벡터의 각 워소는 어떤 채널에 대한 그레디언트의 평균 강도이고, 
최상위 예측 클래스에 대한 각 채널의 중요도를 정량화'''
pooled_grads = tf.reduce_mean(grads,axis=(0,1,2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]

# 마지막 합성곱 층의 출력에 있는 각 채널에 '채널의 중요도'를 곱함
for i in range(pooled_grads.shape[-1]):
  last_conv_layer_output[:,:,i] *= pooled_grads[i]

# 만들어진 특성 맵을 채널별로 평균하면 클래스 활성화 히트맵이 된다.
heatmap = np.mean(last_conv_layer_output,axis=-1)
```

시각화를 위해 히트맵을 0,1 사이로 정규화한다. 최종 결과는 다음과 같다.


```python
from matplotlib import pyplot as plt
heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
```




    
![heat](https://user-images.githubusercontent.com/77332628/200705886-bfd0c2db-f9e5-4b35-8b8e-c2639def13af.png)

마지막으로 위의 히트맵에 원본 그림을 겹친 이미지를 만들어본다.


```python
import matplotlib.cm as cm

img = keras.utils.load_img(img_path) # 원본 이미지 로드
img = keras.utils.img_to_array(img)

heatmap = np.uint8(255*heatmap) # 히트맵 범위 [0,255]로 조정

# 'jet' 컬러맵을 사용해서 히트맵 색 변경
jet = cm.get_cmap('jet')
jet_colors = jet(np.arange(256))[:,:3]
jet_heatmap = jet_colors[heatmap]

# 새로운 히트맵을 담을 이미지 생성
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1],img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

# 히트맵에 40% 투명도 적용후 원본 이미지와 합침
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)

# 합친 이미지 저장
save_path = 'elephant_cam.jpg'
superimposed_img.save(save_path)
```

![elephant_cam](https://user-images.githubusercontent.com/77332628/200705957-4670e7ae-15f7-436b-a832-b39baca57c62.jpg)

이 시각화 기법은 2가지 중요한 질문에 대한 답을 준다

* 왜 네트워크가 이 이미지에 아프리카 코끼리가 있다고 생각하는가?
* 아프리카 코끼리가 사진 어디에 있는가?

위의 결과물 사진에서 볼 수 있듯이, 새끼 코끼리의 귀가 강하게 활성화 되었는데, 이를 통해서 아마 네트워크가 아프리카 코끼리와 인도 코끼리를 구분한 것으로 보인다.



[<케라스 창시자에게 배우는 딥러닝 개정 2판>(길벗, 2022)을 학습하고 개인 학습용으로 정리한 내용입니다.] 출처: 프랑소와 숄레 지음, ⌜케라스 창시자에게 배우는 딥러닝 개정2판⌟, 박해선 옮김, 길벗, 2022 도서보기: https://www.gilbut.co.kr/book/view?bookcode=BN003496
