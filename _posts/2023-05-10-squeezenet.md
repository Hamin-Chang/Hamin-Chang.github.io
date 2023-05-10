---
layout: single
title:  "[IC/개념] 이미지 분류 - SqueezeNet 🗜️"
toc: true
toc_sticky: true
categories:
  - cv-imageclassification
---

## SqueezeNet 논문리뷰

이번 글에서는 [**<U>SqeezeNet 논문</U>**](https://arxiv.org/pdf/1602.07360.pdf)(AlexNet-level accuracy with 50x fewer parameters and < 0.5MB model size)를 리뷰한다.

### 0. Introduction & Related Works

CNN의 구조가 복잡해지고 성능이 좋아지면서 모델이 요구하는 메모리가 증가했다. Squeeze라는 단어에서 알 수 있듯이 해당 SqueezeNet은 모델의 크기 압축에 초점을 둔 모델이다. CNN 모델의 크기를 줄이는 것은 다음의 이점들이 있다.

* 분산 학습시 서버 간에 주고 받아야 할 데이터가 줄어듦
* 자율주행을 위해 클라우드에서 모델을 불러올 때, 작은 대역폭 요구 가능
* FPGA나 제한된 메모리를 요하는 하드웨어에 모델을 올릴 수 있음

SqueezeNet은 기존 모델과 거의 같은 성능을 유지하면서 파라미터 수를 줄이는 것에 집중했다. SqueezeNet은 ImageNet 데이터셋에서의 정확도가 AlexNet과 비슷한 수준을 유지하지만, AlexNet보다 50배나 적은 파라미터 수를 가진다. 용량 또한 0.5MB이하로 줄일 수 있었다고 한다.

**Related Works**

Mirco Architecture

쉽게 말해서 Inception module, Residual Block, Dense Block등과 같이 모듈화된 구조를 사용하는 CNN 모델을 뜻한다.

Macro Architecture

모듈들이 쌓여서 이루는 모델 그자체를 의미한다.



### 1. Architecture Design Strategies

SqueezeNet 논문에서는 파라미터 수를 줄이기 위해 다음 세가지 전략을 사용했다.

1) **3x3 Filter -> 1x1 Filter 대체** : 파라미터 수 9배 절약

2) **3x3 Filter의 input channel 수 감소** : 

Conv layer의 파라미터 수 계산 공식은 다음과 같다. 

$(kernel)$x $(kernel)$ x $(number$ $of$ $input$ $channel)$ x $(number$ $of$ $filter)$ 

input channel을 줄여서 Conv layer의 파라미터 수를 줄인다.

3) **Conv layer가 큰 넓이의 activation map**을 갖도록 **Downsample을 나중에 수행** :

일반적으로 CNN은 Pooling을 통해서 downsampling을 해가면서 이미지의 정보를 압축해나간다. 하지만 큰 activation map을 가지고 있을수록 정보 압축에 의한 손실을 줄일 수 있다. 따라서 이미지 정보의 손실을 줄이기 위해 네트워크 후반부에 downsampling 수행한다.

**1),2)는 정확도를 유지하면서 파라미터 수를 줄이기 위한** 전략이고, **3)은 제한된 파라미터 내에서 정확도를 최대한 높이기 위한** 전략이다.




### 2. Fire Module

SqueezeNet은 Fire Module로 구성된 모델이다. Fire Module의 구조는 다음과 같다.

![1](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/db80b235-11df-4951-a3f2-3d517efe3f80)

Fire Module은 Squeeze layer와 expand layer 두가지 layer로 이루어져있다. Squeeze layer는 설계전략 1)을 적용해서 1x1 Conv filter로만 구성되어 있다. Expand layer는 1x1와 3x3 filter를 함께 사용한다. Fire Module는 다음의 세가지 하이퍼파라미터를 가지고 있다.

* $s_{1x1}$ : Squeeze layer에서 1x1 conv filter의 개수
* $e_{1x1}$ : Expand layer에서 1x1 conv filter의 개수
* $3_{3x3}$ : Expand layer에서 1x1 conv filter의 개수

Fire module을 설계할 때 설계전략 2)를 적용하기 위해서 하이퍼파라미터를 다음과 같이 설정한다.

$s_{1x1}<(e_{1x1}) + (e_{3x3})$

이 수식에 의해 squeeze layer의 필터 수가 expand보다 크지 않도록 제한해서 전체 필터 개수를 제한한다.



### 3. SqueezeNet Architecture

전체적인 구조는 다음과 같다. 

![2](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/a28ba323-b91d-4ff6-9e6b-91151e354f52)

**먼저 Squeeze layer는 1x1 conv filter를 통해서 채널을 압축하고 expand layer는 1x1 conv filter와 3x3 conv filter를 통해서 다시 팽창**시키는 역할을 하게 된다. activation으로는 주로 ReLU를 사용한다.

위의 이미지 중 왼쪽 이미지처럼 예를 들어 input으로 128 channel이 들어오면 1x1 conv filter를 통해서 16 channel로 줄였다가 다시 1x1 conv filter로 64 channel, 3x3 conv filter로 64 channel을 만들고 이 둘을 concatenate해서 다시 128 channel의 output 값을 만든다.

논문의 SqueezeNet의 구조는 다음과 같다.

![3](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/c928f74f-12f5-4604-b844-bbe10ae96e4b)

왼쪽이 가장 기본적인 구조인데, SqueezeNet은 먼저 1개의 Conv layer로 시작하고 그 뒤로 8개의 (fire 2~9) fire module이 이어지는 구조다. 마지막으로 Conv layer와 softmax를 거쳐서 output을 출력한다. 

또한 Max pooling을 통해서 해상도를 줄여나가는데, stride=2로 설정하고 첫번째 conv layer, fire4, fire8뒤에 위치시킨다. 이러한 **Max pooling의 배치는 설계전략 3)을 적용**한 것이다. 그리고 마지막 conv layer뒤에는 average pooling을 적용해서  output size를 조절한다.

위 이미지에서 **가운데 구조는 기본 구조에 simple bypass(skip connection)을 추가**한 것이고, **오른쪽은 기본 구조에 complex bypass(bypass에 1x1 conv layer 추가)를 추가**한 것이다. bypass를 적용해서 위 이미지에서 볼 수 있듯이 fire2의 output과 fire3의 output이 더해져서 fire4의 input이 된다.

**Bypass를 추가하는 이유는 Fire module내에서 bottleneck 문제가 발생하기 때문**이다. Squeeze layer가 가지고 있는 파라미터 수가 적기 때문에 적은 양의 정보가 Squeeze layer를 통과한다고 생각할 수 있다. **이처럼 차원 감소는 모델을 가볍게 해주지만 정보손실이 발생하는데, bypass를 추가해서 이를 보완**한다.

하지만 실험 결과 **simple bypass를 적용한 것이 complex bypass를 적용한 것보다 좋은 성능**을 냈다고 한다. 심지어 **simple bypass는 파라미터 수가 늘어나지 않지만, complex bypass는 1x1 conv filter 때문에 파라미터 수도 증가**한다.

![4](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/f3949e42-1a8f-4b04-8d37-9fb2c1ae417e)




### 4. SqueezeNet evaluation

SqueezeNet을 다른 model compression 기법들과 비교해본 결과는 다음과 같다. (AlexNet에 model compression을 적용)

![5](https://github.com/Hamin-Chang/Hamin-Chang.github.io/assets/77332628/184e4f6c-65c1-4b99-81f3-423f92d4cd33)

결과를 보면 SqueezeNet 기본구조 만으로도 AlexNet보다 파라미터 수가 50배나 줄었으며, SqueezeNet에 Deep compression을 적용하면 파라미터 수를 510배까지 줄일 수 있다. 여기서 주목할 점은 다른 model compression기법보다 훨씬 크게 파라미터 수를 줄이기도 했지만, **원래 AlexNet에서 성능 하락은 없다**는 것이다. 심지어 Top-1 accuracy는 오히려 상승했다.

출처 및 참고문헌 :

1. https://arxiv.org/abs/1602.07360
2. https://velog.io/@woojinn8/LightWeight-Deep-Learning-4.-SqueezeNet
3. https://velog.io/@twinjuy/SqueezeNet-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
4. https://imlim0813.tistory.com/38
5. https://deep-learning-study.tistory.com/520
