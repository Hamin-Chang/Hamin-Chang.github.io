---
layout: single
title:  "[IC/개념] 이미지 분류 - Inception-v2,3 ➰"
toc: true
toc_sticky: true
categories:
  - cv-imageclassification
---



## Inception-v2,3 논문 리뷰


### 1. Inception-v2, 2016
GoogLeNet의 후속 모델인 Inception-v2의 핵심 요소는 크게 3가지로 나눌 수 있다. 

#### 1.1 Conv Filter Factorization
GoogLeNet(Inception-v1) 모델은 VGG, AlexNet에 비해서는 parameter수가 굉장히 적지만 여전히 많은 연산 비용이 든다. Inception-v2는 연산 복잡도를 줄이기 위해 **Conv Filter Factorization** 기법을 사용한다. Conv Filter Factorization은 5x5 conv를 3x3 conv 2개로 분해하는 것처럼 더 작은 합성곱으로 분해하는 것이다. 3x3 conv 보다 큰 필터는 언제든지 3x3 conv로 분해하는 것이 좋다는 연구 결과가 있다.

실제로 Inception-v2에서는 아래 이미지처럼 inception module에서 5x5 conv 부분을 두개의 3x3 conv로 분해한다.

* 기존 inception module

![incep5](https://user-images.githubusercontent.com/77332628/202139032-0784ae75-c81a-4848-a526-ace166902e35.png)


* 작은 합성곱 필터로 분해한 inception module

![incep6](https://user-images.githubusercontent.com/77332628/202139033-9bf83fcc-8db1-44bd-94bf-7daf54c22f3f.png)


#### 1.2 비대칭 합성곱 분해 
그렇다면 3x3 conv 필터를 더 작은 conv 필터로 분해할 수 있을까? Inception-v2 논문의 저자에 따르면 2x2 conv로 분해하는 것보다 nx1 비대칭 conv로 분해하는 것이 더 효과적이라고 한다. 예를 들어 아래 이미지처럼 3x3 conv를 1x3 conv와 3x1 conv로 분해하는 것이다. 

![incep7](https://user-images.githubusercontent.com/77332628/202139037-2fd95781-3ec8-4f33-a844-f6c096de1399.png)


feature map 사이즈가 17~20 사이일 때 효과적이기 때문에 Inception-v2 모델에서는 inception module에서 nXn conv를 nx1 과 1xn conv로 대체하면 다음과 같이 된다.

![incep8](https://user-images.githubusercontent.com/77332628/202139042-a508f0a7-cbaa-4422-be14-8585d46cfb58.png)


#### 1.3 보조 분류기(Auxiliary Classifiers)의 역할의 변화
GoogLeNet에서 보조 분류기를 사용하면 신경망이 수렴하는데 도움이 된다고 했었는데, 실험 결과 별다른 성능 향상에 도움이 안된다는 것이 밝혀졌다. 하지만 Auxiliary classifiers에 Dropout이나 batch normalization을 적용했을 때 메인 분류기의 성능히 향상된 것으로 보아서 auxiliary classifier는 성능 향상보다 정규화 효과가 있을 것이라고 추측한다. 그래서 Inception-v2에서는 1개의 보조 분류기만 사용했다.

* 보조 분류기에 배치 정규화를 적용하니 0.4%의 정확도가 개선되었다는 것을 설명하는 이미지

![incep9](https://user-images.githubusercontent.com/77332628/202139047-8ebf09f5-3ab3-43c5-b862-e6c47dcc23bb.png)


#### 1.4 Grid Size Reduction
일반적인 CNN 신경망은 feature map의 크기를 줄이기 위해 pooling 연산을 사용하는데, pooling을 하면 나타나는 신경망의 표현력이 감소되는 representational bottleneck을 피하기 위해 필터 수를 증가시킨다. 이는 신경망의 표현력을 포기하거나 많은 연산 비용을 감수하는 선택을 해야한다. 예를 들어, dxd 크기를 가진 k개의 feature map은 pooling layer를 거쳐서 (d/2)x(d/2) 크기의 2k feature map이 된다. 연산량을 계산하면, 전자는 $2d^2k^2$가 되고, 후자는 $2(d/2)^2k^2$가 된다. 다음 이미지에서 왼쪽은 연산량이 낮은 대신 표현력이 감소하고, 오른쪽은 표현력이 감소하지 않는 대신 연산량이 늘어난다.

![11](https://user-images.githubusercontent.com/77332628/233845625-3bda35d5-39f9-4ab5-b3c4-c607d8bfc4cf.png)
 

하지만 Inception-v2에서는 표현력을 감소시키지 않고 연산량을 감소시키는 방법을 사용한다. 

![incep10](https://user-images.githubusercontent.com/77332628/202139050-7069e2b9-1bc4-4b86-99b7-5d80496efdcd.png)


위의 이미지와 같이 stride=2인 pooling layer와 conv layer를 병렬로 상용하고 둘을 연결하는 방법을 사용해서 표현력을 감소시키지 않으면서 연산량을 감소시킨다. 최종적으로는 다음 이미지와 같은 방식을 사용했다고 한다.

![incep11](https://user-images.githubusercontent.com/77332628/202139055-5cfff798-0bce-4ac0-bbee-a1130f3ff3bf.png)


#### 1.5 최종 Inception-v2 모델
2.1 ~ 2.4까지 소개된 아이디어들이 최종 적용된 Inception-v2의 architecture 표는 다음과 같다. 7.2.1의 Factorization 방법을 사용해서 3x3 conv 연산 3개로 대체가 되었고, 나머지 아이디어들이 차례대로 적용되었다.

![incep12](https://user-images.githubusercontent.com/77332628/202139058-d2db380e-f5a3-4a82-810c-58bd63f1f5f3.png)

* 첫번째 inception module

![12](https://user-images.githubusercontent.com/77332628/233845627-cb1f97bd-6870-452c-a557-c87b91926605.png)

* 두번째 inception module

![13](https://user-images.githubusercontent.com/77332628/233845628-2a639417-4d69-446e-b094-59a03b77990f.png)

* 세번째 inception module

![14](https://user-images.githubusercontent.com/77332628/233845629-47bc8ed5-5968-48ff-a187-5385a495538d.png)


### 2. Inception-v3, 2016
Inception-v3는 Inception-v2의 architecture는 그대로 가져가고, 다음의 몇가지 기법들을 적용해서 최고의 성능을 내는 모델이다.

#### 2.1 Model Regularization via Label Smoothing
**Label Smoothing** 기법은 정규화 테크닉 하나로 간단하면서도 모델의 일반화 성능을 높여서 주목 받은 기법이다. Label Smoothing에 관해서 간단히 설명하자면 만약 기존 label이 [0,1,0,0]이라면 레이블 스무딩을 적용하면 [0.025,0.925,0.025,0.025]가 되는데, 이는 정답에 대한 확신을 감소시켜서 모델의 일반화 성능을 향상시킨다. 다음은 논문에 나와있는 Label smoothing 수식이다.

![10](https://user-images.githubusercontent.com/77332628/233845623-540aa720-e9f6-4f6e-9ebc-4aa44d3c2a3a.png)

#### 2.2 기타 기법들
* optimizer를 Momentum optimizer 에서 RMSProp optimizer로 변경
* Auxiliary classifier의 FC layer에 Batch Normalization 추가
* Factorized 7x7 conv filters

즉, Inception-v3는 Inception-v2에서 BN-auxiliary + RMSProp + Label Smoothing + Factorized 7x7를 전부 적용한 모델이다.

다음은 Inception-v3의 architecture이다.

![incep13](https://user-images.githubusercontent.com/77332628/202139061-9fa97c46-e11e-4cef-86aa-92d02c020ad8.png)
 

출처 및 참고문헌 :

1. https://arxiv.org/pdf/1512.00567.pdf
2. https://deep-learning-study.tistory.com/517
