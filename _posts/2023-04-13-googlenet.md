---
toc : true
title : '[IC/개념] 이미지 분류 - GoogLeNet ➰'
layout : single
toc: true
toc_sticky: true
categories:
  - cv-imageclassification
---


## GoogLeNet 논문 리뷰

이번 글에서는 [<U>GoogleNet 논문</U>](https://arxiv.org/pdf/1409.4842.pdf)(Going deeper with convolutions)을 리뷰해보도록 한다. GoogLeNet에 대해서 [<U> Inception v3 모델 리뷰</U>](https://hamin-chang.github.io/cv-imageclassification/inception/)글에서 간단하게 다룬 적 있지만 이번 글에서 자세히 알아본다.


### 0. Introduction
GoogLeNet의 코드네임인 Inception이라는 이름은 NIN(Network in Network)라는 논문에서 유래했으며, 영화 Inception의 대사 "we need to go deeper"에서 착안 했는데, 여기서 'deep'은 두 가지 의미를 가진다고 한다.

1. 'Inception module'의 형태로 새로운 차원의 구조 도입
2. 네트워크의 '깊이'가 증가했다는 직접적인 의미

대부분의 실험에서 모델은 1.5억번의 연산을 넘지 않도록 설계했으며, 이는 단순한 학술적인 호기심으로 끝나는 것이 아닌, 실제에서도 사용하기 위함이라고 한다.

Deep neural network의 성능을 향상시키기 위해서는 depth, number of levels, width, number of units를 늘리면 된다. 하지만 이러한 접근에는 두가지 단점이 존재하는데,

1. **모델의 사이즈가 커지면 overfitting**을 야기한다.
2. 균등하게 증가된 network는 컴퓨팅 자원을 더 많이 잡아먹는다. 

이러한 **문제들을 해결하기 위한 근본적인 방법은 fc layer를 sparsely connected한 구조로 변경**하는 것이다. 즉, 각 뉴런이 일부분의 다른 뉴런들과만 연결되어 있는 구조로 변경하는 것이다.

![2](https://user-images.githubusercontent.com/77332628/231642348-05e832c8-64e7-40c2-b887-d48538065747.png)

(왼쪽이 **sparse**, 오른쪽이 **dense** connection)

Arora et al.에 따르면 아주 거대한 sparse deep neural network로 데이터셋의 확률분포가 표현 가능하다면, 최적의 네트워크는 마지막 layer의 activation의 correlation statistics를 분석하고 다음 이미지처럼 **highly correlated output으로 묶으면서 구성**할 수 있다고 한다. 

![3](https://user-images.githubusercontent.com/77332628/231642350-2b285f34-7fd8-419a-9152-7435e929593d.png)

Inception은 arora et al.의 sparse network를 사용한 component로 구현해서 arora et al.의 가설을 따라 가는지 확인하고자 하는 case study로 시작되었다고 한다. 두번의 반복 만에 NiN을 기반으로 한 구조보다 더 나은 성능의 구조를 찾아냈다고 한다.

### 1. GoogLeNet Architecture Components

#### 1.1 1x1 conv filter
본격적으로 GoogLeNet의 architecture를 알아보기 전에, 1x1 conv에 대해서 알아보자. 다음 이미지의 GoogLeNet 구조를 보면 1x1 conv filter가 많이 사용된 것을 확인할 수 있다.

![1](https://user-images.githubusercontent.com/77332628/231642342-a29d2c50-2aa3-4fb0-b528-026cd8c45fad.png)

1x1 conv는 두가지 역할을 한다.

1. **컴퓨터 병목현상을 방지하기 위한 차원축소**
2. 너트워크 크기 제한 : 성능은 크게 저하시키지 않으면서, **네트워크의 depth와 width를 증가**시킨다.

1번 역할을 자세히 알아보자.

예를 들어, 14x14x480 feature map이 있다고 하고 이를 48개의 5x5x480의 filter로 convolution해주면 14x14x48 feature map이 생성되는데, 이때 필요한 연산 횟수는 (14x14x48)x(5x5x480) = 112.9M이 된다.

이번에는 14x14x480 feature map을 16개의 1x1x480 filter로 convolution해줘서 feature map의 개수를 줄여서 14x14x16의 feature map으로 480장의 feature map을 16장으로 줄여보자. 그리고 나서 14x14x16 feature map을 48개의 5x5x16의 filter로 convolution 해주면 14x14x48의 feature map이 생성되고 이는 위의 output과 동일하다. 이때의 연산 횟수는 14x14x16)(1x1x480)+(14x14x48)(5x5x16) = 약 5.3M으로 1x1 conv를 적용해서 훨씬 더 적은 연산량을 가짐을 알 수 있다. 연산량을 줄일 수 있다는 것은 네트워크를 더 깊이 만들 수 있다는 점에서 중요하다.



#### 1.2 Inception Module

이제 본격적으로 Inception 구조에 대해 알아보자. Inception 구조의 주요 아이디어는 CNN에서 각 요소를 최적의 local sparse structure로 근사하고, 이를 dense component로 바꾸는 방법을 찾는 것이다. 다시 말해 최적의 local 구성 요소를 찾고 이를 공간적으로 반복하면 된다. 쉽게 말하자면 위의 이미지처럼 **sparse 매트릭스를 서로 clustering하여 상대적으로 dense한 submatrix**를 만든다는 것이다.

이때, 이전 layer의 각 유닛이 입력 이미지의 특정 부분에 해당된다고 가정했는데, **입력 이미지와 가까운 낮은 layer에서는** 특정 부분에 Correlated unit 들이 집중되어 있다. 이는 **단일 지역에 많은 클러스터들이 집중된다는 뜻이기에 1x1 conv로 처리**할 수 있다.

![4](https://user-images.githubusercontent.com/77332628/231642355-8426568f-a58f-4da5-86f7-8484badc614e.png)

하지만 위 이미지에서처럼 몇몇 위치에서는 **좀 더 넓은 영역의 convolutional filter가 있어야 correlated unit의 비율을 높일 수 있는 상황이 나타날 수도 있기 때문에 feature map을 효과적으로 추출할 수 있도록 1x1, 3x3, 5x5 conv 연산을 병렬적**으로 수행한다. 또한 Po**oling이 CNN의 성공에 있어 필수 요소**이기 때문에 pooling도 추가했다고 한다.

![5](https://user-images.githubusercontent.com/77332628/231642357-3749d8a6-22ae-44f6-b1c5-57c309f2f121.jpeg)

1x1, 3x3, 5x5 conv filter의 수는 모델이 깊어짐에 따라 달라지는데, 만약 위 이미지처럼 높은 layer에서만 포착될 수 있는 높은 추상적 개념의 특징이 있다면, **공간적 집중도가 감소한다는 것을 의미하기 때문에 높은 layer를 향해 모델이 깊어질수록 3x3과 5x5 conv filter의 수도 늘어나**야 한다.

하지만 이 부분에서 큰 문제가 발생한다. 3**x3과 5x5 conv filter가 늘어날수록 연산량이 많아**지는데 입력 feature map의 크기가 크거나 5x5 conv filter의 수가 많아지면 연산량은 더욱 증가하게 된다.

이러한 문**제를 해결하기 위해서 1x1 conv filter를 사용**한다.


![6](https://user-images.githubusercontent.com/77332628/231642359-1449f141-576f-4032-8e7c-387f50d4aa3c.png)

위 이미지의 (b)가 수정한 inception module인데, **3x3과 5x5 conv filter 앞에 1x1 conv filter를 두어 차원을 줄여서 여러 scale을 확보하면서도 연산량을 낮출** 수 있다. 추가적으로, conv 연산 이후에 추가되는 **ReLU를 통해 비선형성을 추가**할 수 있다고 한다.

(b)의 inception module을 사용하면 두가지 효과를 볼 수 있다고 한다.

1. **연산량 문제 없이 각 단계에서 유닛 수를 상당히 증가**시킬 수 있다. 이는 차원 축소를 통해 다음 layer의 input 수를 조절할 수 있기 때문이다.

2. **Visual 정보가 다양한 scale로 처리**되고, 다음 layer는 동시에 서로 다른 layer에서 특징을 추출할 수 있다. **1x1, 3x3, 5x5 conv 연산을 통해 다양한 특징을 추출**할 수 있기 때문이다.

Google 팀에서는 효**율적인 메모리 사용을 위해서 낮은 layer에서는 기본적인 CNN 모델을 적용**하고, 높**은 layer에서 Inception module을 사용**하는 것이 좋다고 한다.

### 2. GoogLeNet

이제 Inception module이 적용된 전체 GoogLeNet의 구조에 대해 알아보자. GoogLeNet의 전체적인 구조는 다음 이미지와 같다.

![7](https://user-images.githubusercontent.com/77332628/231642361-7aebd397-b1cd-443a-ab46-a5b720e8261c.png)

참고로 GoogLeNet이라는 이름은 LeNet-5를 오마주해서 만들어졌다고 한다. 또한 **GoogLeNet은 Inception 모듈들의 한 형태(incarnation)**이다. 

Inception module 내부를 포함한 **모든 conv layer에는 ReLU가 적용**되어 있다. 또한 receptive field의 크기는 224x224로 RGB 컬러 채널을 가지며, mean subtraction을 적용한다.

![8](https://user-images.githubusercontent.com/77332628/231642364-0e48293f-d294-4c30-bb0e-f490667c3ce0.png)

위 표에서 **#3x3 reduce와 #5x5 reduce는 3x3과 5x5 conv layer 앞에 사용되는 1x1 conv filter의 채널 수**를 의미하고 **pool proj열은 max pooling layer 뒤에 오는 1x1 conv filter의 채널 수**를 의미한다. 또한 **모든 reduction 및 projection layer에 ReLU가 사용**된다.

GoogLeNet을 4가지 파트로 나누어 살펴보자.

**Part 1** : 입력 이미지와 가까운 낮은 layer

![9](https://user-images.githubusercontent.com/77332628/231642368-854689ff-0f7e-4e3c-bc3b-c003fb70a5fa.png)

위에서 언급했듯이 효율적인 메모리 사용을 위해 낮은 layer에서는 기본적인 CNN 모델을 적용한다. 따라서 Inception module이 적용되지 않을 것을 볼 수 있다.

**Part 2** : Inception module

![10](https://user-images.githubusercontent.com/77332628/231642371-de4b35cb-24c6-4884-803c-34462ddf456b.png) 

Inception module로서 다양한 feature를 추출하기 위해 1x1, 3x3, 5x5 conv layer가 병렬적으로 연산을 수행하고 있으며, 차원 축소를 통해 연산량을 줄이기 위해 1x1 conv layer가 적용되었다.

**Part 3 : auxiliary classifier**

![11](https://user-images.githubusercontent.com/77332628/231642372-5fc38b25-fdcf-4d09-8688-0f49d53590a5.png)

**모델의 깊이가 굉장히 깊을 경우, 연산의 기울기가 0으로 수렴하는 gradient vanishing 문제가 발생**할 수 있다. 이때, 상대적으로 얕은 신경망의 강한 성능을 통해 신경망의 중간 layer에서 생성된 특징이 매우 차별적이라는 것을 알 수 있다. 따라서 **중간 layer에 auxiliary classifier를 추가해서, 연산 중간중간에 결과를 출력해서 추가적인 역전파**를 일어켜서 **gradient가 전달**될 수 있게끔 할 뿐만 아니라, **정규화 효과**도 나타나도록 하였다.

auxiliary classifier가 보조 classifier인만큼, **지나치게 영향을 주는 것을 막기 위해 auxiliary classifier의 loss에 0.3을 곱**하고, **실제 테스트 시에는 auxiliary classifier를 사용하지 않고**, 모델 제일 끝단의 softmax만을 사용하였다.

다음 이미지에서 박스 친 부분이 모델에서의 auxiliary classifier이다.

![12](https://user-images.githubusercontent.com/77332628/231642373-463847cd-f2c6-4c67-8b0e-cdbd525d0550.png)

Part 4 : main classifier

예측 결과가 나오는 모델의 끝 부분이다. 이 부분에서 최종 classifier 이전에 average pooling layer를 사용하는데, 이는 **GAP(Global Average Pooling)를 적용해서 이전 layer에서 추출된 feature map을 각각 평균 낸 것을 이어 1차원 벡터로** 만들어준다. **1차원 벡터로 만들어야 최종적으로 이미지 분류를 위한 softmax layer와 연결할 수** 있기 때문이다.

![13](https://user-images.githubusercontent.com/77332628/231642374-0b4f9573-c536-4b4f-a0da-b10437035cc1.png)

fc layer를 사용하면 가중치가 7x7x1024x1024 = 51.3M개가 사용되지만 **GAP를 사용하면 가중치가 전혀 필요없고, GAP를 적용하면 fine tuning하기 더 용이**하다고 한다.



### 3. Training Methodology & 실험 결과

GoogLeNet은 다음과 같이 학습되었다.

* Optimizer : **Asynchronous SGD with 0.9 momentum**
* lr schedule : **8 epochs 마다 4%씩 감소**
* 입력 이미지의 **가로,세로 비율은 3:4와 4:3 사이로 유지**하며 **본래 사이즈의 8% ~ 100%가 포함되도록 다양한 크기의 patch**를 사용하였다. 
* **photometric distortions를 통해 학습 데이터 증강**

ILSVRC 2014 classification task의 결과는 다음과 같다.

![14](https://user-images.githubusercontent.com/77332628/231642376-9c381e94-9cb9-4f1c-b71d-9c9255a41312.png)

Ensemble을 적용하고 crop을 최대한 많이 적용할 수록 좋은 성능을 낸 것을 볼 수 있다. ensemble한 7개의 모델들은 sampling 방법의 차이와 입력 이미지의 순서에만 차이를 두었다고 한다. 

### 4. Conclusion

**Inception 구조는 sparse 구조를 Dense 구조로 근사화하여 성능을 개선**했다. 이는 기존의 CNN 성능을 높이기 위한 방법과는 다른 새로운 방법이었으며, **연산량의 증가량은 약간만 증가하고 성능은 대폭 상승**시켰다는데에 의미가 있다.

출처 및 참고문헌:

1. https://arxiv.org/pdf/1409.4842.pdf
2. https://jjuon.tistory.com/26
3. https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
4. https://velog.io/@twinjuy/GoogLeNet-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0Going-deeper-with-convolutions

