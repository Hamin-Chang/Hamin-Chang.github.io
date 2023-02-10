---
title : '[DL/CV] 객체 탐지 - RetinaNet ⚖️'
layout : single
toc : true
toc: true
toc_sticky: true
categories:
  - cv-objectdetection
---

## RetinaNet 논문 (Focal Loss for Dense Object Detection) 읽어보기


이번 글에서는 RetinaNet 논문 ([<U>Focal Loss for Dense Object Detection</U>](https://arxiv.org/pdf/1708.02002.pdf))을 리뷰해보도록 하겠다.

### 0. Class Imbalance 문제
Object detection 모델은 이미지 내의 객체의 영역을 추정하고 IoU threshold에 따라 positive/negative sample로 구분하고 이를 활용해서 학습한다. 하지만 일반적으로 positive sample(객체 영역)은 negative sample(배경 영역)에 비해 매우 적다. 이번 글의 주목적은 이로 인해서 발생하는 positive/negative sample 사이에 큰 차이가 생겨 발생하는 **class imbalance** 문제를 해결하는 것이다.

class imbalance는 학습시 두가지 문제를 야기한다.
* 대부분의 sample이 easy negative, 즉 모델이 class를 예측하기 쉬운 sample이기 때문에 학습이 비효율적으로 진행된다.
* easy negative의 수가 압도적으로 많기 때문에 학습에 미치는 영향력이 커져서 모델의 성능이 하락한다.

Two-stage detector 계열의 모델은 class imbalance 문제를 두가지 측면에서의 해결책으로 해결했다. 
* two-stage cascade : region proposals를 추려내는 방법으로 대부분의 background sample을 걸러내는 방법. ex) selective search, edgeboxes, deepmask, RPN
* sampleing heuristic : positive/negative sample의 비율을 적절하게 유지하는 방법. ex) hard negative mining, OHEM

하지만 위의 두가지 방법을 one-stage detector에 적용하기는 어렵다. one-stage detector는 region proposal 과정 없이 전체 이미지를 dense하게 순회하면서 sampling하는 dense sampling을 수행하기 때문에 two-stage detector에 비해 훨씬 더 많은 후보 영역을 생성한다. 즉 class imbalance 문제가 더 심각하다는 것이다. 따라서 이번에 다루는 논문에서 one-stage detector에 적용할 수 있는 새로운 loss function을 제시한다.



### 1. Focal Loss

**Focal loss**는 one-stage detector에서 객체와 배경 class 사이에 발생하는 극단적인 class imbalacne 문제를 해결하는데 사용되는 loss function이다. Focal loss는 이진 분류에서 사용되는 Cross Entropy(이하 CE) loss function으로부터 비롯된다.

1) **CE loss**

이진 분류에서 사용되는 CE loss는 다음과 같다.

이미지1

* $y∈[1,-1]$ : ground truth class
* $p∈[0,1]$ : 모델이 $y=1$이라고 예측한 확률
 
CE의 문제점은 모든 sample에 대한 예측 결과를 동등하게 가중치를 둔다는 점이다. 이로 인해서 쉽게 분류될 수 있는 sample도 작지 않은 loss를 유발하게 된다. 많은 수의 easy example의 loss가 더해지면 rare한 class를 압도하게 돼서 학습이 제대로 이뤄지지 않는다.

2) **Balanced CE loss**

이러한 문제를 해결하기 위해 가중치 파라미터 $α∈[0,1]$를 곱해준 Balanced CE가 등장한다. 

이미지2

$y=1$일 때 $α$를 곱해주고, $y=-1$일 때 $1-α$를 곱해준다. Balanced CEsms positive/negative sample 사이의 균형은 잡아주지만, easy/hard sample에 대해서는 균형을 잡지 못한다. 논문에서는 Balanced CE를 baseline 손실함수로 잡고 실험을 진행한다.

3) **Focal Loss**

이미지3

Focal loss는 easy example을 down-weight해서 hard negative sample에 집중해서 학습하는 loss function이다. Focal loss는 **modulating factor** $(1-p_t)^γ$와 **tunable focusing parameter**를 CE에 추가한 형태를 가진다. 

이미지4

서로 다른 $γ∈[0,5]$값에 따른 loss는 위 이미지를 보면 알 수 있다. 위 그래프에서 파란색 선은 일반 CE를 나타낸다. 파란색은 경사가 완만하며 $p_t$가 높은 sample과 낮은 sample 사이의 차이가 크지 않다. 반면 Focal loss는 focusing parameter가 커질수록 $p_t$가 높은 sample과 낮은 sample 사이의 차이가 커진다는 것을 볼 수 있다.

이미지5

즉, $y=1$인 class임에도 $p_t$가 낮은 경우와 $y=-1$인 class임에도 $p_t$가 높은 경우에는 Focal loss가 높게 나오고, 반대의 경우에는 down-weight 되어 loss 값이 낮게 나타난다. 이를 통해 Focal loss의 두가지 특성을 알 수 있다.

1. $p_t$와 moduling factor와의 관계
sample이 잘못 분류되고, $p_t$가 작으면, modulating factor는 1과 가까워지며, loss는 영향 받지 않는다. 하지만 반대로 $p_t$ 값이 크면 modulating factor는 0에 가까워지기 때문에 잘 분류된 sample의 loss는 down-weight된다.

2. focusing parameter $γ$의 역할
focusing parameter $γ$는 easy sample을 down-weight하는 정도를 부드럽게 조정한다. $γ$가 커질수록 modulating factor의 영향력이 커진다. 논문에서는 $γ=2$일 때 가장 좋은 결과를 보였다고 한다.

결과적으로 modulating factor는 easy sample의 기여도를 줄이고 sample이 작은 loss를 받는 범위를 확장시키는 기능을 합니다. 예를 들어 $γ=2, p_t=0.9$일 때 CE에 비해 100배 적은 loss를 가지고, $p_t=0.968$일때는 1000배 적은 loss를 가진다. 이는 잘못 분류된 sample을 수정하는 작업의 중요도를 상승시킴을 의미한다.



### 2. Training RetinaNet

논문에서는 Focal loss를 실험하기 위해서 RetinaNet이라는 one-stage detector를 설계한다. RetinaNet은 하나의 backbone network와  각각 classification과 bounding box regression을 수행하는 subnet으로 구성되어 있다. 

이미지6

1) **Feature Pyramid by FPN**

가장 먼저 원본 이미지를 backbone network에 입력해서 서로 다른 5개의 multi-scale feature pyramid를 출력한다. 여기서 backbone network는 ResNet 기반의 FPN(Feature Pyramid Network)를 사용한다. pyramid level은 P3 ~ P7으로 설정하는데, P3~P5는 C3~C5에서 top-down과 lateral connection을 이용해서 출력한 것이고, P6는 C5에 3x3 conv (stride=2)를 적용해서 출력하고, P7는 P6에 3x3 conv (stride=2)를 적용한 이후 ReLu를 적용해서 출력한 것이다. FPN에 대한 자세한 설명은 ([<U>FPN 논문 리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/fpn/))를 참고하길 바란다.

* Input : image
* Process : feature extraction by ResNet + FPN
* Output : feature pyramid (P3~P7)

2) **Classification by Classification subnetwork**

(1)에서 얻은 pyramid level별 feature mpa을 Classification subnetwork에 입력한다. 해당 subnet은 3x3(xC) conv layer - ReLU - 3x3(xKxA) conv layer로 구성되어 있다. 여기서 K는 분류하고자 하는 class의 수,  A는 anchor box의 수를 의미한다. 논문에서는 A=9로 설정한다. 그리고 마지막으로 얻은 feature map의 각 spatial location(feature map의 cell)마다 sigmoid activation function을 적용한다. 이를 통해서 channel 수가 KxA인 5개의 feature map을 얻을 수 있다.

RPN과 달리, classification subnet은 더 깊고, 3x3 conv layer들만 사용하고, box regression subnet과 parameter를 공유하지도 않는다. 

* Input : feature pyramid (P3~P7)
* Process : classification by classification subnetwork
* Output : 5 feature maps with KxA channel

3) Bounding box regression by BBR subnetwork

(1)에서 얻은 pyramid level별 feature map을 BBR subnet에 입력한다. 해당 subnet은 classification subnet과 같이 FCN(Fully Convolutional Network)이다. feature map이 anchor box 별로 4개의 좌표값(x,y,w,h)을 encode하도록 channel수를 조정한다. 최종적으로 channel 수가 4xA인 5개의 feature map을 얻을 수 있다.

RetinaNet의 BBR subnet은 classification subnet과 비슷한 구조를 가지지만 parameter는 전혀 공유하지 않는다. 그래서 최근에 다룬 모델들과 달리 class를 모르는 상태에서 BBR를 진행하기 때문에 더 적은 parameter로 학습할 수 있고, 이는 더 효율적이다.



* Input : feature pyramid (P3~P7)
* Process : bounding box regression by BBR subnet
* Output : 5 feature maps with 4xA channel

4) **Anchors**

각 pyramid level의 feature map마다 우리는 translation-invariant한 anchor box를 사용할 것이다. FPN에서 처럼 {1:2, 1:1, 2:1}의 비율의 anchor box를 사용하지만, 더 밀도있는 탐색을 위해서 각 비율마다 {$2^0,2^{1/3},2^{2/3}$}의 anchor box를 사용해서 총 9개의 anchor box를 사용한다. 

### 3. Inference & 결론

RetinaNet은 single FCN 모델로 구성되어 있기 때문에 inference시에는 간단하게 image를 RetinaNet에 forward해주면 된다. Inference시에 속도를 향상시키기 FPN의 각 pyramid level에서 가장 점수가 높은 1000개(threshold = 0.05 confidence)의 prediction만 사용한다. 최종 출력 결과를 도출하기 위해서 2개의 subnetwork의 출력 결과에서 모든 level의 예측 결과는 병합되고, Non-maximum suppression(threshold=0.5)를 적용한다.

RetinaNet을 COCO 데이터셋을 통해 학습시킨 후 서로 다른 loss function을 사용하여 AP 값을 측정한 결과 CE loss는 30.2%, Balanced Cross Entropy는 31.1%, Focal loss는 34% AP 값을 보였다. 또한 SSD 모델을 통해 positive/negative 비율을 1:3으로, NMS threshold=0.5로 설정한 OHEM과 성능을 비교한 결과, Focal loss를 사용한 경우의 AP값이 3.2% 더 높게 나타났다는 것을 통해 Focal loss가 class imbalance 문제를 기존의 방식보다 효과적으로 해결했다고 볼 수 있다.

이번 논문에서는 모델 전체를 수정한 것이 아니라 loss function을 수정하는 단순한 방법으로 성능을 향상시켰다는 점이 인상 깊었다.

출처 및 참고문헌 

RetinaNet 논문 (https://arxiv.org/pdf/1708.02002.pdf)

개인 블로그 (http://herbwood.tistory.com/19)

