---
layout: single
title:  "[IC/개념] 이미지 분류 - PyramidNet 🐫"
toc: true
toc_sticky: true
categories:
  - cv-imageclassification
---

## PyramidNet 논문 리뷰

일반적으로 CNN 모델은 pooling layer에서 memory 사용을 감소하고 고차원의 feature를 추출하기 위한 down-sampling을 수행하고 filter 수를 급격히 증가시킨다. 이는 고차원 정보의 다양성을 증가시키기 때문에 모델의 성능을 향상시킨다. PyramidNet은 down-sampling을 수행할 때, filter 수를 급격히 증가시키는 대신에, 최대한 모든 위치에서 점진적으로 filter 수를 증가시키고 이 방법이 모델의 일반화 성능을 향상시킨다고 한다. 또한 ResNet 구조에 이를 적용했을 때, 성능을 향상시키는 새로운 residual unit을 제안한다. 미리 말하자면 최종적으로 제안된 PyramidNet이 ResNet의 성능을 뛰어넘는다.

### 1. ResNet vs PyramidNet

PyramidNet은 ResNet을 기반으로 성능을 향상시킨 모델이다. 이 둘의 차이점은 ResNet은 pooling layer에서 feature map의 filter 수를 증가시키는 대신, PyramidNet은 모든 layer에서 filter수를 증가시킨다는 것이다. 그리고 새로운 residual unit을 적용한다.

#### 1.1 Original ResNet 

기존 ResNet에서 feature map의 filter 수는 다음 수식으로 나타낼 수 있다. 

이미지1

위 식에서 $D_k$는 $k$번재 residual unit의 feature map의 filter 수를 나타내고, $n(k)$는 $k$번째 residual unit이 속해있는 그룹인데, 해당 그룹은 동일한 feature map 크기를 갖는다. 결국 위 수식은 down sampling이 되는 block을 지날 때마다 filter의 수가 2배씩 늘어난다는 뜻이다.

#### 1.2 Additive & Multiplicative PyramidNet

이미지2

(a) Additive PyramidNet

Additive PyramidNet은 feature map의 차원 수가 다음 식을 따라 선형하게 증가한다.

이미지3

위 식에서 $D_{k-1}$은 이전 group의 채널 수를 뜻하고, $α$ widening factor라는 하이퍼파라미터이다. $N=Σ^4_{n=2}N_n$은 residual unit의 개수를 뜻한다. 따라서 위 식은 모델이 한 group을 지날 때마다 $α/N$만큼 채널 수를 키운다는 것을 의미한다. $N=4$일 때 최종 feature map의 차원 수는 $16 + (n-1)α/3$이 된다.

(b) Multiplicative PyramidNet

Multiplicative PyramidNet은 feature map의 채널 수가 다음 식을 따라 기하학적으로 증가한다.

이미지4

ImageNet과 CIFAR dataset 둘 다 additive PyramidNet이 multiplicative Pyramid보다 더 좋은 성능을 보였다고 한다.

다음은 CIFAR-100 dataset에서 둘의 성능을 비교한 결과인데, 두 방식 모두 레이어가 깊어질수록 성능이 개선되기는 하지만, 레이어가 깊어지면 Additive PyramidNet의 성능이 더욱 큰 격차로 개선되는 것을 볼 수 있다.

이미지10

### 2. Building Block

residual block은 다음 이미지와 같이 다양하게 구성할 수 있다. PyramidNet은 다양한 residual block을 실험하고 가장 높은 성능을 보인 (d) residual block을 사용한다.

이미지5

논문에서 수행한 다양한 residual block의 실험 내용은 다음과 같다.

1. residual unit에서 addition 이후 ReLU를 적용하면 성능저하가 발생한다. ReLU는 negative 값을 0으로 만들기 때문에 short connection은 항상 non-negative 값만 다음 계층으로 전달한다. 이는 ReLU를 residual block 안으로 옮겨서 문제를 해결한다.

2. residual block에서 많은 수의 ReLU는 오히려 성능을 저하시킨다. 첫 번째 ReLU는 제거하고 conv 사이에서만 ReLU를 추가하는 것이 성능이 제일 좋았다.

3. Batch Normalization(BN)은 빠른 수렴을 위해 값을 정규화하여 활성화함수로 전달한다. 이 BN은 residual unit의 성능을 향상시키는데에 사용할 수 있다. residual block의 마지막에 BN을 배치하면 성능이 향상된다고 한다.

이미지6

### 3. PyramidNet's Performance

기존의 ResNet은 down sampling이 되는 residual block을 제거하면 성능이 크게 감소한다. 이것을 해결하는 또 다른 방법을 찾는 것이 PyramidNet의 등장 배경이었다. 

개선된 ResNet 모델 중 하나인 pre-activation ResNet과의 성능을 비교해봤다.

1) 다음 이미지와 같이 PyramidNet의 test 성능이 pre-activation ResNet을 앞섰다. PyramidNet의 일반화 능력이 더 우수한 것을 알 수 있다.

이미지7

2) 각 유닛들을 지워가며 성능을 평가했을 때, down sampling이 되는 유닛을 지웠을 때 pre-activation ResNet의 성능이 다른 유닛을 제거했을 때의 성능에 비해 상대적으로 크게 떨어졌지만 PyramidNet은 그러지 않았다.

3) 아무것도 제거하지 않았을 떄의 성능과 각 유닛을 제거했을 때의 성능 차의 평균이 pre-activation ResNet이 더욱 높았다. 이는 PyramidNet의 앙상블 효과가 더 강하게 나타남을 의미한다.

이미지8


### 4. Zero-padded Identitiy-mapping Shortcut

input과 output의 크기가 다르면 residual block을 이용할 수 없다. 따라서 둘의 크기를 맞추기 위해 **Zero-padded Identitiy-mapping Shortcut**을 사용한다. 저자는 다음 이미지처럼 Zero-padded Identitiy-mapping Shortcut을 사용하는 것이 residual net + plain net을 혼합하는 효과가 있다고 추측한다.

이미지9


출처 및 참고문헌 

1. https://arxiv.org/pdf/1610.02915.pdf
2. https://imlim0813.tistory.com/44
3. https://deep-learning-study.tistory.com/526

