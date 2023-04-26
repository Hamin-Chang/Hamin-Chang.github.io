---
layout: single
title:  "[IC/개념] 이미지 분류 - Inception-v4 & Inception-ResNet 📚"
toc: true
toc_sticky: true
categories:
  - cv-imageclassification
---


## Inception-v4 & Inception-ResNet 논문 리뷰

이번 글에서는 [**<U>Inception-v4 & Inception-ResNet 논문</U>**](https://arxiv.org/pdf/1602.07261.pdf)(Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning)을 리뷰한다. 

Inception 계열의 모델들 Inception-v1(GoogLeNet), Inception-v2,3는 이미지 분류 대회에서 항상 좋은 성적을 거둬왔다. Inception 계열 모델들의 특징은 적은 파리미터를 갖기는 하지만 모델의 구성이 좀 복잡하다는 것이다. 본 논문에서는 Inception-v4와 Inception-ResNet을 소개한다. Inception-ResNet은 Incepton-v4에 residual connection을 결합한 것으로, 학습 속도가 빨라졌다고 한다.

이미지1

(그래프에서 나타난 것처럼 Inception 계열 모델들은 높은 성능을 가지고 파라미터의 개수도 비교적 적다.)

### 1. Inception-v4

Inception-v4는 이전 버전에서의 단점을 개선하고, inception block을 균일하게 획일화했다. 다음은 Inception-v4의 전체 구조다.

이미지2

각 모듈이 어떻게 이루어져 있는지 알아보자. 참고로 각 모듈의 V 표시는 padding=valid로 적용한 경우라서 해당 layer를 통과하면 feature size가 축소되고 V가 없다면 zero-padding을 적용한 것이라서 입력과 출력의 feature map size가 동일하게 유지된다. 

1) Stem

299x299x3의 input image가 Stem Block을 거쳐서 35x35x384 사이즈를 만든다. 

이미지3

2) Inception-A

Stem으로부터 35x35x384 feature map을 받아서 처리하는 첫번째 inception block으로서 Inception-A block을 연달아 4개를 이어 붙여서 사용한다.

이미지4

3) Reduction-A

이미지5

다음으로 feature map 사이즈를 반으로 줄여주는 Reduction-A를 거친다. 위의 이미지를 보면 filter 개수가 $k,l,m,n$으로 숫자가 아닌 알파벳으로 되어있는데, Reduction-A block이 Inception-ResNet v1,2에서도 사용되는데, Table 1의 값으로 사용하기 때문이다. 

이미지6

4) Inception-B

17x17x1024의 feature map을 처리하는 Inception-B 모듈이다. 연달아 7개를 이어 붙여서 사용한다.

이미지7

5) Reduction-B

Inception-B에서 나온 feature map의 사이즈를 반으로 줄이는 block이다.

이미지8

6) Inception-C

마지막으로 다음 이미지의 Inception-C block 3개를 거친 후,

이미지9

Average Pooling -> DropOut (rate=0.2) -> Softmax 순으로 전체 모델을 구성한다.

### 2. Inception-ResNet

Inception-ResNet은 Inception network와 residual block을 결합한 모델로, v1과 v2 버전이 있다. 두 모델의 전체적인 구조는 같지만, stem의 구조와 각 inception-resnet block에서 사용하는 filter 수가 다르다.

Inception-ResNet v1은 Inception-v3와 연산량이 비슷하고, Inception-ResNet v2는 Inception-v4와 연산량이 비슷하다고 한다.

Inception-ResNet v1과 v2의 전체적인 구조는 다음과 같다.

이미지10

1) Stem

이미지11

(왼 : v1 , 오 : v2)

v2에서는 기존 Inception-v4에서 사용하는 stem을 사용한다.

2) Inception-ResNet-A

이미지12

(왼 : v1 , 오 : v2)

마지막 1x1 conv에서 filter 수가 다르다. 참고로 위 이미지에서 Linear는 activation 함수를 사용하지 않는다는 것을 의미한다.

3) Inception-ResNet-B

이미지13

(왼 : v1 , 오 : v2)

4) Inception-ResNet-C

이미지14

(왼 : v1 , 오 : v2)

5) Reduction-A

이미지15

v1과 v2 모두 Inception-v4와 같은 Reduction-A를 사용한다.

6) Reduction-B

이미지16

(왼 : v1 , 오 : v2)

### 3. Scaling of Residual

이 논문에서 제시하는 아이디언데, 의미가 있는 부분이라고 생각한다. 보통 filter 개수가 1000개를 넘어가면 모델의 학습이 굉장히 불안정해져서 모델이 죽어버리는 현상이 발생한다. 이 현상은 learning rate를 낮추거나 추가적인 batch normalization을 통해서는 해결하지 못한다.

이미지17

따라서 이 논문에서 위 그림과 같이 Residual을 더하기 전에 Scaling factor를 0.1~0.3 사이로 설정해서 Residual의 값을 대폭 줄였다. 이렇게 하더라도 정확도에는 미치는 영향이 없고, 학습을 안정시키는 효과를 준다고 한다.

### 4. Experiments

연산량이 비슷한 모델끼리의 학습 결과를 비교했다.

1) Inception-v3 vs Inception-resnet-v1 학습 곡선 비교

이미지18

2) Inception-v4 vs Inception-resnet-v2 학습 곡선 비교

이미지19

두 그래프에서 모두 Inception-ResNet이 더 빠르게 error를 줄였다.

각 모델끼리 성능을 비교해봐도 Inception-ResNet-v2가 가장 좋은 성능을 보인다.

이미지20

출처 및 참고문헌:

1. https://arxiv.org/pdf/1602.07261.pdf
2. https://m.blog.naver.com/phj8498/222685190718
3. https://deep-learning-study.tistory.com/525





