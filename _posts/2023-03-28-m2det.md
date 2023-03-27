---
title : '[DL/CV] 객체 탐지 - M2Det 🍰'
layout : single
toc : true
toc: true
toc_sticky: true
categories:
  - cv-objectdetection
---

## M2Det 논문 읽어보기

이번 글에서는 [<U>M2Det 논문</U>](https://arxiv.org/pdf/1811.04533.pdf)(M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network)을 리뷰해보도록 하겠다. 

### 0. 기존 FPN의 한계

본 논문에서는 먼저 multi-scale feature map 생성을 위해 주로 사용되던 FPN(Feature Pyramid Network)의 두가지 한계에 대해 언급한다.

![1](https://user-images.githubusercontent.com/77332628/228092058-e452e31d-427f-4ae2-9906-de940656b68a.jpeg)

1. FPN은 classification task를 위해 설계된 backbone network로부터 feature map을 추출하는데, 이를 통해 구성된 feature pyramid는 object detection task를 수행하기 위해 충분히 대표적이거나 일반적(representative)이지 않다. 
2. Feature pyramid의 각 level의 feature map은 주로 backbone network의 single-level layer로 구성되어 있기 때문에 객체의 외형에 따른 인식 성능의 차이가 발생한다.

논문에서는 두번째 한계에 대해 부연 설명한다. 일반적으로 네트워크의 high-level feature는 classification task에 적합하고, low-level feature는 localization task에 적합하다. 이 외에도 전자는 복잡한 외형의 특징을 포착하는데 유리하고, 후자는 단순한 외형을 파악하는데 유리하다.

현실의 데이터에서 비슷한 크기를 가지지만 객체에 대한 외형의 복잡도는 상당히 다를 수 있다. 예를 들어 이미지에서 신호등과 멀리 있는 사람은 비슷한 크기를 가지지만 사람의 외형이 더 복잡하다. 이 같은 경우 single-level feature map을 사용하게 되면 두 객체를 모두 포착하지 못하는 문제가 발생할 수도 있다.

본 논문에서는 위에서 언급한 기존 FPN의 문제를 해결하는 multi-scale, multi-level feature map을 사용하는 one-stage detector인 M2Det에 대해 다룬다.




### 1. MLFPN (Mutli-Level Feature Pyramid Network)

![2](https://user-images.githubusercontent.com/77332628/228092061-63b23c4b-0361-468e-ae74-e3a39cbfedbf.png)

M2Det의 자세한 아이디어를 알아보기 전에 전체적인 구조에 대해 알아보자. 논문에서는 서로 다른 크기와 외형의 복잡도를 가진 객체를 포착하기 위해 보다 효율적인 feature pyramid를 설계하는 **MLFPN(Mutli-Level Feature Pyramid Network)**을 제시한다. 

MLFPN은 크게 **FFM, TUM, SFAM**의 세 가지 모듈로 구성되어 있다. 먼저 **FFM은 Feature Fusion Module**로 backbone network로부터 받은 얕은 feature과 깊은 feature를 fuse하여 **base feature를 생성**한다. 그 다음 TUM은 Thinned U-shaped Module로 서로 다른 크기를 가진 feature map을 생성한다. 그 후 **FFMv2에서 base feature과 이전 TUM의 가장 큰 scale의 feature map을 fuse**하고, 그 다음 TUM에 입력한다. 마지막 **Scale-wise Feature Aggregation Module인 SFAM**에서 multi-level, multi-scale를 **scale-wise feature concatenation과 channel-wise attention 연산**을 통해 집계한다. 

최종적으로 MLFPM과 SSD를 결합해서 M2Det이라는 end-to-end one-stage detector를 설계한다. 이제 모듈의 연산 과정과 역할을 하나씩 자세히 알아보자.

### 2. FFM (Feature Fusion Module)

![3](https://user-images.githubusercontent.com/77332628/228092064-3bb4391b-ca46-4665-baf9-950419fb8635.png)

FFM(Feature Fusion Module)은 네트워크에 있는 서로 다른 feature를 융합(fuse)하는 모듈로, 같은 역할을 수행하지만 서로 다른 구조의 FFMv1과 FFMv2가 있다.

1) **FFMv1**은 backbone network로부터 서로 다른 scale의 두 feature map을 추출한 후 융합해서 **base feature map을 생성**한다. 위 이미지의 (a)와 같이 각각의 feature map에 conv 연산을 적용하고, scale이 작은 feature map은 upsample 시킨 후 concat 하여 하나의 feature map을 얻는다. 이 과정을 통해서 얕은 layer와 깊은 layer에서 추출된 feature map 두 개를 사용하기 때문에 **풍부한 semantic 정보**를 MLFPN에 제공하는 역할을 한다.

2) **FFMv2**는 FFMv1이 생성한 base feature에 대해 conv 연산을 적용한 후 이전 TUM의 가장 큰 scale의 feature map을 입력받아 concat해서 **다음 TUM에 전달**하는 역할을 수행한다. 위 이미지의 (b)와 같이 동작하는데, 이 때 입력으로 사용하는 두 feature map의 scale이 같음을 알 수 있다. 




### 3. TUM (Thinned U-shape Module)


![4](https://user-images.githubusercontent.com/77332628/228092065-fa131fc9-3b57-4c0f-a946-87d7b2d054ca.png)

TUM은 입력받은 feature map에 대해서 multi-scale feature map을 생성하는 역할을 수행하며, **Encoder-Decoder 구조**로 U자형 구조를 가진다.

1) **Encoder network**에서는 입력받은 feature map에 대해 3x3 conv(stride=2) 연산을 적용해서 **scale이 다른 다수의 feature map**({E1, E2, E3, E4, E5})을 출력한다.

2) **Decoder network**는 Encoder network에서 출력한 다수의 feature map에 대해 더 높은 level(=scale이 더 작은)에 대해 upsample을 수행한 후 바로 아래 level의 feature map과 element-wise하게 더해준 후 1x1 conv 연산을 수행한다. 이를 통해 최종적으로 scale이 다른 다수의 feature map({D1, D2, D3, D4, D5, D6})을 출력한다.

![5](https://user-images.githubusercontent.com/77332628/228092068-39dbfb49-f6ce-4332-807f-ba91eb0fa2e7.png)

위 이미지와 같이 MLFPN 내부에서 TUM은 FFM과 서로 교차하는 구조를 가진다. FFMv1에서 얻은 base feature map을 첫 번째 TUM에 입력해서 feature map({D1, D2, D3, D4, D5, D6})을 얻고, TUM의 출력 결과 중 가장 큰 scale의 feature map과 base feature map을 FFMv2를 통해 fuse한 후 두번째 TUM에 입력하고, 이러한 과정을 반복한다. 논문에서는 총 8개의 TUM을 사용했다고 한다.

각각의 TUM의 Decoder network의 출력값은 **입력으로 주어진 feature map의 level에 대한 multi-scale feature map**에 해당한다. 따라서 전체 TUM에 대해 봤을 떄, 축적된 모든 TUM의 feature map은 multi-level, multi-scale feature를 형성하게 된다. 즉, 초반의 TUM은 shallow-level feature, 중간의 TUM은 medium-level feature, 후반의 TUM은 deep-level feature를 제공하게 되는 것이다.

### 4. SFAM (Scale-wise Feature Aggregation Module)

![6](https://user-images.githubusercontent.com/77332628/228092070-6ed47f1f-36d4-4ff7-9249-ec4f3fbc3058.png)

**SFAM(Scale-wise Feature Aggregation Module)**은 TUMs에 의해 생성된 multi-level, multi-scale feature를 구성하는 **scale-wise feature concatenation**과 **channel-wise attention** 매커니즘을 통해 집계해서 multi-level feature pyramid로 구성하는 역할을 수행한다.

1) Scale-wise feature concatenation

![7](https://user-images.githubusercontent.com/77332628/228092072-5b2bbb13-d2a7-4798-baa7-a7a27d412e4e.png)

Scale-wise feature concatenation은 각각의 TUM으로부터 생성된 multi-level feature map을 같은 scale 별로 concat하는 과정이다. 각각의 TUM은 특정 level의 feature map을 출력한다. 위 이미지에 나와 있는 예시의 경우 3개의 TUM이 각각 shallow, medium, deep level feature map들을 생성하고, 각 level의 feature maps는 3개의 서로 다른 scale의 feature map으로 구성되어 있다. 여기서 같은 scale을 가지는 feature map끼리 concat 해줌으로서, 서로 다른 level을 가진 같은 scale의 feature map 3개가 생성된다.

논문에서는 8개의 TUM이 각각 6개의 multi-scale feature map을 출력한다고 한다. 따라서 실제 Scale-wise feature concatenation 과정을 수행하면, 서로 다른 level에 대한 정보를 함축한 8개의 feature map이 결합되어서 최종적으로 6개의 multi-level, multi-scale feature map을 출력한다.

2) Channel-wise attention

하지만 논문에서는 단순히 Scale-wise faeture concatenation만으로는 충분히 적용 가능(adaptive)하지 않다고 언급했다. Channel-wise attention 모듈은 feature가 가장 많은 효율을 얻을 수 있는 channel에 집중(attention)하도록 설계하는 작업이다. 본 모듈에서는 Scale-wise feature concatenation 과정에서 출력한 feature map을 **SE(Squeeze Excitation) block**에 입력한다. 그럼 SE block이란 무엇일까?

![8](https://user-images.githubusercontent.com/77332628/228092075-9a145655-aaaf-44b5-9123-f3847067f4ed.png)

SE(Squeeze Excitation) block은 CNN에 부착해서 사용할 수 있는 블록으로, 연산량을 늘리지 않으면서 정확도를 향상시킨다. SE block은 다음 3가지 step으로 구성되어 있다.

1. Squeeze step : 입력으로 들어온 HxWxC 크기의 feature map에 대해 Global Average Pooling을 수행한다. 이를 통해서 channel을 하나의 숫자로 표현하는 것이 가능하다. (위 이미지에서 $F_{sq}$ 과정)

2. Excitation step : 앞서 얻은 1x1xC feature map에 대해 2개의 fc layer를 적용하여 channel별 상대적 중요도를 구한다. 이때 두 번째 fc layer의 activation function을 sigmoid로 지정한다. 이 과정을 통해서 최종 output은 0~1 사이 값을 가져 channel별 중요도를 파악하는 것이 가능하다. (위 이미지에서 $F_{ex}$ 과정)

3. Recalibration step : 앞선 과정에서 구한 channel별 중요도와 원본 feature map을 channel별로 곱해줘서 channel별 중요도를 재보정(recalibrate)해준다. (위 이미지에서 $F_{scale}$ 과정)




### 5. Training M2Det

![9](https://user-images.githubusercontent.com/77332628/228092079-02fea00c-e675-4d23-839f-ac34d222bedb.png)

1) Extract two feature maps from backbone network

가장 먼저 backbone network로부터 서로 다른 level에서 서로 다른 scale을 가진 두개의 feature map을 추출한다. (backbone network로 VGG 혹은 ResNet 사용)

* Input : input image
* Process : extract two feature maps
* Output : two feature maps within different scales

2) Generate Base feature map by FFMv1

* Input : two feature maps within different scales
* Process : fuse two feature maps
* Output : Base feature map

3) Generate Mutli-level, Multi-scale feature maps by FFMv2 + TUM

논문에서는 TUM을 8개로 설정했기 때문에 TUM과 FFMv2를 교차하는 과정을 반복한다.

* Input : Base feature map
* Process : Iterate through FFMv2s and TUMs
* Output : 8 Multi-level, Multi-scale feature maps

4) Construct Final Feature pyramid by SFAM 

* Input : 8 Multi-level, Multi-scale feature maps
* Process : Scale-wise feature concatenation and Channel-wise attention in SFAM
* Output : Feature pyramid with 6 recalibrated feature maps

5) Prediction by classification brand and bbox regression branch

Feature pyramid의 각 level별 feature map을 두 개의 병렬로 구성된 conv layer에 입력하여 class score와 bbox regressor를 얻는다.

* Input : Feature pyramid with 6 recalibrated feature maps
* Process : classification & bbox regression
* Output : 6 class scores and bbox regressions

### 6. Detection & 결론

실제 detection시에는 네트워크에서 예측한 bounding box에 대해 Soft-NMS를 적용해서 최종 prediction을 출력한다.

![dfd](https://user-images.githubusercontent.com/77332628/228092210-47b121ac-05d7-49c0-a95d-965fc2fd7f2f.png)

M2Det은 MS COCO 데이터셋을 통해 실험한 결과, AP 값이 44.2%를 보이면서 당시 모든 one-stage detector의 성능을 뛰어넘는 놀라운 결과를 보였다고 한다.

![10](https://user-images.githubusercontent.com/77332628/228092081-2ea44e46-cb6e-4f2d-8faa-2522557fd8d5.jpeg)

M2Det의 가장 큰 특징은 multi-sclae feature map보다 발전된 multi-level로 구성된 Feature Pyramid를 설계했다는 점이라고 생각한다. 

위 이미지에는 사람 객체 2개, 차 객체 2개, 신호등 객체 1개를 포함하고 있다. 여기서 사람 객체끼리, 그리고 차 객체끼리 서로 다른 크기를 가지고, 신호등은 작은 사람과 작은 차와 비슷한 크기를 가진다. 위의 객체에 대한 활성도를 통해 다음과 같은 사실을 알 수 있다.

* 작은 사람과 작은 차는 큰 feature map에서 강한 활성화 정도를 보이는 반면, 큰 사람과 큰 차는 작은 크기의 feature map에서 강한 활성화 정도를 보인다. 이는 multi-scale featuer이 필요함을 뜻한다.

* 신호등, 작은 사람, 작은 차는 같은 크기의 feature map에서 큰 activation value를 가진다. 이는 세 객체가 서로 비슷한 크기를 가지고 있기 때문이다. 

* 사람, 차, 신호등은 각각 highest-level, middle-level, lowest-level feature map에서 가장 큰 activation value를 가지는데, 이는 multi-level feature가  객체의 외형의 복잡도를 잘 포착하고 있음을 나타낸다.

출처 및 참고문헌 :

개인 블로그 (https://herbwood.tistory.com/23)

M2Det 논문 (https://arxiv.org/pdf/1811.04533.pdf)
