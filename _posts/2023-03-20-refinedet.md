---
title : '[DL/CV] 객체 탐지 - RefineDet 🔧'
layout : single
toc : true
toc: true
toc_sticky: true
categories:
  - cv-objectdetection
---


## RefineDet 논문 읽어보기

이번 글에서는 RefineDet 논문 [<U>RefineDet(Single-Shot Refinement Neural Network for Object Detection</U>](https://arxiv.org/pdf/1711.06897.pdf)을 리뷰해보도록 하겠다. 논문에서는 two-stage detector의 다음 2가지 특징을 설명한다. 

1. two-stage 구조와 더불어 sampling heuristic을 사용하기 때문에 class imbalance 문제가 one-stage detector보다 덜 심각하다.
2. 예측된 bbox의 파라미터를 최적화하기 위해 two-stage cascade를 사용하고, 객체를 표현하기 위해 two-stage feature를 사용하기 때문에 객체에 대한 보다 정교한 예측이 가능해진다. 

논문의 저자는 위에서 언급한 two-stage detection의 장점을 모두 살릴 수 있는 one-stage detector인 RefineDet 모델을 소개한다.

### 0. Preview

![1](https://user-images.githubusercontent.com/77332628/226246358-366b0685-a678-43d6-869e-645238fa81ee.png)

먼저 RefineDet의 전체적인 구조를 살펴보자. RefineDet은 기존의 one-stage detector인 SSD 모델에서 개선된 모델로, SSD와 같이 이미지 전체를 한번에 처리(one-stage detector)하지만 SSD와는 달리 두 개의 단계의 구조를 가지고 있다. SSD에 대한 설명은 [<U>SSD 논문리뷰</U>](https://hamin-chang.github.io/cv-objectdetection/ssd/)를 참고하길 바란다.

RefineDet은 서로 연결되어 있는 ARM, ODM 모듈로 구성되어 있다. **ARM(Anchor Refinement Module)**은 backbone network에서 추출한 multi-scale feature map을 입력으로 받아서 일련의 과정을 거쳐서 feature map을 조정(refine)해서 ODM에 제공한다. **ODM(Objet Detection Module)**은 ARM에서 조정된 feature map을 기반으로 객체에 대한 정확한 위치와 class label을 예측한다. 이 때 ARM에서 출력된 feature map을 ODM에 사용할 수 있도록 변환시켜주는 **TCB(Transfer Connection Block)**이 두 모듈 사이에 존재한다.



### 1. ARM & ODM


![2](https://user-images.githubusercontent.com/77332628/226246362-0f56790f-c62f-4f77-9626-4a6e24818c6e.png)

먼저 **ARM(Anchor Refinement Module)**은 생성된 anchor box 중에서 적절한 sample을 골라내고 이를 조정(refine)하는 역할을 한다. anchor의 위치와 크기를 대략적으로(coarsely) 조정하여, 연결되어 있는 후속 모듈에 초기화가 잘 된 anchor를 제공한다. 따라서 ARM은 two-stage detector에서 사용하는 Region Proposal Network와 같은 기능을 수행한다고 볼 수 있다.

ARM은 backbone network의 지정한 layer에서 feature map을 추출하고 해당 layer에 대해서 conv 연산을 추가하느 구조를 가진다. 이를 통해서 refined된 **anchor box의 위치 정보**를 담고 있는 feature map과 해당 anchor box의 fo**reground/background lable**에 대한 정보를 가지고 있는 feature map을 얻을 수 있다. 참고로 foreground/background label에 대한 정보는 preview에서 다룬 class imbalance를 해결하는데 사용된다.

![5](https://user-images.githubusercontent.com/77332628/226246367-ec52dcfd-e985-401f-a5a7-313e230b6cc8.png)


그 다음 **ODM(Object Detection Module)**은 ARM으로부터 refined anchor에 대한 정보를 입력으로 받아서 객체에 대한 정확한 위치와 class label을 예측하는 역할을 한다. 위 이미지에서 ARM과 ODM을 연결해주는 TCB(Transfer Connection Block)은 바로 뒤에서 다루겠다. ODM은 TCB에서 출력된 feature map에 conv 연산을 적용해서 **객체의 세밀한 위치와 class label에 대한 정보**를 담고 있는 feature map을 출력한다.

### 2. TCB


![3](https://user-images.githubusercontent.com/77332628/226246363-bde38b85-fff7-49cd-a42a-0d968346b8ed.png)

TCB(Transfer Connection Block)은 ARM과 ODM을 연결시키기 위해서 ARM의 서로 다른 layer로부터 비롯된 feature map을 ODM이 요구하는 형태에 맞게 변환시켜주는 역할을 하는데, 이를 통해 ODM이 ARM과 feature를 공유할 수 있도록 해준다. (참고로 anchor의 정보 feature map만 TCB에 입력하고, positive/negative label feature map은 TCB에 입력하지 않는다.)


![4](https://user-images.githubusercontent.com/77332628/226246364-3f788956-2cc0-47b9-bd30-ac6e0af38d17.png)

TCB는 2개의 feature map을 입력받는다. feature map1은 ARM으로부터 anchor와 관련된 feature map인데, 이를 일련의 conv layer(conv-relu-conv)를 거쳐서 channel 수를 256으로 맞춘다.

그리고 f**eature map2는 backbone network의 후속 layer에서 추출**한 feature map을 ARM과 TCB의 conv layer에 입력시켜서 얻은 결과이다. feature map1보다 더 깊은 layer에서 추출했기 때문에 feature map2는 feature map1보다 작다. 따라서 feature **map2에 deconvolution 연산을 적용하고 feature map1과 element-wise하게 더해**준다. 그리고 합쳐진 feature map을 conv layer(conv-relu-conv)에 입력해서 얻은 결과를 ODM에 전달하는 것이다.

이러한 과정을 통해서 더 깊은 layer에서 얻은 high-level feature를 활용할 수 있게 된다. 따라서 **TCB는** 서로 다른 scale을 가진 feature map을 upsampling한 후 element-wise하게 더해주는 **FPN과 같은 역할**을 한다고 볼 수 있다.

TCB를 사용하지 않으면 mAP 값이 1.1% 하락한다고 한다.



### 3. RefineDet의 특징

1) Two-step Cascaded Regression

One-stage detector는 작은 객체를 포착하지 못하는 문제가 자주 발생한다. RefineDet은 이러한 문제를 해결하기 위해서 Two-step Cascaded Regression을 사용했다. 이는 위 내용과 같이 ARM에서 anchor의 크기와 위치를 조정하는 과정을 거치고, ODM에서 세밀한 bounding box regression을 수행하는 것으로 적용된다.

ARM을 사용하지 않아서 Two-step Cascaded Regression을 수행하지 않으면 mAP 값이 2.2% 하락했다고 한다.

2) Negative Anchor Filtering

논문에서는 class imbalance 문제를 줄이기 위해서 negative anchor filtering 과정을 추가했다. 모델을 학습시킬때 ARM이 출력한 결과에 대해서 만약 negative confidence가 사전에 설정한 threshold값보다 높다면 ODM에 해당 anchor box를 전달하지 않는 방법이다. 이렇게 하면 refined hard negative anchor box, 즉 조정된 hard negative(모델이 예측하기 어려워하는)sample과 refined positive anchor box만을 ODM에 전달하게 된다. 따라서 모델이 새로운 샘플에 대해 더욱 견고한 예측을 수행할 수 있으며, 이를 통해 더욱 정확한 객체 탐지가 가능해진다.



### 4. Training RefineDet

![5](https://user-images.githubusercontent.com/77332628/226246367-ec52dcfd-e985-401f-a5a7-313e230b6cc8.png)


1) Multi-scale feature extraction from backbone netwrok

RefineDet은 backbone network로 VGG-16에 extra layer를 부착한 형태의 network를 사용한다.

* Input : Input image
* Process : feature extraction from designated layers
* Output : multi-scale feature maps {v1, v2, v3, v4}

2) Refine anchor boxes by ARM

{v1, v2, v3, v4}을 ARM에 입력해서 refined anchor에 대한 feature map과 positive/negative 여부에 대한 feature map을 추출하고, 두번째 feature map에 대해서 negative anchor filtering 과정을 수행한다.

* Input : multi-scale feature maps {v1, v2, v3, v4}
* Process : refine anchors (conv layers, negative anchor filtering)
* Output : refined anchors {(v1r1, v1r2), (v2r1, v2r2), (v3r1, v3r2), (v4r1, v4r2)} 

3) TCB as FPN

위 내용에서 언급한 것과 같은 FPN의 역할을 하는 TCB의 과정을 거쳐서 ARM의 feature map을 ODM이 요구하는 형태로 변환한다.

* Input : efined anchors {(v1r1, v1r2), (v2r1, v2r2), (v3r1, v3r2), (v4r1, v4r2)} 
* Process : transfer ARM features
* Output :  transfered features {(v1r1 + deconv(T(v2r1))), (v2r1 + deconv(T(v3r1))), (v3r1 + deconv(T(v4r1))), v4r1}

4) Predict BBR and class label by ODM

ARM에서 얻은 positive/negative 여부에 대한 feature map과 TCB에서 얻은 transfered features를 입력받아서 최종 prediction을 수행한다.

* Input : pos/neg features {v1r2, v2r2, v3r2, v4r2} &  transfered features {(v1r1 + deconv(T(v2r1))), (v2r1 + deconv(T(v3r1))), (v3r1 + deconv(T(v4r1))), v4r1}
* Process : final prediction
* Output : bounding box regressors, class scores

### 5. 결론

RefineDet은 OC 2007, 2012 데이터셋을 통해 실험한 결과, mAP 값이 각각 85.8%, 86.8%을 보이며, 당시 state-fo-the-art 모델보다 더 높은 정확도를 보였다고 한다. 또한 MS COCO 데이터셋에서는 mAP 값이 41.8%를 보였다.

특히 RefineDet은 two-stage detector에서 주로 사용되는 아이디어들을 one-stage detector에 자연스럽게 적용했다는 점이 흥미로웠다.

출처 및 참고문헌 :

개인 블로그 (https://herbwood.tistory.com/22)

RefineDet 논문 (https://arxiv.org/pdf/1711.06897.pdf)
