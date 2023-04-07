---
title : '[CV/Pytorch] 파이토치로 컨볼루션 구현하기 1 🌐'
layout: single
toc: true
toc_sticky: true
categories:
  - CVBasic
---
## 파이토치로 구현하는 컨볼루션(컨볼루션 개념, 패딩, 풀링)

이전 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/birdplane2/))에서는 선형 계층에 존재하는 여러 파라미터를 이용해서 데이터에 맞춘 (혹은 과적합된) 단순 신경망을 만들었다. 하지만 완전 연결 모델은 새나 비행기의 특성을 일반화해서 훈련하지 않고 그저 훈련셋을 암기하는 성향을 보였다. 그렇기 때문에 완전 연결 모델은 너무 많은 파라미터가 필요하고 객체의 위치에 독립적이지도 못했다. 그래서 이번 글에서는 새로운 방법으로 모델을 구축하는 방법에 대해 다뤄보고자 한다.

### 1. 컨볼루션이란?
새로운 모델을 구축하는 것에 대해 배우기 전에, 컨볼루션이라는 컴퓨터 비전의 기본 개념에 대해 알아보자. 이전 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/birdplane2/))에서 우리는 **평행이동의 불변성**이라는 지역화된 패턴이 이미지의 어떤 위치에 있더라도 동일하게 출력에 영향을 주는 성질을 본 적이 있다. 평행이동 불변성을 보장하는 선형 연산이 존재하는데, 이것이 바로 **컨볼루션 (convolution)**이다. 컨볼루션 (정확하게는 이산 컨볼루션) 은 2차원 이미지에 가중치 행렬을 스칼라곱을 수행하는 것으로 정의한다. 가중치 행렬은 **커널 (kernel)**이라고 부르며, 입력의 모든 이웃에 대해 수행한다. 다음 이미지처럼 컨볼루션은 진행된다.

![cnn1](https://user-images.githubusercontent.com/77332628/210292682-38f8c77e-b94a-43bc-8157-bd0dbfbbd430.jpg)

(출처 : http://jase.tku.edu.tw/articles/jase-202202-25-1-0020)

위의 이미지처럼 커널이 평행이동 하면서 입력의 모든 위치에 대해 가중치의 합을 구해서 출력 이미지를 만든다. RGB 이미지처럼 채널이 여러개인 경우 가중치 행렬은 KxKx3 행렬로 각 채널에 대한 가중치 집합이 존재하고, 이를 합쳐서 출력값 계산에 기여하게 된다. 이렇게 컨볼루션을 사용하면,

* 주위 영역에 대한 지역 연산이 가능하고,
* 평행이동 불변성을 가지며,
* 더 적은 파라미터를 사용한다.

더 적은 파라미터를 사용하는 이유는 완전 연결 모델과는 달리 컨볼루션에서의 파라미터 수는 이미지 픽셀 수에 의존하지 않는 대신 컨볼루션 커널의 크기와 모델에서 얼마나 많은 컨볼루션 필터를 쓰는지에 의존하기 때문이다.

### 2. 컨볼루션 사용해보기
이제 파이토치로 컨볼루션을 구현해보자. torch.nn 모듈은 1,2,3차원에 대한 컨볼루션을 제공하는데, 우리는 이미지용인 2차원 컨볼루션인 nn.Conv2d를 CIFAR-10 데이터에 대해 사용한다. nn.Conv2d에 전달하는 인자는 최소 입력 피처 수와 출력 피처 수, 커널의 크기이다. 우리의 첫 컨볼루션 모듈은 RGB 채널을 가지기 때문에 픽셀당 3개의 입력 피처를 가지고 출력 피처는 임의로 16으로 정한다. 출력 이미지가 더 많은 채널을 가질수록 신경망의 용량도 커진다. 일단 커널 크기는 3x3으로 정하자.


```python
import torch.nn as nn
conv = nn.Conv2d(3,16,kernel_size=3) # 커널 크기에 (3,3)을 전달해도 동일하다.
conv
```




    Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))



컨볼루션의 weight 텐서는 어떤 차원 정보를 가질까? 커널이 3x3이기 때문에 가중치 역시 3x3을 사용한다. 출력 픽셀 하나에 대해 커널은 입력 채널이 in_ch =3이므로 출력 픽셀 값 하나에 대해 가중치는 in_ch x 3 x 3 이되고, 이 값을 출력 채널 만큼 가지기 때문에 출력 채널 수가 out_ch = 16이므로 전체 가중치 텐서는 out_ch x in_ch x 3 x3, 즉 16x3x3x3이 된다. 편향값의 크기는 출력 채널에 따라 16이다. 한 번 확인해보자.


```python
conv.weight.shape, conv.bias.shape
```




    (torch.Size([16, 3, 3, 3]), torch.Size([16]))



2차원 컨볼루션은 2차원 이미지를 출력한다. 위에서 만든 컨볼루션의 경우는 커널 가중치와 편향값 conv.weight가 랜덤으로 초기화되기 때문에 출력 이미지 자체가 특별한 의미를 갖지는 않는다. 하나의 입력 이미지로 conv 모듈을 사용하려면, nn.Conv2d는 입력으로 BxCxHxW를 받기 때문에 보통 0번째 차원을 unsqueeze를 통해 배치 차원으로 사용한다.


```python
# cifar10 데이터 로드 (저번 글에서 다룸)
from torchvision import datasets, transforms
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

# cifar10에서 비행기와 새의 이미지만 추출 (저번 글에서 다룸)
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10 
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data-unversioned/p1ch7/cifar-10-python.tar.gz





      0%|          | 0/170498071 [00:00<?, ?it/s]



    Extracting ../data-unversioned/p1ch7/cifar-10-python.tar.gz to ../data-unversioned/p1ch7/
    Files already downloaded and verified



```python
img, _ = cifar2[0]
output = conv(img.unsqueeze(0))
img.unsqueeze(0).shape, output.shape
```




    (torch.Size([1, 3, 32, 32]), torch.Size([1, 16, 30, 30]))



입력 이미지의 크기와 출력 이미지의 크기가 다르다! 이미지를 시각화해보자.


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4.8))  
ax1 = plt.subplot(1, 2, 1)   
plt.title('output')   
plt.imshow(output[0, 0].detach(), cmap='gray')
plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)  
plt.imshow(img.mean(0), cmap='gray')  
plt.title('input')  
plt.show()
```


    
![conv00](https://user-images.githubusercontent.com/77332628/210292686-851292a3-a82f-4108-b9d2-c1ded472eaf1.png)
    


역시나 출력 이미지가 입력 이미지보다 픽셀이 조금 잘린것 같다. 이에 대한 해결 방안이 있다!

#### 2.1. 경계 패딩하기
출력 이미지가 입력 이미지보다 작아지는 것은 이미지의 경계에서 이뤄지는 작업에 따른 부작용이다. 컨볼루션 계산을 해보면 각 모서리에는 픽셀의 주변에 값이 없는 픽셀도 있기 때문에 크기가 홀수인 컨볼루션 커널의 길이의 절반만큼 양쪽의 그림이 잘린다. 그렇기 때문에 각 차원에서 딱 두 픽셀만큼 없어진다. 파이토치는 이 문제점을 해결하기 위해 이미지의 경계값에 값이 0인 픽셀들은 **패딩 (padding)**해주는 기능을 제공한다. 다음 이미지에서 패딩을 하면 입력과 출력 이미지의 크기가 똑같이 유지되는 과정을 나타낸다.

![conv2](https://user-images.githubusercontent.com/77332628/210292689-9239fc30-564d-4a56-96b6-c2b62e97233c.gif)

(출처 : https://excelsior-cjh.tistory.com/79)


```python
conv = nn.Conv2d(3,1,kernel_size=3, padding=1) # 크기를 유지할 것이기 때문에 1로 패딩한다.
output = conv(img.unsqueeze(0))
img.unsqueeze(0).shape, output.shape
```




    (torch.Size([1, 3, 32, 32]), torch.Size([1, 1, 32, 32]))



입력과 출력 이미지 모두 32x32로 크기가 같게 유지가 되었다. 그리고 패딩의 사용 여부와 관계없이 weight과 bias의 크기는 변하지 않는다는 점을 기억하자. 

#### 2.2 컨볼루션은 블랙박스가 아니다!
이전 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/pytrain2/#1-%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EC%9E%90%EB%8F%99-%EA%B3%84%EC%82%B0))에서 nn.Linear에서는 weight와 bias는 역전파를 통해서 학습되는 파라미터라고 배웠다. 하지만 이와 다르게 컨볼루션에서는 가중치를 직접 설정해서 컨볼루션에서 어떤 일이 일어나는지 알아볼 수도 있다. 

일단 교란 변수를 제거하기 위해 bias는 0으로 만들고, 가중치에 상수값을 넣어서 어떤 가중치를 넣었을 때 출력값이 어떻게 달라지는지 알아보자.


```python
import torch

with torch.no_grad():
  conv.bias.zero_()

with torch.no_grad():
  conv.weight.fill_(1.0/9.0) # 출력 픽셀 = 주변 픽셀에 대한 평균 이도록 가중치 설정
```


```python
output = conv(img.unsqueeze(0))
plt.figure(figsize=(10, 4.8)) 
ax1 = plt.subplot(1, 2, 1)   
plt.title('output')  
plt.imshow(output[0, 0].detach(), cmap='gray')
plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)  
plt.imshow(img.mean(0), cmap='gray')  
plt.title('input') 
plt.show()
```


    
![conv01](https://user-images.githubusercontent.com/77332628/210292687-33fd1df2-8dc3-4d38-a528-00babd30498f.png)

    


출력 픽셀이 입력 픽셀의 주변 픽셀에 대한 평균이 되도록 가중치를 설정했기 때문에 출력이 입력에 비해 픽셀 간의 변화가 부드러워졌다. 이번에는 입력 픽셀의 오른쪽 픽셀에서 오른쪽 픽셀을 빼는 계산을 수행하는 가중치를 설정해보자. 이렇게 되면 출력 픽셀은 수직 경계의 커널이 적용되면 큰 값을 출력할 것이기 때문에 이 커널은 가로로 인접한 두 영역 사이의 수직 경계를 탐색하는 역할을 할 것이다.


```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])
    conv.bias.zero_()
    
output = conv(img.unsqueeze(0))
plt.figure(figsize=(10, 4.8))  
ax1 = plt.subplot(1, 2, 1)   
plt.title('output')  
plt.imshow(output[0, 0].detach(), cmap='gray')
plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)  
plt.imshow(img.mean(0), cmap='gray') 
plt.title('input') 
plt.show()
```


    
![conv02](https://user-images.githubusercontent.com/77332628/210292690-7ba35011-e64f-4e56-8365-dff0ccd59869.png)
    


이렇듯 커널에 따라서 중요한 특징을 더 잘 탐지하는 다양한 종류의 필터를 만들어서 사용할 수 있다. 전통적인 필터 설계로는 컴퓨터 비전 전문가들의 역할로 이러한 필터를 최적으로 조합해서 이미지의 특징을 강조하는 식으로 물체를 인식했었지만, 이제는 어떻게 이미지를 가장 효과적으로 인식하는지를 몰라도, 딥러닝을 통해서 데이터로부터 **커널을 자동으로** 만들게 된다. 예를 들어서 이전 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/birdplane2/#4-%EB%B6%84%EB%A5%98%EB%A5%BC-%EC%9C%84%ED%95%9C-%EC%86%90%EC%8B%A4%EA%B0%92))에서 소개한 관측값과 출력값 사이의 음의 크로스엔트로피 손실을 최소화하는 관점에서 컨볼루션 신경망의 역할은 어떤 멀티 채널 이미지를 다른 멀티 채널 이미지로 변환하는 연속된 계층인 필터 더미 집합인 커널을 추정하는 것이며, 여기서 각 채널은 피처에 대응하게 된다. 다음 이미지는 훈련을 통해 자동으로 커널을 학습하는 과정을 나태난다.

![conv3](https://user-images.githubusercontent.com/77332628/210292691-f30d3db4-6e4a-4d4f-a0ec-8b39d146cc02.png)

(출처 : https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-8/74)

#### 2.2.3 풀링으로 깊은 컨볼루션 만들기
위의 내용에서 컨볼루션을 배우면서 완전 연결 모델에서 컨볼루션으로 넘어오면서 지역성이나 평행이동 불변성을 해결했다. 이때, 3x3이나 5x5의 작은 커널을 사용하는 것을 추천했는데, 이 정도 크기가 지역성의 한계다. 이미지 안의 물체나 구조가 3픽셀이나 5픽셀밖에 안된다는 보장은 없기 때문에 더 큰 범위에서 신경망이 패턴을 인식하게 하기 위한 해결책이 필요하다. 

가능한 방법 중 하나는 더 큰 컨볼루션 커널을 사용하는 것이다. 32x32 이미지에 대해서는 32x32 커널까지 만들 수 있지만, 그러면 완전 연결된 아핀 변환으로 수렴해서 컨볼루션의 장점을 잃어버리게 된다. 컨볼루션을 차례로 층층이 쌓으면서 동시에 연속적인 컨볼루션 사이의 이미지를 다운샘플링하는 방법을 사용하면 된다.

다운샘플링은 여러 방법으로 수행한다. 이미지를 절반으로 다운샘플링하는 것은 네개의 입력 픽셀을 받아서 한 픽셀을 출력하는 작업과 동일하다. 네개의 입력 픽셀 중 어떤 값을 출력할 것인지는 우리가 정하면 된다.

* 네 개의 픽셀 평균하기 : 평균 풀링 (average pooling) 
* 네개의 픽셀 중 최댓값 : **맥스 풀링 (max pooling)** 데이터의 75%를 버린다는 단점이 있지만 오늘날 가장 널리 사용된다. 

우리는 맥스 풀링을 사용해보자. 맥스 풀링은 다음 이미지처럼 동작한다.

![conv4](https://user-images.githubusercontent.com/77332628/210292692-86fe3c52-46a2-4cda-818a-68fa4528398d.png)

(출처 : https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-8/74)

직관적으로 생각하면, 컨볼루션층의 출력 이미지의 값은 특정 커널에 대응하는 패턴이 발견될 때 높은 값을 가진다. 맥스 풀링을 통해 2x2 인접 픽셀에서 최댓값을 뽑는 다는 것은 약한 신호는 버리고 강한 신호의 피처를 발견하는 과정으로 볼 수 있다. 맥스 풀링은 nn.MaxPool2d 모듈에 있다. 인자로는 이미지를 절반으로 줄이고 싶다면 2로 지정하면 된다. 



```python
pool = nn.MaxPool2d(2) # 맥스 풀링
output = pool(img.unsqueeze(0))

img.unsqueeze(0).shape, output.shape
```




    (torch.Size([1, 3, 32, 32]), torch.Size([1, 3, 16, 16]))



이제 컨볼루션과 다운샘플링을 조합해서 어떻게 큰 패턴을 모델이 인식하는지 알아보자. 다음 이미지에서는 8x8 이미지에 3x3 커널을 적용해서 같은 크기의 멀티 채널 출력 이미지를 얻고, 맥스 풀링을 적용해서 4x4 이미지를 얻고 다른 3x3 커널을 적용한다. 두번째 커널셋을 적용할 댸는 반으로 줄어든 이미지 안에서 3x3의 인접 픽셀에 대해 동작하는 것이기 때문에, 원래 입력의 8x8 인접 영역에 대해 효과적으로 동작하는 셈이다. 두번째 커널셋은 첫번째 커널셋의 피처를 받아 추가적인 피처를 추출한다. 이런 메카니즘으로 CIFAR-10에서 얻은 32x32 이미지보다 훨씬 더 복잡한 장면을 인식하는 컨볼루션 신경망을 구축할 수 있다.

![conv5](https://user-images.githubusercontent.com/77332628/210292694-5e1cda7f-59ea-46f6-99ab-3b69990d943e.png)

(출처 : https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-8/74)

### 3. 컨볼루션을 신경망에 적용하기
이제 우리가 알고 있는 빌딩 블럭들을 활용해서 새와 비행기를 탐지하는 컨볼루션 신경망을 만들어보자. 


```python
model = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(16,8,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),)
                      # 이후에 더 연결..)
```

1. 첫번 컨볼루션은 3 RGB 채널을 16개의 독립적인 피처를 만들어서 새와 비행기에 대한 저수준의 피처를 찾아내고, Tanh 활성 함수를 적용한다. 
2. 그 결과로 만들어진 16채널의 32x32 이미지를 첫 MaxPool2d를 통해서 16 채널의 16x16 이미지로 다운샘플링하고 8채널의 16x16 출력을 만드는 다른 컨볼루션으로 들어간다. 
3. 이제 출력은 좀 더 높은 수준의 피처를 가질 것이고, 이어서 Tanh 활성함수와 8채널 8x8 출력을 위한 맥스 풀링을 수행한다.

이 신경망은 결국 음의 로그 가능도로 넣을 수 있는 확률 값을 뽑아줘야한다. 확률은 1차원 벡터인 숫자 쌍으로 나와야 하는데, 지금은 멀티 채널의 2차워 피처인 상태다. 그렇기 때문에 8채널의 8x8 이미지를 1차원 벡터로 바꿔서 완전 연결 계층으로 신경망을 마무리해야 한다.


```python
model = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(16,8,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Flatten(), # 8차원 8x8 이미지를 1차원 벡터로 변환
                      nn.Linear(8 * 8 * 8, 32),
                      nn.Tanh(),
                      nn.Linear(32,2))
```

위 코드는 다음 이미지의 신경망을 구현한 코드다.

![conv6](https://user-images.githubusercontent.com/77332628/210292697-de4520dc-ddcf-46fd-ad30-ca896503ba93.png)

(출처 : https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-8/74)

위 코드에서 등장한 nn.Flatten()은 8차원 8x8 이미지를 512 요소를 가진 1차원 벡터로 차원 정보를 변경하는 부분이다. 이는 마지막 MaxPool2d의 출력에 대해 view를 호출하면 해결할 수 있지만, nn.Sequential을 사용할 때는 모듈의 출력을 명시적으로 볼 수 없기 때문에 이번에는 nn.Faltten()으로 대체했다.

구축한 모델의 파라미터 수를 세어보자.


```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
```




    (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])



작은 이미지의 제한된 데이터셋을 생각하면 수긍할 만한 파라미터 수이다. 모델의 용량을 늘리려면 컨볼루션층의 출력 채널 수를 늘려서 연결되는 선형 계층도 함께 키워주면 된다.

이번 글에서는 컨볼루션의 개념과 파이토치로 컨볼루션을 nn.Sequential로 간단하게 구현해봤다. 다음 글에서는 모델을 구축하는 또 다른 방법인 nn.Module 서브클래싱과 함수형 API를 배우고 훈련하는 것에 대해 배워보도록 하겠다.

[<파이토치 딥러닝 마스터:모의암 진단 프로젝트로 배우는 신경망 모델 구축부터 훈련,튜닝,모델 서빙까지>(책만, 2022)을 학습하고 개인 학습용으로 정리한 내용입니다.] 도서보기: https://www.gilbut.co.kr/book/view?bookcode=BN003496



