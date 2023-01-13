---
title : '[CV/Pytorch] 파이토치로 컨볼루션 구현하기 3 🌐'
layout: single
toc: true
toc_sticky: true
categories:
  - pytorchCV
---

## 파이토치로 구현하는 컨볼루션 (모델 너비 늘리기, 깊은 모델 만들기(잔차 연결), L2 정규화, 드랍아웃, 배치 정규화)

저번 글 ([**링크**](https://hamin-chang.github.io/pytorchCV))에서 우리는 nn.Module 서브클래싱을 통해서 이미지를 분류하는 컨볼루션 신경망을 만들고나서 훈련시키고 모델은 GPU에서 훈련 시키는 법을 배웠다. 하지만 사실 새와 비행기를 분류하는 것은 그다지 복잡한 문제가 아니었다. 그렇다면 더 복잡한 문제를 해결하기 위해서는 어떻게 해야할까? 이번 글에서는 여러가지 개념적인 도구들을 소개해서 모델을 설계하고 업그레이드 하는 방법들에 대해서 다뤄보겠다.

### 1. 메모리 용량 늘리기 : 너비

모델의 차원 정보들을 먼저 살펴볼건데, 첫번째로 신경망의 **너비 차원**을 다뤄본다. 신경망의 너비 차원은 신경망 계층 내의 뉴런 수 혹은 컨볼루션의 채널 수에 해당하는 값이다. 파이토치에서는 모델의 너비를 쉽게 늘릴 수 있다. 첫 번째 컨볼루션의 출력 채널 수를 늘리고 이어지는 계층도 여기에 맞춰 키워주게 되는데, 이렇게 하면 완전 연결 계층으로 전환되는 forward 함수에도 반영해줘야 한다.


```python
import torch.functional as F
import torch.nn as nn
import torch

class NetWidth(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 16 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

각 계층에서 채널과 피처의 수를 나타내는 값들은 직접적으로 모델의 파라미터 수에 영향을 미친다. 다른 부분이 다 동일한 경우 이러한 변화는 모델의 용량을 증가시킨다. 위의 모델이 가진 파라미터 수가 어떻게 되는지 확인해보자.


```python
model = NetWidth()
params = [p.numel() for p in model.parameters()]

sum(params), params
```




    (38386, [864, 32, 4608, 16, 32768, 32, 64, 2])



모델의 용량이 클수록 모델이 다룰 수 있는 입력이 다양해지지만, 동시에 과적합할 가능성도 커지기 때문에 이는 잘 조절을 해야한다.

### 2. 더 복잡한 구조의 모델 구축하기 : 깊이
모델의 두번째 차원은 **깊이**다. 딥러닝의 '딥'이 모델의 깊이를 얘기하는 것이다. 그럼 모델이 깊은 경우가 얕은 경우보다 무조건 좋은 걸까? 그것은 상황에 따라 다를 것이다. 모델이 깊어질수록 신경망은 더욱 복잡한 함수에 근사할 수 있다. 컴퓨터 비전에서 얕은 신경망은 사진에서 살마의 모습을 식별할 수 있다면 신경망이 깊어질수록 사람의 얼굴과 얼굴 안의 입까지 구별해낼 수 있다. 어떤 입력에 대해 뭔가를 말할 수 있도록 맥락을 이해하려면 모델을 깊게 만들어서 계층적인 정보를 다룰 수 있게 해야한다.

모델의 깊이는 생각보다 간단한 문제가 아니다. 모델에 깊이가 더해질수록 훈련은 수렴하기 어려워진다. 역전파를 매우 깊은 신경망을 수행하게 되면, 파라미터에 대한 손실 함수의 미분은 이전 계층의 손실값과 파라미터 사이의 미분 연산의 체인 연결에서 오는 많은 수로 곱해져야한다. 곱하는 수가 작을 수도 있고 부동소수점 근사 과정에서 작은 값들이 사라져버릴 수도 있다. 다시 말해서 연쇄 곱셈이 길게 이어지게 되면 기울기에 기여하는 파라미터 값이 사라져버려서 훈련이 수렴하지 않는다는 것이다.

#### 2.1 스킵 커넥션
이는 카이밍 히와 공저자들이 만든 **잔차 신경망 (residual network)**인 **ResNet**이 해결했다. 잔차신경망의 기법은 간단하다. 다음 이미지처럼 입력을 계층 블럭의 출력에 연결하는 것이다. 

![res1](https://user-images.githubusercontent.com/77332628/211038108-8d221d15-9102-499d-8865-e10fa9446524.png)

(출처:https://arxiv.org/abs/1512.03385)

바꿔 말하면 표준 피드포워드 경로에 추가적으로 첫번째 활성 함수의 출력을 마지막 부분의 입력으로 사용하는 것이다. 이를 **아이덴티티 매핑**이라고도 부른다. 이렇게 해서 체인으로 길게 연결된 다른 연산들로 곱해질 기회가 줄어들고, 파라미터에 대한 손실값의 편미분으로 손실값에 대한 기울기에 더욱 직접적으로 관여하게 된다. 파이토치로 잔차 연결을 한번 구현해보자.



```python
class NetRes(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out # 다음 계층의 입력값으로 사용할 출력값을 따로 저장
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2) # out1을 입력값으로 더해준다.
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

그럼 파이토치로 컨볼루션 신경망에서 100개 이상의 계층을 만드는 방법은 직접 100개의 잔차 연결을 구현해야할까? 그럼 너무 코드가 복잡해지기 때문에 일반적으로 (Conv2d, ReLU, Conv2d) + skip connection 같은 빌딩 블럭을 정의하고 for 루프를 사용해서 신경망을 동적으로 구축한다.

먼저 컨볼루션과 활성함수 , 스킵 커넥션으로 이뤄진 블럭을 위한 연산을 제공하는 모듈 서브 클래스를 정의하자.


```python
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)  
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans) # 이 글 후반부에 다룬다. (훈련 도중 기울기 값 보존 위한것)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # 표준편차를 가지는 표준 랜덤 요소로 초기화해주는 커스텀 초기화
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x  # 잔차 연결
```

이제 init에서 ResBlock 인스턴스 리스트를 포함한 nn.Sequential을 만들어서 ResBlock 100개를 가진 모델을 만들자. nn.Sequential을 사용하면 한 블럭의 출력을 다음 블럭의 입력으로 사용할 수 있고, 블럭 내의 모든 파라미터를 Net이 볼 수 있게 해준다. 이렇게 만든 sequential을 forward에서 호출해서 100개의 블럭을 거쳐서 출력을 만든다.


```python
class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(  # n_blocks 만큼 반복
            *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

역전파는 잘 진행될 것이지만, 신경망은 수렴하는 데 꽤 걸릴것이다. 이런 방법으로 우리가 해결하려는 문제에 맞는 깊이의 신경망을 구축할 수 있다.

### 3. 모델의 일반화 성능 높이기
모델의 훈련만큼 중요한 것이, 모델이 처음보는 데이터에 대해 얼마나 잘 반응하는지 이다. 일반화 성능을 높이기 위한 기법들에 대해 알아보자.

#### 3.1 가중치 페널티 : 파라미터 제어
일반화를 안정적으로 수행하기 위해서 손실값에 정규화 항을 넣는 방법이 있다. 이 정규화 항을 조작해서 모델의 가중치를 상대적으로 작게 만든다. 즉, 훈련을 통해서 증가할 수 있는 크기를 제한하는 것이다. 큰 가중치에 페널티를 부과하는 셈이다. 이렇게 하면 손실값은 개별 샘플에 맞춰서 얻는 이득이 줄어들게 된다. 가장 유명한 정규화 항으로 L2와 L1 정규화 항이 있는데, L2 정규화는 모델의 모든 가중치에 대한 제곱합이고, L1 정규화는 모델의 모든 가중치 절댓값의 합이다. 이번 글에서는 L2 정규화에 대해서만 다룬다.

L2 정규화는 가중치 감쇠라고도 한다. 손실 함수에 L2 정규화를 더하는 것은 최적화(훈련) 단계에서 현재 값에 비례해서 각 가중치를 줄이는 역할을 한다. 이는 편향값 같은 신경망의 모든 파라미터에 적용된다. 파이토치에서 손실값에 항을 하나 추가해서 이를 손쉽게 구현할 수 있다.


```python
import datetime

def training_loop_l2reg(n_epochs, optimizer, model, loss_fn,
                        train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs
            labels = labels
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()  # pow(2.0)으로 모든 가중치를 제곱
                          for p in model.parameters())  # L1 정규화인 경우 pow(2.0)을 abs()로 교체
            loss = loss + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
```

#### 3.2 드랍아웃 : 입력 하나에 과의존 하지 않기
**드랍아웃 (Dropout)**에 대한 개념은 굉장히 단순하다. 훈련을 반복할 때마다 신경망의 뉴런 출력 중 랜덤으로 몇개를 0으로 만드는 작업을 수행하는 것이다. 이 방법을 활용하면, 매 훈련마다 조금씩 다른 적용되는 뉴런의 필터가 다르기 때문에, 신경망이 각 입력 샘플을 암기하려는 기회를 줄이게 되어서 과적합을 방지한다. 데이터 증강과 비슷한 효과를 내지만, 증강과는 다르게 신경망 전체에 이러한 효과를 낸다.

![res2](https://user-images.githubusercontent.com/77332628/211038114-799412ff-92cb-4ad2-b954-c6990a1db6eb.png)

(출처 :  https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e)

파이토치에서는 nn.Dropout 모듈을 여러 계층의 컨볼루션 모듈 사이에 넣어서 드롭 아웃을 구현할 수 있다. 모듈의 인자에는 얼마나 많은 비율의 입력을 0으로 만들지 결정하는 인자를 입력하면 된다.


```python
class NetDropout(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)  # 40%의 입력을 0으로 만든다.
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = self.conv2_dropout(out)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

드롭아웃은 훈련 중에 활성화되고 훈련이 끝난 모델을 검증셋을 통해 검증 정확도를 내거나 모델을 제품으로 사용할 때는 드롭 아웃을 비활성화 시켜야 한다. Dropout 모듈의 train 프로퍼티를 통해 제어하면 된다. 이는 model.train() 혹은 model.eval()호출로 전환해서 제어하면 된다. 

### 3.3 배치 정규화 : 활성 함수 억제하기
**배치 정규화 (Batch Normalization)**의 핵심은 입력 범위를 신경망의 활성 함수로 바꿔서 미니 배치가 원하는 분포를 가지게 하는 것이다. 비선형 활성 함수를 활용하면, 함수의 임계 영역에서 입력이 활성 함수에 너무 많이 작용해서 기울기가 소실되고 훈련이 느려지는 상황을 피할 수 있다.

실제로 배치 정규화는 미니 배치 샘플을 통해 중간 위치에서 얻은 평균과 표준편차를 사용해서 중간 입력값을 이동하고 범위를 바꾼다. 따라서 모델이 보는 개별 샘플이나 이로 인한 이후의 활성화 단계에서는 랜덤하게 뽑아 만들어진 미니 배치에서 의존한 값의 이도과 범위 조정이 반영되어 있는 상태다. 이 자체로 원칙적인 데이터 증강인 셈이 되는 것이다. 배치 정규화를 다룬 논문에서는 배치 정규화가 드랍아웃의 필요을 없애거나 줄여준다고 주장한다. 

파이토치에서는 nn.BatchNorm2d (1d,3d도 포함)를 사용해서 배치 정규화를 구현하면 된다. 배치 정규화의 목적은 활성 함수의 입력 범위를 조정하는 것이기 때문에 선형 변환(우리의 경우 컨볼루션) 뒤에 위치시키면 된다.


```python
class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1) # 배치 정규화
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, 
                               padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2) # 배치 정규화
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
```

배치 정규화도 드랍아웃처럼 훈련 때와 추론(검증)때 각기 다르게 동작해야 한다. 추론 시에 출력은 모델이 이미 봤던 다른 입력의 통계에 의존하는 특정 입력을 위한 것이 되어서는 안된다. 미니 배치가 실행되면 현재의 미니 배치에 대한 평균과 표준편차를 구하는 것과 더불어 파이토치가 전체 데이터셋에 대한 평균과 표준편차도 대략적으로 업데이트하기 때문에 사용자는 추론시 model.eval()을 명시해서 모델이 배치 정규화 모듈을 가지는 경우 추정값을 고정하고 정규화에 사용하기만 해야한다. 동작중인 추정을 해제하고 다시 미니 배치 통계를 추정해가길 원하면 다시 model.train()을 호출하면 된다.

이번 글에서는 모델의 성능을 높이기 위한 여러가지 기법들에 대해서 알아봤다. 이제는 파이토치로 컴퓨터 비전을 위한 신경망을 구축하는 기본적인 방법들에 대한 것은 어느 정도 아는 상태가 되었다. 이제 본격적으로 파이토치로 구현하는 컴퓨터 비전 모델들을 알아보자!!

[<파이토치 딥러닝 마스터:모의암 진단 프로젝트로 배우는 신경망 모델 구축부터 훈련,튜닝,모델 서빙까지>(책만, 2022)을 학습하고 개인 학습용으로 정리한 내용입니다.] 도서보기: https://www.gilbut.co.kr/book/view?bookcode=BN003496
