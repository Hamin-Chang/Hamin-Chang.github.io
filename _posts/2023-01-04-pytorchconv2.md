---
title : '[CV/Pytorch] 파이토치로 컨볼루션 구현하기 2 🌐'
layout: single
toc: true
toc_sticky: true
categories:
  - CVBasic
---
## 파이토치로 구현하는 컨볼루션(nn.Module 서브클래싱, 함수형 API, 모델 저장, GPU 훈련)

### 1. 나만의 모듈 만들기 (nn.Module 서브클래싱)
신경망을 구축하다 보면 기존에 있는 모듈에서 지원하지 않는 연산들이 필요할 때가 있다. 예를 들면 이 글에서 다룰 잔차 연결을 구현하기 위해서는 직접 모듈을 만들어야 한다. nn.Module의 서브클래스를 직접 만들어서 이미 만들어져 있는 모듈이나 nn.Sequential처럼 사용할 수 있도록 해보자.

nn.Sequential은 계층 뒤에 다른 계층을 붙이는 단순한 일을 수행하기 때문에 더 유연한 모델을 만들려면 nn.Sequential 대신 다른 방법을 사용해야 하는데, 파이토치에서는 nn.Module 서브클래싱으로 모델에서 어떤 연산이든 수행 가능하게 해준다. nn.Module을 서브클래싱하려면 먼저 forward 함수를 정의해서 모듈로 입력을 전달하고 출력을 반환하게 해야 한다. 이 부분이 모듈의 연산을 정의하는 부분이다. 참고로 파이토치에서는 표준 torch 연산을 사용하기만 한다면 자동미분 기능이 자동으로 역방향 경로를 만들기 때문에 nn.Module에서는 backward가 필요없다.

우리가 정의할 연산도 결국은 컨볼루션 같은 이미 존재하는 모듈이나 커스텀 모듈을 사용할 것이다. 이 서브 모듈을 포함하려면 생성자  __ init __ 에 정의하고 self에 할당해서 forward 함수에서 사용할 수 있게 만들면 된다. 모든 코드에 앞서 super().__ init __을 호출하는 것을 잊지 말자! (잊어버리면 파이토치가 알려주긴 한다.)

먼저 저번 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/pytorchconv1/))에서 구축한 우리의 신경망을 서브모듈로 작성해보자. 먼저 nn.Conv2d와 nn.Linear, 활성 함수 등 nn.Sequential에 넘겼던 모든 모듈들을 생성자에 넣고 이 인스턴스를 forward에서 순서대로 사용하자.


```python
''' 저번글에서 Sequential로 구축한 모델
model = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(16,8,kernel_size=3,padding=1),
                      nn.Tanh(),
                      nn.MaxPool2d(2),
                      nn.Flatten(), <== 밑에선 forward에서 view로 구현
                      nn.Linear(8 * 8 * 8, 32),
                      nn.Tanh(),
                      nn.Linear(32,2))
'''
```


```python
import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
    self.act1 = nn.Tanh()
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
    self.act2 = nn.Tanh()
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(8*8*8,32)
    self.act3 = nn.Tanh()
    self.fc2 = nn.Linear(32,2)

  def forward(self,x):
    out = self.pool1(self.act1(self.conv1(x)))
    out = self.pool2(self.act2(self.conv2(out)))
    out = out.view(-1,8*8*8) # 위의 코드에서 Flatten()을 구현
    out = self.act3(self.fc1(out))
    out = self.fc2(out)

    return out
```

Net 클래스는 nn.Sequential으로 구현했던 모델과 같은 서브모듈이다. 다만 명시적으로 forward 함수를 작성했기 때문에 self.pool2의 출력을 직접 view를 통해서 BxN 벡터로 만들 수 있었다. 배치에 얼마나 많은 샘플이 들어있는지 알 수 없기 때문에 view의 배치 차원을 -1로 설정했다.

다음 이미지는 Net 클래스로 구축한 신경망이다.

![conv1](https://user-images.githubusercontent.com/77332628/210529470-7c2ae818-1004-4ca3-924b-c1cd3f2dcb2d.png)

(출처 : https://livebook.manning.com/book/deep-learning-with-pytorch/chapter-8/122)

분류 신경망의 목적은 일반적으로 큰 수의 픽셀을 가진 이미지에서 출발해 정보를 압축해가면서 분류 클래스의 확률 벡터로 만들어 가는 것이다. 이 목적에 따라서 위에서 구축한 모델에 대해 두가지 설명한 점이 있다. 

1. 첫째로, 중간에 나타나는 값의 개수가 점점 줄어드는 모습에 우리가 목표하는 바가 반영되어 있다. 컨볼루션의 채널 수가 점점 줄어들고, 풀링에서 픽셀 수가 줄어들면서 선형 계층에서는 입력 차원보다 낮은 차원을 출력한다. 
2. 두번째로는, 최초의 컨볼루션에서는 예외적으로 입력 크기에 대해 출력 크기가 줄어들지 않는다는 점이다. 위 이미지에서 첫번째 컨볼루션을 본다면 채널이 3에서 16으로 늘어났다. 

#### 1.1 파이토치가 파라미터와 서브모듈을 유지하는 방법
앞서의 코드에서처럼 생성자 안에서 nn.Module의 속성에 nn.Module의 인스턴스를 할당하면 재미있게도 모듈이 서브모듈로 등록된다. 그래서 이렇게 하면 Net은 사용자의 추가 행위 없이 서브모듈의 파라미터에 접근할 수 있게 된다.


```python
model = Net()

numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
```




    (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])



parameters()는 생성자에서 속성으로 할당된 모든 서브모듈을 찾아 이들의 parameters()를 재귀적으로 호출한다. 서브 모듈이 얼마나 중첩되어 있든지 간데 어떤 nn.Module의 모든 자식 파라미터의 리스트에 접근할 수 있다. 이제 모듈을 직접 만드는 법을 배웠는데, Net 클래스 구현을 되돌아보면 파라미터가 없는 nn.Tanh나 nn.MaxPool2d 같은 서브 모듈은 굳이 등록할 필요 없이 view 호출처럼 그냥 forward 함수에서 직접 호출하는게 더 쉽지 않을까?

#### 1.2 함수형 API
당연히 파라미터가 없는 nn.Tanh나 nn.MaxPool2d 같은 서브 모듈은 굳이 등록할 필요 없이 view 호출처럼 그냥 forward 함수에서 직접 호출하는게 더 쉽다! 이런 이유로 파이토치는 모든 nn 모듈에 대한 **함수형 (functional)**을 제공한다. 여기서 **함수형**이란 내부 파라미터 없이 출력값이 전적으로 입력값에 의존한다는 것이다. torch.nn.functional은 nn에서 찾을 수 있는 모듈과 동일한 함수를 많이 제공한다. 다만 함수형이기 때문에 호출시 인자로 입력값 외에 파라미터도 받는다.

다시 모델로 돌아가서, 선형 계층과 컨볼루션 계층은 nn 모듈을 사용해서 Net이 훈련하는 동안 파라미터를 관리하는 것이 당연하다. 하지만 풀링과 활성 함수 같은 경우 파라미터가 없기 때문에 함수형으로 대체하는 것이 가능하다. 함수형 API 를 사용해서 모듈을 다시 만들어보자.


```python
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1) 
    self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1) 
    self.fc1 = nn.Linear(8 * 8 * 8, 32)
    self.fc2 = nn.Linear(32, 2)
        
  def forward(self, x):
      out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)  # 함수형 사용
      out = F.max_pool2d(torch.tanh(self.conv2(out)), 2) # 함수형 사용
      out = out.view(-1, 8 * 8 * 8)
      out = torch.tanh(self.fc1(out))
      out = self.fc2(out)
      return out
```

훨씬 간결한 Net 클래스가 만들어졌다. 생성자에서의 초기화를 위해 파라미터가 필요한 모듈의 인스턴스를 만들어 두는 부분은 여전히 필요하다.

함수적인 방식으로 말미암아 nn.Module API가 무엇인지도 분명해진다. Module은 Parameter라는 모습으로 상태를 저장하고 순방향 전달에 필요한 명령 집합을 서브 모듈로 저장하는 컨테이너로 정의할 수 있다.

언제 함수형을 사용하고 언제 모듈 API를 사용할지는 취향의 문제이다. 보통 신경망이 매우 단순해서 nn.Sequential이 쓰고 싶어진다면 모듈, 직접 순방향 흐름을 작성하는 경우라면 파라미터로 상태를 저장할 필요가 없는 함수형 API를 사용하는 편이 낫다.

이제 모델이 동작하는지 점검하고 훈련 루프로 넘어가보자.


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

img, _ = cifar2[0]

model = Net()
model(img.unsqueeze(0))
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data-unversioned/p1ch7/cifar-10-python.tar.gz





      0%|          | 0/170498071 [00:00<?, ?it/s]



    Extracting ../data-unversioned/p1ch7/cifar-10-python.tar.gz to ../data-unversioned/p1ch7/
    Files already downloaded and verified





    tensor([[-0.0238,  0.1660]], grad_fn=<AddmmBackward0>)



숫재 2개를 얻은 것으로 보아 정보가 제대로 흘러가는 것으로 보인다. 

### 2. 컨볼루션 훈련시키기
이제 우리가 구축한 컨볼루션 신경망을 훈련시켜보자. 훈련루프를 정의할건데, 이는 이전 글 들([**링크**](https://hamin-chang.github.io/pytorchBasic))에서 다룬 훈련 루프와 매우 유사하다. 이번 글에서는 정확도 향상을 위해 훈련루프를 조금씩 수정하고, GPU에서 훈련 루프를 돌리는 법에 대해 다룰것이다. 먼저 훈련루프를 정의해보자. 

먼저, 우리가 만든 컨볼루션 신경망은 중첩된 두개의 루프를 가지고 있다. 바깥 루프는 **에포크** 단위로 돌고, 안쪽 루프는 **Dataset**에서 배치를 만드는 **DataLoader** 단위로 돈다. 각 루프에서 우리는,

1. 모델에 입력값을 주입하고, (순방향 전달)
2. 손실값을 계산하고, (순방향 전달)
3. 이전 기울기를 0으로 초기화하고,
4. loss.backward()를 호출해서 모든 파라미터에 대한 손실값의 기울기를 계산한다. (역방향 전달)
5. 옵티마이저를 통해 손실값을 낮추도록 파라미터를 조정한다.


```python
import datetime # 시간 경과를 보기 위한 모듈

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
  for epoch in range(1, n_epochs + 1):  # 에포크 단위로 도는 바깥 루프
    loss_train = 0.0
    for imgs,labels in train_loader:  # DataLoader 단위로 도는 안쪽 루프

      outputs = model(imgs) # 모델에 배치 주입

      loss = loss_fn(outputs,labels) # 최소화하려는 손실값 계산

      optimizer.zero_grad() # 이전 기울기 0 초기화

      loss.backward() # 신경망이 학습할 모든 파라미터에 대한 기울기 계산 (역전파)

      optimizer.step() # 파라미터 조정
 
      loss_train += loss.item()  # 에포크 동안 확인한 손실값을 모두 더함. (item() 사용해서 파이썬 수로 변환)

    if epoch == 1 or epoch % 10 == 0 :
      print(f'{datetime.datetime.now()} Epoch {epoch} , Training loss {loss_train/len(train_loader)}')
```

훈련 루프를 정의햇으니 이제 손싥밧을 출력하면서 100에포크만큼 훈련시켜보자.


```python
import torch.optim as optim
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), lr =1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(n_epochs =100,
              optimizer=optimizer,
              model = model,
              loss_fn = loss_fn,
              train_loader=train_loader)
```

    2023-01-04 09:40:06.168752 Epoch 1 , Training loss 0.5785413316101026
    2023-01-04 09:40:36.191634 Epoch 10 , Training loss 0.3315784791662435
    2023-01-04 09:41:08.679297 Epoch 20 , Training loss 0.2908569341822035
    2023-01-04 09:41:41.678284 Epoch 30 , Training loss 0.2648145875353722
    2023-01-04 09:42:14.321410 Epoch 40 , Training loss 0.2432608529450787
    2023-01-04 09:42:54.410752 Epoch 50 , Training loss 0.22328709488271908
    2023-01-04 09:43:29.434575 Epoch 60 , Training loss 0.20595651433156553
    2023-01-04 09:44:05.834183 Epoch 70 , Training loss 0.19092911999126908
    2023-01-04 09:44:42.441434 Epoch 80 , Training loss 0.1779184837345105
    2023-01-04 09:45:19.876821 Epoch 90 , Training loss 0.162368363754195
    2023-01-04 09:45:58.648592 Epoch 100 , Training loss 0.15085834886427898


훈련을 통해서 굉장히 낮은 훈련 손실값을 얻어냈다! 이제 훈련셋과 검증셋으로 정확도를 측정해보자.


```python
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                         shuffle=False)

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) 
                total += labels.shape[0]  
                correct += int((predicted == labels).sum())  

        print("Accuracy {}: {:.2f}".format(name , correct / total))

validate(model, train_loader, val_loader)
```

    Accuracy train: 0.94
    Accuracy val: 0.89


약 82%의 검증 정확도를 보였던 이전 글 ([**링크**](https://hamin-chang.github.io/pytorchbasic/birdplane2/#5-%EB%B6%84%EB%A5%98%EA%B8%B0-%ED%9B%88%EB%A0%A8)) 의 완전 연결 모델보다는 좋은 검증 정확도인 89%를 얻었다. 심지어 더 적은 양의 파라미터를 사용했다. 이미지에 있는 물체를 인식하는 작업이 지역성이나 평행이동 불변성에 대응하여 더 일반화 되었다.

#### 2.1 모델 파라미터 저장하기
이 정도면 만족할 만한 모델인 것 같으니, 저장해보는 것도 좋을 것 같다. 저장은 쉽다. 파일로 모델을 저장해보자.


```python
data_path = '/content/covolution_example_model'
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')
```

이제 birds_vs_airplanes.pt 파일에 model의 모든 파라미터가 들어있다. 즉 두개의 컨볼루션 모듈과 두개의 선형 모델에 포함된 가중치와 편향값을 저장하고 있지만, 모델 구조는 포함되어 있지 않다. 따라서 실제 서비스 목적으로 모델을 제공하려면  model 클래스도 저장했다가 인스턴스로 만들고 파라미터를 파일에서 읽어와야 한다.


```python
loaded_model = Net()
loaded_model.load_state_dict(torch.load(data_path+'birds_vs_airplanes.pt'))
```




    <All keys matched successfully>



### 3. GPU에서 훈련시키기
이제 신경망도 구축했고 훈련도 할 수 있다. 이제 훈련을 좀 더 빠르게 하고 싶은 욕심이 생긴다. GPU에서 훈련을 시키면 훈련 속도를 더 높일 수 있다. .to 메소드를 사용하면 데이터 로더에서 얻은 텐서를 GPU로 옮길 수 있다. 이러면 연산 작업은 자연스럽게 GPU에서 이루어지기 떄문에 당연히 파라미터도 GPU로 옮겨야 한다. 다행히 nn.Module에도 .to 메소드가 있어서 파라미터를 GPU로 옮길 수 있다.

Module.to와 Tensor.to는 조금 다르다. Module.to는 모듈 인스턴스 자체를 수정하지만 Tensor.to는 새 텐서를 반환한다. 따라서 파라미터를 원하는 디바이스로 이동시킨 후 옵티마이저를 만드는 식의 구현을 추천한다. 또한 가능하다면 GPU로 옮겨서 작업하는 방식도 추천한다. torch.cuda.is_available 값에 따라 device 변수를 설정하는 것이 좋은 구현 패턴이다.


```python
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) # GPU 사용이 가능하면 device를 GPU로 바꾼다.
print(f'Training on device {device}')
```

    Training on device cuda


이러면 Tensor.to 메소드로 데이터 로더가 올려준 텐서를 GPU로 옮기도록 훈련 루프를 수정할 수 있다. 코드는 입력은 GPU로 옮기는 부분을 제외하면 위의 훈련 루프와 완전히 동일하다.


```python
import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)  # imgs와
            labels = labels.to(device=device) # labels를 GPU로 옮긴다.
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
```

검증 루프 함수에서도 동일한 방식의 수정이 필요하다. 이후 모델을 인스턴스화하고 device로 옮긴 후 이전처럼 실행하면 된다.


```python
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)

model = Net().to(device=device)  # 모델(파라미터)을 GPU로 옮긴다.
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)
```

    2023-01-04 09:46:12.012417 Epoch 1, Training loss 0.593362849419284
    2023-01-04 09:46:15.465845 Epoch 10, Training loss 0.33493344400339065
    2023-01-04 09:46:19.075174 Epoch 20, Training loss 0.29751644032016683
    2023-01-04 09:46:22.761621 Epoch 30, Training loss 0.2718368280370524
    2023-01-04 09:46:26.587916 Epoch 40, Training loss 0.2462609575432577
    2023-01-04 09:46:30.471063 Epoch 50, Training loss 0.22344799505867016
    2023-01-04 09:46:34.313723 Epoch 60, Training loss 0.20529802841175893
    2023-01-04 09:46:38.406868 Epoch 70, Training loss 0.1884742724430409
    2023-01-04 09:46:42.320554 Epoch 80, Training loss 0.17566043271380624
    2023-01-04 09:46:46.093160 Epoch 90, Training loss 0.16446289441483036
    2023-01-04 09:46:50.233325 Epoch 100, Training loss 0.14804159472607503


신경망을 저장하고 모델의 가중치를 읽어올 때 문제가 있다. 파이토치는 가중치를 저장할 때 어떤 디바이스를 기억해뒀다가 가중치를 읽어드릴 때도 그 디바이스를 사용하기 때문에, GPU에 있던 가중치는 나중에 GPU로 읽어드린다. 나중에 어떤 디바이스에서 모델을 돌릴지 모르기 때문에 신경망을 CPU로 옮긴 후에 저장하든가 파일에서 읽어드린 후 CPU로 옮기는 것이 낫다. 이는 간단하게 해결되는데, 가중치를 로딩할 때 파이토치가 기억하는 디바이스 정보를 덮어쓰면 된다. torch.load 인자에 map_loaction 키워드를 전달하면 된다.


```python
loaded_model = Net().to(device=device)
loaded_model.load_state_dict(torch.load('/content/covolution_example_modelbirds_vs_airplanes.pt',
                                        map_location=device)) # load 할 때 device 정보 덮어쓰기
```




    <All keys matched successfully>



이번 글에서는 직접 모듈을 만드는 nn.Module 서브클래싱 방법과, 함수형 API에 대해 배웠고, 직접 만든 컨볼루션 신경망으로 훈련하고, 이 모델을 저장하는 방법과 더욱 빠른 훈련을 위해 GPU에서 훈련하는 방법에 대해 배웠다. 다음 글에서는 모델의 성능을 높이기 위한 여러가지 방법에 대해 다뤄보겠다.

[<파이토치 딥러닝 마스터:모의암 진단 프로젝트로 배우는 신경망 모델 구축부터 훈련,튜닝,모델 서빙까지>(책만, 2022)을 학습하고 개인 학습용으로 정리한 내용입니다.] 도서보기: https://www.gilbut.co.kr/book/view?bookcode=BN003496
