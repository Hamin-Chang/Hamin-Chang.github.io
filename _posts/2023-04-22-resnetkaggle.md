---
title: '[IC/Kaggle] Garbage Classification - 쓰레기 종류 분류하기 🗑️ '
toc: true
toc_sticky: true
categories:
  - kaggle-imageclassification
---

## ResNet50으로 쓰레기 종류 분류하기

이번 글에서는 사전 훈련된 ResNet50 모델을 사용해서 쓰레기 이미지들에서의 쓰레기들의 종류를 분류한다. ResNet에 대한 개념은 [**<U>ResNet 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/resnet/)를 참고하길 바란다. 코드는 [**<U>AADHAV VIGNESH의 kaggle notebook</U>**](https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy/notebook)의 코드를 사용했다.

### 0. 라이브러리 import & 데이터셋 준비
먼저 필요한 라이브러리들을 import한다.


```python
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
```

데이터셋의 label(쓰레기 종류)를 출력해보자.


```python
data_dir = '/kaggle/input/garbage-classification/Garbage classification/Garbage classification'

classes = os.listdir(data_dir)
print(classes)
```

    ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']


입력 이미지를 resize하고 tensor로 바꿔준다.


```python
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformations)
```

전처리한 입력 이미지 중 하나의 예시를 출력해보자.


```python
import matplotlib.pyplot as plt
%matplotlib inline

def show_sample(img, label):
    print('label:', dataset.classes[label], '(Class No.'+str(label)+')')
    plt.imshow(img.permute(1,2,0))
    
img, label = dataset[555]
show_sample(img, label)
```

    label: glass (Class No.1)





    
![1](https://user-images.githubusercontent.com/77332628/233769162-20bc749b-de07-47dc-b3d9-1ea9f7baaa76.png)    



이제 이미지 데이터셋을 train, valid와 test용으로 나눠준다. 각각 1593, 176, 758개로 나눠주고, DataLoader를 사용해서 데이터를 로드해줄텐데, 이때 하나의 batch당 32개의 입력 데이터를 할당해준다.


```python
random_seed = 42 # 원본 글의 결과값과 같게 하기 위해 seed=42로 설정
torch.manual_seed(random_seed)

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)
```




    (1593, 176, 758)




```python
from torch.utils.data.dataloader import DataLoader
batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# batch의 이미지 시각화
from torchvision.utils import make_grid
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1,2,0))
        break
    
show_batch(train_dl)
```




    
![2](https://user-images.githubusercontent.com/77332628/233769164-d9c5d058-3a17-4a41-ad4f-af61a3897c04.png)    



### 1. Model 준비


```python
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    
    # batch마다 train loss 계산
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # prediction 생성
        loss = F.cross_entropy(out, labels) # cross entropy 손실함수로 loss 계산
        return loss
    
    # batch마다 valid loss, accuracy 계산
    def validation_step(self, batch):
        images, labels = batch
        out = self(images) # prediction 생성
        loss = F.cross_entropy(out, labels) # loss 계산
        acc = accuracy(out, labels) # accuracy 계산
        return {'val_loss': loss.detach(), 'val_acc':acc}
    
    # epoch마다 validation loss, accuracy 계산
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
    
    
    def epoch_end(self, epoch, result):
        print ('Epoch {} : train_loss : {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(
        epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
```

pre-trained된 ResNet50 모델을 사용한다. 


```python
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # pre-trained ResNet50 불러오기
        self.network = models.resnet50(pretrained=True)
        
        # ResNet50의 마지막 layer fine-tuning
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
model = ResNet()
```

    Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/checkpoints/resnet50-19c8e357.pth





    HBox(children=(FloatProgress(value=0.0, max=102502400.0), HTML(value='')))



    


### 2. GPU 사용하기


```python
def get_default_device():
    # GPU 사용가능 여부 확인
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    # tensor들을 cuda 또는 cpu로 옮기는 함수
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    # cuda 또는 cpu로 data를 옮기는 dataloader 정의
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

device = get_default_device()
device
```




    device(type='cuda')




```python
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)
```




    ResNet(
      (network): ResNet(
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer2): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer3): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (5): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (fc): Linear(in_features=2048, out_features=6, bias=True)
      )
    )



### 3. 모델 훈련시키기

모델을 훈련시키기 위한 함수들을 정의해주고,


```python
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
```


```python
model = to_device(ResNet(), device)
evaluate(model, val_dl)
```




    {'val_loss': 1.7893962860107422, 'val_acc': 0.1215277835726738}



모델을 훈련한 결과를 출력한다.


```python
num_epochs = 8
opt_func = torch.optim.Adam
lr = 5.5e-5

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
```

    Epoch 1 : train_loss : 1.4694, val_loss: 1.2721, val_acc: 0.8333
    Epoch 2 : train_loss : 1.1827, val_loss: 1.1565, val_acc: 0.9340
    Epoch 3 : train_loss : 1.1020, val_loss: 1.1537, val_acc: 0.9115
    Epoch 4 : train_loss : 1.0717, val_loss: 1.1459, val_acc: 0.9045
    Epoch 5 : train_loss : 1.0652, val_loss: 1.1219, val_acc: 0.9288
    Epoch 6 : train_loss : 1.0588, val_loss: 1.1172, val_acc: 0.9479
    Epoch 7 : train_loss : 1.0568, val_loss: 1.1056, val_acc: 0.9601
    Epoch 8 : train_loss : 1.0595, val_loss: 1.0989, val_acc: 0.9601


훈련 결과들(accuracy, training loss, validation loss)을 그래프로 출력해보자.


```python
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs # of epochs')
    
plot_losses(history)
```




    
![3](https://user-images.githubusercontent.com/77332628/233769165-d39b149a-a3ae-466f-8301-c3f512b02d8c.png)    




```python
def plot_accuracy(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs # of epochs')

plot_accuracy(history)
```




    
![4](https://user-images.githubusercontent.com/77332628/233769166-89057410-5fb7-43b7-976e-156178a8a2db.png)    



### 4. 모델 테스트하기

이제 모델이 예측한 답이 정답인지 확인해보자. test 데이터셋의 10개의 이미지를 랜덤으로 불러와서 예측값이 맞는지 확인해보자.


```python
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    probs, preds = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]

import random
random_inputs = random.sample(range(1,759),10)

for x in random_inputs:
    img, label = test_ds[x]
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Label: {dataset.classes[label]}")
    plt.show()
    print(f"Predicted: {predict_image(img, model)}")
```




    
![5](https://user-images.githubusercontent.com/77332628/233769167-75eb1a27-4057-4209-b270-dda71347fcee.png)    



    Predicted: paper





    
![6](https://user-images.githubusercontent.com/77332628/233769169-2e5b487a-5b2e-43a4-a709-c6bd22386055.png)    



    Predicted: paper





    
![7](https://user-images.githubusercontent.com/77332628/233769170-a50c6a0b-987b-4214-bef8-b497e431d980.png)    



    Predicted: glass





    

![8](https://user-images.githubusercontent.com/77332628/233769171-d9f1b521-ebdf-46c5-937c-65be12fff901.png)    



    Predicted: plastic





    
![9](https://user-images.githubusercontent.com/77332628/233769173-9ff42d12-13e4-45f9-abf3-d6a63260f4a7.png)    



    Predicted: metal





    
![10](https://user-images.githubusercontent.com/77332628/233769174-8ce4fe12-463b-4658-bf3b-bdf78dcd666f.png)    



    Predicted: plastic





    

![11](https://user-images.githubusercontent.com/77332628/233769175-524fb557-00c4-4069-8478-bfc8ad774d72.png)    



    Predicted: cardboard





    
![13](https://user-images.githubusercontent.com/77332628/233769177-a08ab9a2-69c4-4863-8655-658892f106c7.png)    



    Predicted: paper





    
![14](https://user-images.githubusercontent.com/77332628/233769178-d4f38439-755e-4a24-ad8d-b88a52360ab6.png)    



    Predicted: plastic





    
![15](https://user-images.githubusercontent.com/77332628/233769179-3507252c-565f-42f7-9fca-97d6dd7c8501.png)    



    Predicted: plastic


다 정답이다! 높은 정답률을 보여주는걸로 보아 모델의 성능이 쓸만한것 같다.

출처 및 참고문헌 :

AADHAV VIGNESH의 kaggle notebook (https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy/notebook)


