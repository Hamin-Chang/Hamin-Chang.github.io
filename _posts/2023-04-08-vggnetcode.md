---
title : '[IC/Pytorch] 파이토치로 VGGNet 구현하기 👇' 
layout: single
toc: true
toc_sticky: true
categories:
  - ICCode
---

## Pytorch로 VGGNet 구현하기

이번 글에서는 VGGNet을 실제 pytorch 코드로 구현해본다. [<U>roytravel의 repository</U>](https://github.com/roytravel/paper-implementation/blob/master/vggnet/vggnet.pyhttps://github.com/roytravel/paper-implementation/blob/master/vggnet/vggnet.py)의 코드를 사용한다.

논문에서 사용한 깊이가 서로 다른 4개의 모델을 구축하는데, 이는 CONFIGURES를 통해서 모델마다 깊이가 다르도록 설계한다.

이미지1


```python
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

CONFIGURES = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, num_classes : int = 1000, init_weights : bool = True, vgg_name : str = 'VGG19') -> None :
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.features = self._make_layers(CONFIGURES[vgg_name], batch_norm = False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.classifier = nn.Sequential(
                nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=num_classes)
        )
        if init_weights:
            self._init_weight()
    
    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # fan out: neurons in output layer
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # return torch.Size([2, 1000])
        x = self.classifier(x)
        return x
    
    def _make_layers(self, CONFIGURES:list, batch_norm: bool = False) -> nn.Sequential :
        layers : list = []
        in_channels = 3
        for value in CONFIGURES:
            if value == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else : 
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=value, kernel_size=3, padding=1)
                if batch_norm :
                    layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
                else :
                    layers += [conv2d, nn.ReLU(inplace=True)]
                
                in_channels += value
        return nn.Sequential(*layers)
```

이제 위의 코드로 구축한 VGGNet 중 VGG19를 사용해서 STL10 dataset을 사용한 간단한 학습 과정을 구현한다.


```python
if __name__ == '__main__':
    # hyper-parameter 설정
    seed = torch.initial_seed()
    BATCH_SIZE = 24
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    CHECKPOINT_PATH = './checkpoint/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # VGG19 모델 사용
    vggnet = VGGNet(num_classes=10, init_weights=True, vgg_name="VGG19")
    
    # data augmentation 수행 
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224), # input image를 224로 resize & crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48235, 0.45882, 0.40784), std=(1.0/255.0, 1.0/255.0, 1.0/255.0))
        ])
    
    # train & test dataset 준비
    train_dataset = datasets.STL10(root='./data',download=True, split='train', transform=preprocess)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.STL10(root='./data', download=True, split='test', transform=preprocess)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # loss function, optimizer, lr scheduler 설정
    criterion = nn.CrossEntropyLoss().cuda() 
    optimizer = torch.optim.SGD(lr=LEARNING_RATE, weight_decay=5e-3, params=vggnet.parameters(), momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    vggnet = torch.nn.parallel.DataParallel(vggnet, device_ids=[0,1]) # GPU 2개로 분산 
    
    start_time = time.time()
    labels = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    for epoch in range(NUM_EPOCHS):
        print("lr: ", optimizer.param_groups[0]['lr'])
        for idx, _data in enumerate(train_dataloader, start=0):
            images, labels = _data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = vggnet(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if idx % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output ,1)
                    accuracy = torch.sum(preds == labels)
                    print ('Epoch: {} \tStep: {}\tLoss: {:.4f} \tAccuracy: {}'.format(epoch+1, idx, loss.item(), accuracy.item() / BATCH_SIZE))
                    scheduler.step(loss)
        
        state = {
            'epoch' : epoch,
            'optimizer' : optimizerim.state_dict(),
            'model' : vggnet.state_dict(),
            'seed' : seed
        }
        if epoch % 10 == 0 : 
            torch.save(state, CHECKPOINT_PATH + 'model_{}.pth'.format(epoch))
                    
```

다음 글에서는 VGGNet을 이용해서 캐글 문제를 푼 사례를 가져와보겠다.

출처 및 참고 문헌 : 

roytravel의 repository (https://github.com/roytravel/paper-implementation/blob/master/vggnet/vggnet.pyhttps://github.com/roytravel/paper-implementation/blob/master/vggnet/vggnet.py)

VGG 논문 (https://arxiv.org/pdf/1409.1556.pdf)
