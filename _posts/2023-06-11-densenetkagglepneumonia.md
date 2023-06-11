---
title: '[IC/Kaggle] Pneumonia X-Ray classification - 폐렴 판단하기 🤢'
toc: true
toc_sticky: true
categories:
  - kaggle-imageclassification
---

## DenseNet으로 X-Ray Pneumonia 판단하기

이번 글에서는 DenseNet 모델을 이용해서 가슴 X-Ray 이미지를 보고 Pneumonia(폐렴)인지 아닌지를 판단해본다. 이번 글의 코드는 [**<U>GEORGII SIROTENKO의 Kaggle notebook</U>**](https://www.kaggle.com/code/georgiisirotenko/pytorch-x-ray-transfer-learning-densenet)를 참고했다.

먼저 이번 글에서 필요한 library들을 import하자.


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import os
from sklearn.model_selection import train_test_split
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### 1. Dataset 분석

먼저 데이터의 분포를 확인해보자. Normal과 Pneumonia 클래스간의 불균형이 있을 수도 있다.


```python
path = '../input/chest-xray-pneumonia/chest_xray/chest_xray'

train_samplesize = pd.DataFrame.from_dict(
                    {'Normal': [len([os.path.join(path+'/train/NORMAL', filename) 
                     for filename in os.listdir(path+'/train/NORMAL')])], 
                     'Pneumonia': [len([os.path.join(path+'/train/PNEUMONIA', filename) 
                        for filename in os.listdir(path+'/train/PNEUMONIA')])]})

sns.barplot(data=train_samplesize).set_title('Training Set Data Imbalance', fontsize=20)
plt.show()
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_3_0.png)
    



Normal 클래스의 data 개수가 훨씬 적기 때문에 data augmentation을 통해서 Normal과 Pneumonia data 개수를 동일하게 맞춰주는 과정을 거친다.


```python
# transformer 정의
transformer = {
    'dataset1': transform.Compose([transform.Resize(255),
                                            transform.CenterCrop(224),
                                            transform.RandomHorizontalFlip(),
                                            transform.RandomRotation(10),
                                            transform.RandomGrayscale(),
                                            transform.RandomAffine(translate=(0.05,0.05), degrees=0),
                                            transform.ToTensor()
                                           ]),
    
    'dataset2' : transform.Compose([transform.Resize(255),
                                            transform.CenterCrop(224),
                                            transform.RandomHorizontalFlip(p=1),
                                            transform.RandomGrayscale(),
                                            transform.RandomAffine(translate=(0.1,0.05), degrees=10),
                                            transform.ToTensor()
                                    
                                           ]),
    'dataset3' : transform.Compose([transform.Resize(255),
                                            transform.CenterCrop(224),
                                            transform.RandomHorizontalFlip(p=0.5),
                                            transform.RandomRotation(15),
                                            transform.RandomGrayscale(p=1),
                                            transform.RandomAffine(translate=(0.08,0.1), degrees=15),
                                            transform.ToTensor()
                                           ]),
}

# Train dataset 다시 정의
dataset1 = ImageFolder(path+'/train', transform=transformer['dataset1'])

dataset2 = ImageFolder(path+'/train', transform=transformer['dataset2'])

dataset3 = ImageFolder(path+'/train', transform=transformer['dataset3'])

norm1, _ = train_test_split(dataset2, test_size=3875/(1341+3875), shuffle=False) # Augmentated Normal data 1341개 추가
norm2, _ = train_test_split(dataset3, test_size=4023/(1341+3875), shuffle=False) # Augmentated Normal data 1193개 추가

dataset = ConcatDataset([dataset1, norm1, norm2])

len(dataset)
```




    7750




```python
print(dataset1.classes)
```

    ['NORMAL', 'PNEUMONIA']


데이터들을 무작위로 뽑아서 시각화해보자. 사실 폐렴 환자와 정상인의 X-Ray 사진의 차이는 뚜렷하지는 않다.


```python
def plot_samples(samples):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,8))
    for i in range(len(samples)):
        image = cv2.cvtColor(imread(samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//5][i%5].imshow(image)
        if i<5:
            ax[i//5][i%5].set_title('Normal', fontsize=20)
        else:
            ax[i//5][i%5].set_title('Pneumonia', fontsize=20)
        ax[i//5][i%5].axis('off')
    

rand_samples = random.sample([os.path.join(path+'/train/NORMAL', filename)
                             for filename in os.listdir(path+'/train/NORMAL')],5) + \
               random.sample([os.path.join(path+'/train/PNEUMONIA', filename)
                             for filename in os.listdir(path+'/train/PNEUMONIA')],5)

plot_samples(rand_samples)
plt.suptitle('Training Set Samples', fontsize=30)
plt.show()
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_8_0.png)
    



### 2. Data 준비하기

Input 데이터에 Validation 데이터가 따로 있지만, 그 개수가 16개로 모델을 정확하게 평가하기에는 너무 적기 때문에 Train dataset의 30%를 따로 Validation을 위한 데이터로 사용할 것이다.


```python
random_seed = 2020
torch.manual_seed(random_seed);

train_ds, val_ds = train_test_split(dataset, test_size=0.3, random_state=random_seed)
len(train_ds), len(val_ds)
```




    (5425, 2325)



이제 DataLoader를 이용해서 data를 모델에 사용할 준비를 한다.


```python
batch_size = 50

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
loaders = {'train':train_dl, 'val':val_dl}
dataset_sizes = {'train':len(train_ds), 'val':len(val_ds)}
```

Data augmentation의 결과를 출력해보자.


```python
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]) ; ax.set_yticks([])
        ax.imshow(make_grid(images[:60], nrow=10).permute(1,2,0))
        break
        
show_batch(train_dl)
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_14_0.png)
    



### 3. DenseNet 학습하기

DenseNet 모델을 불러와서 문제를 풀건데, DenseNet에 대한 개념은 [**<U>DenseNet 논문 리뷰</U>**](https://hamin-chang.github.io/cv-imageclassification/DenseNet/)를 참고하고, DenseNet의 구조를 pytorch code로 구현한 것을 보고 싶다면 [**<U>DenseNet pytorch 구현</U>**](https://hamin-chang.github.io/cv-imageclassification/DenseNet/)를 참고하길 바란다.

먼저 모델의 정확도를 계산하는 함수를 정의한다.


```python
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds
```

DenseNet 모델을 torchvision을 통해 불러온다.


```python
model = torchvision.models.densenet161(pretrained=True)
```

    Downloading: "https://download.pytorch.org/models/densenet161-8d451a50.pth" to /root/.cache/torch/hub/checkpoints/densenet161-8d451a50.pth





      0%|          | 0.00/110M [00:00<?, ?B/s]



DenseNet에는 오직 한개의 fc layer가 존재한다. 불러온 DenseNet161이 1000개의 class가 있는 dataset에서 pretrained 되었지만 pneumonia 데이터는 2개의 class 밖에 없기 때문에 1개의 fc layer만 동결시키지 않고 훈련시킨다.


```python
for param in model.parameters():
    param.requires_grads = False

in_features = model.classifier.in_features

model.classifier = nn.Linear(in_features, 2)
```


```python
losses = {'train':[], 'val':[]}
accuracies = {'train':[], 'val':[]}
```

densenet161을 훈련하기 위한 함수를 정의하고 본격적으로 훈련을 시작한다.


```python
def train(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outp = model(inputs)
                    _, pred = torch.max(outp, 1)
                    loss = criterion(outp, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            if phase == 'train':
                print('Epoch: {}/{}'.format(epoch+1, epochs))
            print('{} - loss:{}, accuracy:{}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                print('Time : {}m {}s'.format((time.time() - since) // 60, (time.time() - since)%60))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
        scheduler.step()
    time_elapsed = time.time() - since
    print('Training Time {}m {}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best Accuracy {}'.format(best_acc))
    
    model.load_state_dict(best_model)
    return model
```


```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)

model.to(device)
epochs=10
model = train(model, criterion, optimizer, scheduler, epochs)
```

    Epoch: 1/10
    train - loss:0.2720892391583887, accuracy:0.9163133640552995
    val - loss:0.1725978432323343, accuracy:0.9389247311827957
    Time : 1.0m 29.173414945602417s
    Epoch: 2/10
    train - loss:0.15560477635552805, accuracy:0.9502304147465438
    val - loss:0.130439769877221, accuracy:0.9595698924731183
    Time : 2.0m 56.904738426208496s
    Epoch: 3/10
    train - loss:0.13122935277251055, accuracy:0.9577880184331797
    val - loss:0.1377651693039043, accuracy:0.9466666666666667
    Time : 4.0m 24.51378083229065s
    Epoch: 4/10
    train - loss:0.11674996891192027, accuracy:0.9594470046082949
    val - loss:0.11094462312757969, accuracy:0.9608602150537634
    Time : 5.0m 52.35554385185242s
    Epoch: 5/10
    train - loss:0.1087935124674151, accuracy:0.9622119815668203
    val - loss:0.11254314501439372, accuracy:0.9591397849462365
    Time : 7.0m 20.362493753433228s
    Epoch: 6/10
    train - loss:0.10562851104868172, accuracy:0.9660829493087557
    val - loss:0.10764719846267853, accuracy:0.963010752688172
    Time : 8.0m 48.05663704872131s
    Epoch: 7/10
    train - loss:0.10343055585013007, accuracy:0.9629493087557603
    val - loss:0.10777980846262747, accuracy:0.9621505376344086
    Time : 10.0m 15.926006555557251s
    Epoch: 8/10
    train - loss:0.09879091670436244, accuracy:0.9695852534562212
    val - loss:0.10655936702925672, accuracy:0.9643010752688173
    Time : 11.0m 43.8165078163147s
    Epoch: 9/10
    train - loss:0.10392717974183197, accuracy:0.9666359447004609
    val - loss:0.1063992977142334, accuracy:0.9643010752688173
    Time : 13.0m 11.65030312538147s
    Epoch: 10/10
    train - loss:0.1010592248857296, accuracy:0.9662672811059908
    val - loss:0.10579956879699103, accuracy:0.963010752688172
    Time : 14.0m 39.30699348449707s
    Training Time 14.0m 39.3079195022583s
    Best Accuracy 0.9643010752688173


이 코드의 저자는 learning rate를 epoch마다 줄이면 더 좋은 성능을 낸다고 한다.


```python
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.to(device)
grad_clip = None
weight_decay = 1e-4
epochs = 10
model = train(model, criterion, optimizer, scheduler, epochs)
```

    Epoch: 1/10
    train - loss:0.09891576016180148, accuracy:0.9647926267281106
    val - loss:0.07548122516562862, accuracy:0.9711827956989247
    Time : 1.0m 34.82368564605713s
    Epoch: 2/10
    train - loss:0.02249502353901301, accuracy:0.991889400921659
    val - loss:0.05300364624308322, accuracy:0.9819354838709677
    Time : 3.0m 9.286401748657227s
    Epoch: 3/10
    train - loss:0.013685281556244502, accuracy:0.9950230414746544
    val - loss:0.06332757886779565, accuracy:0.9776344086021506
    Time : 4.0m 44.042033672332764s
    Epoch: 4/10
    train - loss:0.006319989015730227, accuracy:0.9987096774193548
    val - loss:0.06202164890403579, accuracy:0.9827956989247312
    Time : 6.0m 18.736663579940796s
    Epoch: 5/10
    train - loss:0.005722267822504515, accuracy:0.9981566820276497
    val - loss:0.055847952166454544, accuracy:0.9823655913978495
    Time : 7.0m 53.3470664024353s
    Epoch: 6/10
    train - loss:0.0014783768939250794, accuracy:0.999815668202765
    val - loss:0.064247196903252, accuracy:0.9823655913978495
    Time : 9.0m 28.04390859603882s
    Epoch: 7/10
    train - loss:0.0006718468114066186, accuracy:1.0
    val - loss:0.05058621831556847, accuracy:0.9840860215053764
    Time : 11.0m 2.7737956047058105s
    Epoch: 8/10
    train - loss:0.0007421334864798553, accuracy:0.999815668202765
    val - loss:0.06641483131106334, accuracy:0.9853763440860215
    Time : 12.0m 38.08850932121277s
    Epoch: 9/10
    train - loss:0.0004901284807165882, accuracy:1.0
    val - loss:0.05573757034196, accuracy:0.9853763440860215
    Time : 14.0m 13.110046863555908s
    Epoch: 10/10
    train - loss:0.00338567983972978, accuracy:0.9985253456221198
    val - loss:0.09718256729644953, accuracy:0.967741935483871
    Time : 15.0m 47.73419165611267s
    Training Time 15.0m 47.73462271690369s
    Best Accuracy 0.9853763440860215


### 4. Accuracy와 Loss 시각화

이제 훈련의 결과를 그래프로 나타낸다.


```python
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs*2+1))
ax1.plot(epoch_list, accuracies['train'], label='Train Accuracy')
ax1.plot(epoch_list, accuracies['val'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs*2+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, losses['train'], label='Train Loss')
ax2.plot(epoch_list, losses['val'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs*2+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_28_0.png)
    



### 5. Test

이제 새로운 data를 학습된 densenet161을 이용해서 테스트해보자.


```python
def validation_step(batch):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    acc, preds = accuracy(out, labels)
    
    return {'val_loss': loss.detach(), 'val_acc':acc.detach(),
           'preds' : preds.detach(), 'labels': labels.detach()}


def test_prediction(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    # combine predictions
    batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]
    # combine labels
    batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]
    
    return {'test_loss':epoch_loss.item(), 'test_acc':epoch_acc.item(),
           'test_preds':batch_preds, 'test_labels':batch_labels}


@torch.no_grad()
def test_predict(model, test_loader):
    model.eval()
    # testing for each batch
    outputs = [validation_step(batch) for batch in test_loader]
    results = test_prediction(outputs)
    print('test_loss : {:.4f}, test_acc : {:.4f}'.format(results['test_loss'], results['test_acc']))
    
    return results['test_preds'], results['test_labels']
```

이제 test dataset을 준비하고, 테스트한다.


```python
testset = ImageFolder(path + '/test', transform=transform.Compose([transform.Resize(255),
                                                                  transform.CenterCrop(224),
                                                                  transform.ToTensor()]))

test_dl = DataLoader(testset, batch_size=256)
model.to(device)
preds, labels = test_predict(model, test_dl)
```

    test_loss : 0.5912, test_acc : 0.9098


Test 결과를 시각화해보자.


```python
idxs = torch.tensor(np.append(np.arange(start=0, stop=5, step=1),
                             np.arange(start=500, stop=505, step=1)))

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,14))

for c,i in enumerate(idxs):
    img_tensor, label = testset[i]
    ax[c//5][c%5].imshow(img_tensor[0,:,:], cmap='gray')
    ax[c//5][c%5].set_title('Label : {}\nPrediction : {}'.format(testset.classes[label],
                                                                testset.classes[preds[i]]),
                                                         fontsize=25)
    ax[c//5][c%5].axis('off')
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_34_0.png)
    



추가적으로, 학습한 모델이 test 데이터를 예측할 때의 얼마나 confident한지 알아볼 수 있다.


```python
fig, ax = plt.subplots(figsize=(8,12), ncols=2, nrows=8)

for row in range(8):
    img, label = testset[row]
    pred = torch.exp(model(img.to(device).unsqueeze(0)))
    class_name = ['NORMAL','PNEUMONIA']
    classes = np.array(class_name)
    pred = pred.cpu().data.numpy().squeeze()
    ax[row][0].imshow(img.permute(1,2,0))
    ax[row][0].set_title('Real : {}'.format(class_name[label]))
    ax[row][0].axis('off')
    ax[row][1].barh(classes, pred)
    ax[row][1].set_aspect(0.1)
    ax[row][1].set_yticks(classes)
    ax[row][1].set_yticklabels(classes)
    ax[row][1].set_title('Predicted Class')
    ax[row][1].set_xlim(0, 1.)
    plt.tight_layout()
    
```




    
![png](pytorch-x-ray-transfer-learning-densenet_files/pytorch-x-ray-transfer-learning-densenet_36_0.png)
    


