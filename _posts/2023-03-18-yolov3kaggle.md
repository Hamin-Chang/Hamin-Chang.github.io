---
title: '[Kaggle] Vehicle Detection - YOLO v3로 자동차 탐지 🚗'
toc: true
toc_sticky: true
categories:
  - kaggle-objectdetection
---
## YOLO v3로 자동차 탐지하기

이번 글에서는 캐글에 있는 노트북을 참고해서 실제로 YOLO v3로 객체 탐지를 수행해본다. [<U>CLAUDIO FANCONI의 노트북</U>](https://www.kaggle.com/code/fanconic/yolov3-keras-image-object-detection)을 참고했으며, 참고한 노트북의 작성자도 YOLO v3 모델은 [<U>https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras</U>](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras )에서 가져왔다고 한다. 그리고 이번 글에서는 파이토치가 아닌 케라스로 모델을 구축하는데, 이 점 참고하길 바란다.

### 1. YOLOv3 NET 

먼저 YOLO v3 모델을 케라스로 구축해준다. (https://github.com/experiencor/keras-yolo3 를 참고했다고 한다.)


```python
import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
```

    Using TensorFlow backend.
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /opt/conda/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])



```python
def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
    return add([skip_connection, x]) if skip else x
```

다음에는 YOLO v3모델을 직접 코드로 구현할건데, 다음과 같은 모델의 구조를 구현한다.

![1](https://user-images.githubusercontent.com/77332628/226091694-aa61e82f-979d-42d1-8de4-8f320a2d9a2c.png)


```python
def make_yolov3_model():
    input_image = Input(shape=(None, None, 3))
    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
    skip_36 = x
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
    skip_61 = x
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)
    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])
    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)
    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])
    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)
    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model
            
            
            
    
```

위의 코드가 굉장히 길어서 복잡해 보이지만 FPN 구조를 활용해서 3개의 multi-scale feature map을 얻는 과정에 주목하면 된다.

이제 가중치를 불러오는 Class를 정의한다.


```python
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,  = struct.unpack('i', w_f.read(4))
            minor,  = struct.unpack('i', w_f.read(4))
            revision,  = struct.unpack('i', w_f.read(4))
            if (major*10 + minor) >= 2 and major < 1000 and minor <1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (major>1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]
    
    def load_weights(self, model):
        for i in range(106):
            try :
                conv_layer = model.get_layer('conv_' + str(i))
                print('loading weights of convolution #' + str(i))
                if i not in [81,93,105]:
                    norm_layer = model.get_layer('bnorm_'+str(i))
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    beta = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance
                    weights = norm_layer.set_weights([gamma, beta, mean, var])
                
                if len(conv_layer.get_weights()) > 1 :
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print('no convolution #' + str(i))
    def reset(self):
        self.offset = 0
                    
```

### 2. Visualization Functions 

이제 모델이 예측한 결과를 원본 이미지에 시각화해주는 함수와 class를 정의한다.


```python
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class BoundBox : 
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1 :
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score
    
def _sigmoid(x):
    return 1./(1.+ np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh) : continue
            # 처음 4개의 요소는 x,y,w,h 값
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position x
            y = (row + y) / grid_h # center position y
            w = anchors[2*b + 0] * np.exp(w) / net_w # bbox width
            h = anchors[2*b + 1] * np.exp(h) / net_h # bbox height
            # 마지막 요소는 class probabilites
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/ net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1 :
        if x4 < x1 :
            return 0
        else:
            return min(x2,x4) - x1
    
    else:
        if x2<x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0 :
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0 : continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
```


```python
from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array

# 입력 이미지를 load 하고 prepare 한다.
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# threshold 값에 따른 결과값 얻기
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # label이 threshold 이상일 때만 취급
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    return v_boxes,v_labels,v_scores

# 얻은 결과값 시각화
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # 이미지 load
    data = pyplot.imread(filename)
    # 이미지 시각화
    pyplot.imshow(data)
    # bbox 그리기 위한 axis 설정
    ax = pyplot.gca()
    # 각 bbox 그리기
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # bbox 좌표 얻기
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # bbox의 너비, 높이 구하기
        width, height = x2-x1, y2-y1
        # bbox 그리기
        rect = Rectangle((x1,y1), width, height, fill=False, color='green')
        ax.add_patch(rect)
        # bbox에 해당하는 label과 score 쓰기
        label = '%s (%.3f)' %(v_labels[i], v_scores[i])
        pyplot.text(x1,y1,label, color='red')
    # 결과값 표시하기
    pyplot.show()
```

### 3. Create YOLO v3 Model

이제 위에서 정의한 함수들을 이용해서 모델을 정의하고, 이미 pre-trained된 가중치를 load해서 저장한다.


```python
# model 정의
model = make_yolov3_model()

# load pre-trained weights
weight_reader = WeightReader('../input/lyft-3d-recognition/yolov3.weights')

# 가중치를 모델에 적용
weight_reader.load_weights(model)

# 모델 저장
model.save('model.h5')
```

    loading weights of convolution #0
    loading weights of convolution #1
    loading weights of convolution #2
    loading weights of convolution #3
    no convolution #4
    loading weights of convolution #5
    loading weights of convolution #6
    loading weights of convolution #7
    no convolution #8
    loading weights of convolution #9
    loading weights of convolution #10
    no convolution #11
    loading weights of convolution #12
    loading weights of convolution #13
    loading weights of convolution #14
    no convolution #15
    loading weights of convolution #16
    loading weights of convolution #17
    no convolution #18
    loading weights of convolution #19
    loading weights of convolution #20
    no convolution #21
    loading weights of convolution #22
    loading weights of convolution #23
    no convolution #24
    loading weights of convolution #25
    loading weights of convolution #26
    no convolution #27
    loading weights of convolution #28
    loading weights of convolution #29
    no convolution #30
    loading weights of convolution #31
    loading weights of convolution #32
    no convolution #33
    loading weights of convolution #34
    loading weights of convolution #35
    no convolution #36
    loading weights of convolution #37
    loading weights of convolution #38
    loading weights of convolution #39
    no convolution #40
    loading weights of convolution #41
    loading weights of convolution #42
    no convolution #43
    loading weights of convolution #44
    loading weights of convolution #45
    no convolution #46
    loading weights of convolution #47
    loading weights of convolution #48
    no convolution #49
    loading weights of convolution #50
    loading weights of convolution #51
    no convolution #52
    loading weights of convolution #53
    loading weights of convolution #54
    no convolution #55
    loading weights of convolution #56
    loading weights of convolution #57
    no convolution #58
    loading weights of convolution #59
    loading weights of convolution #60
    no convolution #61
    loading weights of convolution #62
    loading weights of convolution #63
    loading weights of convolution #64
    no convolution #65
    loading weights of convolution #66
    loading weights of convolution #67
    no convolution #68
    loading weights of convolution #69
    loading weights of convolution #70
    no convolution #71
    loading weights of convolution #72
    loading weights of convolution #73
    no convolution #74
    loading weights of convolution #75
    loading weights of convolution #76
    loading weights of convolution #77
    loading weights of convolution #78
    loading weights of convolution #79
    loading weights of convolution #80
    loading weights of convolution #81
    no convolution #82
    no convolution #83
    loading weights of convolution #84
    no convolution #85
    no convolution #86
    loading weights of convolution #87
    loading weights of convolution #88
    loading weights of convolution #89
    loading weights of convolution #90
    loading weights of convolution #91
    loading weights of convolution #92
    loading weights of convolution #93
    no convolution #94
    no convolution #95
    loading weights of convolution #96
    no convolution #97
    no convolution #98
    loading weights of convolution #99
    loading weights of convolution #100
    loading weights of convolution #101
    loading weights of convolution #102
    loading weights of convolution #103
    loading weights of convolution #104
    loading weights of convolution #105



```python
# 위의 코드로 가중치가 적용된 모델이 output 파일에 저장되어 있기 때문에
# 모델을 load해서 사용한다.
from keras.models import load_model
model = load_model('model.h5')
```

    /opt/conda/lib/python3.6/site-packages/keras/engine/saving.py:292: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
      warnings.warn('No training configuration found in save file: '


우리가 정의한 모델을 표시해보자.


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, None, 3 0                                            
    __________________________________________________________________________________________________
    conv_0 (Conv2D)                 (None, None, None, 3 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    bnorm_0 (BatchNormalization)    (None, None, None, 3 128         conv_0[0][0]                     
    __________________________________________________________________________________________________
    leaky_0 (LeakyReLU)             (None, None, None, 3 0           bnorm_0[0][0]                    
    __________________________________________________________________________________________________
    zero_padding2d_1 (ZeroPadding2D (None, None, None, 3 0           leaky_0[0][0]                    
    __________________________________________________________________________________________________
    conv_1 (Conv2D)                 (None, None, None, 6 18432       zero_padding2d_1[0][0]           
    __________________________________________________________________________________________________
    bnorm_1 (BatchNormalization)    (None, None, None, 6 256         conv_1[0][0]                     
    __________________________________________________________________________________________________
    leaky_1 (LeakyReLU)             (None, None, None, 6 0           bnorm_1[0][0]                    
    __________________________________________________________________________________________________
    conv_2 (Conv2D)                 (None, None, None, 3 2048        leaky_1[0][0]                    
    __________________________________________________________________________________________________
    bnorm_2 (BatchNormalization)    (None, None, None, 3 128         conv_2[0][0]                     
    __________________________________________________________________________________________________
    leaky_2 (LeakyReLU)             (None, None, None, 3 0           bnorm_2[0][0]                    
    __________________________________________________________________________________________________
    conv_3 (Conv2D)                 (None, None, None, 6 18432       leaky_2[0][0]                    
    __________________________________________________________________________________________________
    bnorm_3 (BatchNormalization)    (None, None, None, 6 256         conv_3[0][0]                     
    __________________________________________________________________________________________________
    leaky_3 (LeakyReLU)             (None, None, None, 6 0           bnorm_3[0][0]                    
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, None, None, 6 0           leaky_1[0][0]                    
                                                                     leaky_3[0][0]                    
    __________________________________________________________________________________________________
    zero_padding2d_2 (ZeroPadding2D (None, None, None, 6 0           add_1[0][0]                      
    __________________________________________________________________________________________________
    conv_5 (Conv2D)                 (None, None, None, 1 73728       zero_padding2d_2[0][0]           
    __________________________________________________________________________________________________
    bnorm_5 (BatchNormalization)    (None, None, None, 1 512         conv_5[0][0]                     
    __________________________________________________________________________________________________
    leaky_5 (LeakyReLU)             (None, None, None, 1 0           bnorm_5[0][0]                    
    __________________________________________________________________________________________________
    conv_6 (Conv2D)                 (None, None, None, 6 8192        leaky_5[0][0]                    
    __________________________________________________________________________________________________
    bnorm_6 (BatchNormalization)    (None, None, None, 6 256         conv_6[0][0]                     
    __________________________________________________________________________________________________
    leaky_6 (LeakyReLU)             (None, None, None, 6 0           bnorm_6[0][0]                    
    __________________________________________________________________________________________________
    conv_7 (Conv2D)                 (None, None, None, 1 73728       leaky_6[0][0]                    
    __________________________________________________________________________________________________
    bnorm_7 (BatchNormalization)    (None, None, None, 1 512         conv_7[0][0]                     
    __________________________________________________________________________________________________
    leaky_7 (LeakyReLU)             (None, None, None, 1 0           bnorm_7[0][0]                    
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, None, None, 1 0           leaky_5[0][0]                    
                                                                     leaky_7[0][0]                    
    __________________________________________________________________________________________________
    conv_9 (Conv2D)                 (None, None, None, 6 8192        add_2[0][0]                      
    __________________________________________________________________________________________________
    bnorm_9 (BatchNormalization)    (None, None, None, 6 256         conv_9[0][0]                     
    __________________________________________________________________________________________________
    leaky_9 (LeakyReLU)             (None, None, None, 6 0           bnorm_9[0][0]                    
    __________________________________________________________________________________________________
    conv_10 (Conv2D)                (None, None, None, 1 73728       leaky_9[0][0]                    
    __________________________________________________________________________________________________
    bnorm_10 (BatchNormalization)   (None, None, None, 1 512         conv_10[0][0]                    
    __________________________________________________________________________________________________
    leaky_10 (LeakyReLU)            (None, None, None, 1 0           bnorm_10[0][0]                   
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, None, None, 1 0           add_2[0][0]                      
                                                                     leaky_10[0][0]                   
    __________________________________________________________________________________________________
    zero_padding2d_3 (ZeroPadding2D (None, None, None, 1 0           add_3[0][0]                      
    __________________________________________________________________________________________________
    conv_12 (Conv2D)                (None, None, None, 2 294912      zero_padding2d_3[0][0]           
    __________________________________________________________________________________________________
    bnorm_12 (BatchNormalization)   (None, None, None, 2 1024        conv_12[0][0]                    
    __________________________________________________________________________________________________
    leaky_12 (LeakyReLU)            (None, None, None, 2 0           bnorm_12[0][0]                   
    __________________________________________________________________________________________________
    conv_13 (Conv2D)                (None, None, None, 1 32768       leaky_12[0][0]                   
    __________________________________________________________________________________________________
    bnorm_13 (BatchNormalization)   (None, None, None, 1 512         conv_13[0][0]                    
    __________________________________________________________________________________________________
    leaky_13 (LeakyReLU)            (None, None, None, 1 0           bnorm_13[0][0]                   
    __________________________________________________________________________________________________
    conv_14 (Conv2D)                (None, None, None, 2 294912      leaky_13[0][0]                   
    __________________________________________________________________________________________________
    bnorm_14 (BatchNormalization)   (None, None, None, 2 1024        conv_14[0][0]                    
    __________________________________________________________________________________________________
    leaky_14 (LeakyReLU)            (None, None, None, 2 0           bnorm_14[0][0]                   
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, None, None, 2 0           leaky_12[0][0]                   
                                                                     leaky_14[0][0]                   
    __________________________________________________________________________________________________
    conv_16 (Conv2D)                (None, None, None, 1 32768       add_4[0][0]                      
    __________________________________________________________________________________________________
    bnorm_16 (BatchNormalization)   (None, None, None, 1 512         conv_16[0][0]                    
    __________________________________________________________________________________________________
    leaky_16 (LeakyReLU)            (None, None, None, 1 0           bnorm_16[0][0]                   
    __________________________________________________________________________________________________
    conv_17 (Conv2D)                (None, None, None, 2 294912      leaky_16[0][0]                   
    __________________________________________________________________________________________________
    bnorm_17 (BatchNormalization)   (None, None, None, 2 1024        conv_17[0][0]                    
    __________________________________________________________________________________________________
    leaky_17 (LeakyReLU)            (None, None, None, 2 0           bnorm_17[0][0]                   
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, None, None, 2 0           add_4[0][0]                      
                                                                     leaky_17[0][0]                   
    __________________________________________________________________________________________________
    conv_19 (Conv2D)                (None, None, None, 1 32768       add_5[0][0]                      
    __________________________________________________________________________________________________
    bnorm_19 (BatchNormalization)   (None, None, None, 1 512         conv_19[0][0]                    
    __________________________________________________________________________________________________
    leaky_19 (LeakyReLU)            (None, None, None, 1 0           bnorm_19[0][0]                   
    __________________________________________________________________________________________________
    conv_20 (Conv2D)                (None, None, None, 2 294912      leaky_19[0][0]                   
    __________________________________________________________________________________________________
    bnorm_20 (BatchNormalization)   (None, None, None, 2 1024        conv_20[0][0]                    
    __________________________________________________________________________________________________
    leaky_20 (LeakyReLU)            (None, None, None, 2 0           bnorm_20[0][0]                   
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, None, None, 2 0           add_5[0][0]                      
                                                                     leaky_20[0][0]                   
    __________________________________________________________________________________________________
    conv_22 (Conv2D)                (None, None, None, 1 32768       add_6[0][0]                      
    __________________________________________________________________________________________________
    bnorm_22 (BatchNormalization)   (None, None, None, 1 512         conv_22[0][0]                    
    __________________________________________________________________________________________________
    leaky_22 (LeakyReLU)            (None, None, None, 1 0           bnorm_22[0][0]                   
    __________________________________________________________________________________________________
    conv_23 (Conv2D)                (None, None, None, 2 294912      leaky_22[0][0]                   
    __________________________________________________________________________________________________
    bnorm_23 (BatchNormalization)   (None, None, None, 2 1024        conv_23[0][0]                    
    __________________________________________________________________________________________________
    leaky_23 (LeakyReLU)            (None, None, None, 2 0           bnorm_23[0][0]                   
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, None, None, 2 0           add_6[0][0]                      
                                                                     leaky_23[0][0]                   
    __________________________________________________________________________________________________
    conv_25 (Conv2D)                (None, None, None, 1 32768       add_7[0][0]                      
    __________________________________________________________________________________________________
    bnorm_25 (BatchNormalization)   (None, None, None, 1 512         conv_25[0][0]                    
    __________________________________________________________________________________________________
    leaky_25 (LeakyReLU)            (None, None, None, 1 0           bnorm_25[0][0]                   
    __________________________________________________________________________________________________
    conv_26 (Conv2D)                (None, None, None, 2 294912      leaky_25[0][0]                   
    __________________________________________________________________________________________________
    bnorm_26 (BatchNormalization)   (None, None, None, 2 1024        conv_26[0][0]                    
    __________________________________________________________________________________________________
    leaky_26 (LeakyReLU)            (None, None, None, 2 0           bnorm_26[0][0]                   
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, None, None, 2 0           add_7[0][0]                      
                                                                     leaky_26[0][0]                   
    __________________________________________________________________________________________________
    conv_28 (Conv2D)                (None, None, None, 1 32768       add_8[0][0]                      
    __________________________________________________________________________________________________
    bnorm_28 (BatchNormalization)   (None, None, None, 1 512         conv_28[0][0]                    
    __________________________________________________________________________________________________
    leaky_28 (LeakyReLU)            (None, None, None, 1 0           bnorm_28[0][0]                   
    __________________________________________________________________________________________________
    conv_29 (Conv2D)                (None, None, None, 2 294912      leaky_28[0][0]                   
    __________________________________________________________________________________________________
    bnorm_29 (BatchNormalization)   (None, None, None, 2 1024        conv_29[0][0]                    
    __________________________________________________________________________________________________
    leaky_29 (LeakyReLU)            (None, None, None, 2 0           bnorm_29[0][0]                   
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, None, None, 2 0           add_8[0][0]                      
                                                                     leaky_29[0][0]                   
    __________________________________________________________________________________________________
    conv_31 (Conv2D)                (None, None, None, 1 32768       add_9[0][0]                      
    __________________________________________________________________________________________________
    bnorm_31 (BatchNormalization)   (None, None, None, 1 512         conv_31[0][0]                    
    __________________________________________________________________________________________________
    leaky_31 (LeakyReLU)            (None, None, None, 1 0           bnorm_31[0][0]                   
    __________________________________________________________________________________________________
    conv_32 (Conv2D)                (None, None, None, 2 294912      leaky_31[0][0]                   
    __________________________________________________________________________________________________
    bnorm_32 (BatchNormalization)   (None, None, None, 2 1024        conv_32[0][0]                    
    __________________________________________________________________________________________________
    leaky_32 (LeakyReLU)            (None, None, None, 2 0           bnorm_32[0][0]                   
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, None, None, 2 0           add_9[0][0]                      
                                                                     leaky_32[0][0]                   
    __________________________________________________________________________________________________
    conv_34 (Conv2D)                (None, None, None, 1 32768       add_10[0][0]                     
    __________________________________________________________________________________________________
    bnorm_34 (BatchNormalization)   (None, None, None, 1 512         conv_34[0][0]                    
    __________________________________________________________________________________________________
    leaky_34 (LeakyReLU)            (None, None, None, 1 0           bnorm_34[0][0]                   
    __________________________________________________________________________________________________
    conv_35 (Conv2D)                (None, None, None, 2 294912      leaky_34[0][0]                   
    __________________________________________________________________________________________________
    bnorm_35 (BatchNormalization)   (None, None, None, 2 1024        conv_35[0][0]                    
    __________________________________________________________________________________________________
    leaky_35 (LeakyReLU)            (None, None, None, 2 0           bnorm_35[0][0]                   
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, None, None, 2 0           add_10[0][0]                     
                                                                     leaky_35[0][0]                   
    __________________________________________________________________________________________________
    zero_padding2d_4 (ZeroPadding2D (None, None, None, 2 0           add_11[0][0]                     
    __________________________________________________________________________________________________
    conv_37 (Conv2D)                (None, None, None, 5 1179648     zero_padding2d_4[0][0]           
    __________________________________________________________________________________________________
    bnorm_37 (BatchNormalization)   (None, None, None, 5 2048        conv_37[0][0]                    
    __________________________________________________________________________________________________
    leaky_37 (LeakyReLU)            (None, None, None, 5 0           bnorm_37[0][0]                   
    __________________________________________________________________________________________________
    conv_38 (Conv2D)                (None, None, None, 2 131072      leaky_37[0][0]                   
    __________________________________________________________________________________________________
    bnorm_38 (BatchNormalization)   (None, None, None, 2 1024        conv_38[0][0]                    
    __________________________________________________________________________________________________
    leaky_38 (LeakyReLU)            (None, None, None, 2 0           bnorm_38[0][0]                   
    __________________________________________________________________________________________________
    conv_39 (Conv2D)                (None, None, None, 5 1179648     leaky_38[0][0]                   
    __________________________________________________________________________________________________
    bnorm_39 (BatchNormalization)   (None, None, None, 5 2048        conv_39[0][0]                    
    __________________________________________________________________________________________________
    leaky_39 (LeakyReLU)            (None, None, None, 5 0           bnorm_39[0][0]                   
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, None, None, 5 0           leaky_37[0][0]                   
                                                                     leaky_39[0][0]                   
    __________________________________________________________________________________________________
    conv_41 (Conv2D)                (None, None, None, 2 131072      add_12[0][0]                     
    __________________________________________________________________________________________________
    bnorm_41 (BatchNormalization)   (None, None, None, 2 1024        conv_41[0][0]                    
    __________________________________________________________________________________________________
    leaky_41 (LeakyReLU)            (None, None, None, 2 0           bnorm_41[0][0]                   
    __________________________________________________________________________________________________
    conv_42 (Conv2D)                (None, None, None, 5 1179648     leaky_41[0][0]                   
    __________________________________________________________________________________________________
    bnorm_42 (BatchNormalization)   (None, None, None, 5 2048        conv_42[0][0]                    
    __________________________________________________________________________________________________
    leaky_42 (LeakyReLU)            (None, None, None, 5 0           bnorm_42[0][0]                   
    __________________________________________________________________________________________________
    add_13 (Add)                    (None, None, None, 5 0           add_12[0][0]                     
                                                                     leaky_42[0][0]                   
    __________________________________________________________________________________________________
    conv_44 (Conv2D)                (None, None, None, 2 131072      add_13[0][0]                     
    __________________________________________________________________________________________________
    bnorm_44 (BatchNormalization)   (None, None, None, 2 1024        conv_44[0][0]                    
    __________________________________________________________________________________________________
    leaky_44 (LeakyReLU)            (None, None, None, 2 0           bnorm_44[0][0]                   
    __________________________________________________________________________________________________
    conv_45 (Conv2D)                (None, None, None, 5 1179648     leaky_44[0][0]                   
    __________________________________________________________________________________________________
    bnorm_45 (BatchNormalization)   (None, None, None, 5 2048        conv_45[0][0]                    
    __________________________________________________________________________________________________
    leaky_45 (LeakyReLU)            (None, None, None, 5 0           bnorm_45[0][0]                   
    __________________________________________________________________________________________________
    add_14 (Add)                    (None, None, None, 5 0           add_13[0][0]                     
                                                                     leaky_45[0][0]                   
    __________________________________________________________________________________________________
    conv_47 (Conv2D)                (None, None, None, 2 131072      add_14[0][0]                     
    __________________________________________________________________________________________________
    bnorm_47 (BatchNormalization)   (None, None, None, 2 1024        conv_47[0][0]                    
    __________________________________________________________________________________________________
    leaky_47 (LeakyReLU)            (None, None, None, 2 0           bnorm_47[0][0]                   
    __________________________________________________________________________________________________
    conv_48 (Conv2D)                (None, None, None, 5 1179648     leaky_47[0][0]                   
    __________________________________________________________________________________________________
    bnorm_48 (BatchNormalization)   (None, None, None, 5 2048        conv_48[0][0]                    
    __________________________________________________________________________________________________
    leaky_48 (LeakyReLU)            (None, None, None, 5 0           bnorm_48[0][0]                   
    __________________________________________________________________________________________________
    add_15 (Add)                    (None, None, None, 5 0           add_14[0][0]                     
                                                                     leaky_48[0][0]                   
    __________________________________________________________________________________________________
    conv_50 (Conv2D)                (None, None, None, 2 131072      add_15[0][0]                     
    __________________________________________________________________________________________________
    bnorm_50 (BatchNormalization)   (None, None, None, 2 1024        conv_50[0][0]                    
    __________________________________________________________________________________________________
    leaky_50 (LeakyReLU)            (None, None, None, 2 0           bnorm_50[0][0]                   
    __________________________________________________________________________________________________
    conv_51 (Conv2D)                (None, None, None, 5 1179648     leaky_50[0][0]                   
    __________________________________________________________________________________________________
    bnorm_51 (BatchNormalization)   (None, None, None, 5 2048        conv_51[0][0]                    
    __________________________________________________________________________________________________
    leaky_51 (LeakyReLU)            (None, None, None, 5 0           bnorm_51[0][0]                   
    __________________________________________________________________________________________________
    add_16 (Add)                    (None, None, None, 5 0           add_15[0][0]                     
                                                                     leaky_51[0][0]                   
    __________________________________________________________________________________________________
    conv_53 (Conv2D)                (None, None, None, 2 131072      add_16[0][0]                     
    __________________________________________________________________________________________________
    bnorm_53 (BatchNormalization)   (None, None, None, 2 1024        conv_53[0][0]                    
    __________________________________________________________________________________________________
    leaky_53 (LeakyReLU)            (None, None, None, 2 0           bnorm_53[0][0]                   
    __________________________________________________________________________________________________
    conv_54 (Conv2D)                (None, None, None, 5 1179648     leaky_53[0][0]                   
    __________________________________________________________________________________________________
    bnorm_54 (BatchNormalization)   (None, None, None, 5 2048        conv_54[0][0]                    
    __________________________________________________________________________________________________
    leaky_54 (LeakyReLU)            (None, None, None, 5 0           bnorm_54[0][0]                   
    __________________________________________________________________________________________________
    add_17 (Add)                    (None, None, None, 5 0           add_16[0][0]                     
                                                                     leaky_54[0][0]                   
    __________________________________________________________________________________________________
    conv_56 (Conv2D)                (None, None, None, 2 131072      add_17[0][0]                     
    __________________________________________________________________________________________________
    bnorm_56 (BatchNormalization)   (None, None, None, 2 1024        conv_56[0][0]                    
    __________________________________________________________________________________________________
    leaky_56 (LeakyReLU)            (None, None, None, 2 0           bnorm_56[0][0]                   
    __________________________________________________________________________________________________
    conv_57 (Conv2D)                (None, None, None, 5 1179648     leaky_56[0][0]                   
    __________________________________________________________________________________________________
    bnorm_57 (BatchNormalization)   (None, None, None, 5 2048        conv_57[0][0]                    
    __________________________________________________________________________________________________
    leaky_57 (LeakyReLU)            (None, None, None, 5 0           bnorm_57[0][0]                   
    __________________________________________________________________________________________________
    add_18 (Add)                    (None, None, None, 5 0           add_17[0][0]                     
                                                                     leaky_57[0][0]                   
    __________________________________________________________________________________________________
    conv_59 (Conv2D)                (None, None, None, 2 131072      add_18[0][0]                     
    __________________________________________________________________________________________________
    bnorm_59 (BatchNormalization)   (None, None, None, 2 1024        conv_59[0][0]                    
    __________________________________________________________________________________________________
    leaky_59 (LeakyReLU)            (None, None, None, 2 0           bnorm_59[0][0]                   
    __________________________________________________________________________________________________
    conv_60 (Conv2D)                (None, None, None, 5 1179648     leaky_59[0][0]                   
    __________________________________________________________________________________________________
    bnorm_60 (BatchNormalization)   (None, None, None, 5 2048        conv_60[0][0]                    
    __________________________________________________________________________________________________
    leaky_60 (LeakyReLU)            (None, None, None, 5 0           bnorm_60[0][0]                   
    __________________________________________________________________________________________________
    add_19 (Add)                    (None, None, None, 5 0           add_18[0][0]                     
                                                                     leaky_60[0][0]                   
    __________________________________________________________________________________________________
    zero_padding2d_5 (ZeroPadding2D (None, None, None, 5 0           add_19[0][0]                     
    __________________________________________________________________________________________________
    conv_62 (Conv2D)                (None, None, None, 1 4718592     zero_padding2d_5[0][0]           
    __________________________________________________________________________________________________
    bnorm_62 (BatchNormalization)   (None, None, None, 1 4096        conv_62[0][0]                    
    __________________________________________________________________________________________________
    leaky_62 (LeakyReLU)            (None, None, None, 1 0           bnorm_62[0][0]                   
    __________________________________________________________________________________________________
    conv_63 (Conv2D)                (None, None, None, 5 524288      leaky_62[0][0]                   
    __________________________________________________________________________________________________
    bnorm_63 (BatchNormalization)   (None, None, None, 5 2048        conv_63[0][0]                    
    __________________________________________________________________________________________________
    leaky_63 (LeakyReLU)            (None, None, None, 5 0           bnorm_63[0][0]                   
    __________________________________________________________________________________________________
    conv_64 (Conv2D)                (None, None, None, 1 4718592     leaky_63[0][0]                   
    __________________________________________________________________________________________________
    bnorm_64 (BatchNormalization)   (None, None, None, 1 4096        conv_64[0][0]                    
    __________________________________________________________________________________________________
    leaky_64 (LeakyReLU)            (None, None, None, 1 0           bnorm_64[0][0]                   
    __________________________________________________________________________________________________
    add_20 (Add)                    (None, None, None, 1 0           leaky_62[0][0]                   
                                                                     leaky_64[0][0]                   
    __________________________________________________________________________________________________
    conv_66 (Conv2D)                (None, None, None, 5 524288      add_20[0][0]                     
    __________________________________________________________________________________________________
    bnorm_66 (BatchNormalization)   (None, None, None, 5 2048        conv_66[0][0]                    
    __________________________________________________________________________________________________
    leaky_66 (LeakyReLU)            (None, None, None, 5 0           bnorm_66[0][0]                   
    __________________________________________________________________________________________________
    conv_67 (Conv2D)                (None, None, None, 1 4718592     leaky_66[0][0]                   
    __________________________________________________________________________________________________
    bnorm_67 (BatchNormalization)   (None, None, None, 1 4096        conv_67[0][0]                    
    __________________________________________________________________________________________________
    leaky_67 (LeakyReLU)            (None, None, None, 1 0           bnorm_67[0][0]                   
    __________________________________________________________________________________________________
    add_21 (Add)                    (None, None, None, 1 0           add_20[0][0]                     
                                                                     leaky_67[0][0]                   
    __________________________________________________________________________________________________
    conv_69 (Conv2D)                (None, None, None, 5 524288      add_21[0][0]                     
    __________________________________________________________________________________________________
    bnorm_69 (BatchNormalization)   (None, None, None, 5 2048        conv_69[0][0]                    
    __________________________________________________________________________________________________
    leaky_69 (LeakyReLU)            (None, None, None, 5 0           bnorm_69[0][0]                   
    __________________________________________________________________________________________________
    conv_70 (Conv2D)                (None, None, None, 1 4718592     leaky_69[0][0]                   
    __________________________________________________________________________________________________
    bnorm_70 (BatchNormalization)   (None, None, None, 1 4096        conv_70[0][0]                    
    __________________________________________________________________________________________________
    leaky_70 (LeakyReLU)            (None, None, None, 1 0           bnorm_70[0][0]                   
    __________________________________________________________________________________________________
    add_22 (Add)                    (None, None, None, 1 0           add_21[0][0]                     
                                                                     leaky_70[0][0]                   
    __________________________________________________________________________________________________
    conv_72 (Conv2D)                (None, None, None, 5 524288      add_22[0][0]                     
    __________________________________________________________________________________________________
    bnorm_72 (BatchNormalization)   (None, None, None, 5 2048        conv_72[0][0]                    
    __________________________________________________________________________________________________
    leaky_72 (LeakyReLU)            (None, None, None, 5 0           bnorm_72[0][0]                   
    __________________________________________________________________________________________________
    conv_73 (Conv2D)                (None, None, None, 1 4718592     leaky_72[0][0]                   
    __________________________________________________________________________________________________
    bnorm_73 (BatchNormalization)   (None, None, None, 1 4096        conv_73[0][0]                    
    __________________________________________________________________________________________________
    leaky_73 (LeakyReLU)            (None, None, None, 1 0           bnorm_73[0][0]                   
    __________________________________________________________________________________________________
    add_23 (Add)                    (None, None, None, 1 0           add_22[0][0]                     
                                                                     leaky_73[0][0]                   
    __________________________________________________________________________________________________
    conv_75 (Conv2D)                (None, None, None, 5 524288      add_23[0][0]                     
    __________________________________________________________________________________________________
    bnorm_75 (BatchNormalization)   (None, None, None, 5 2048        conv_75[0][0]                    
    __________________________________________________________________________________________________
    leaky_75 (LeakyReLU)            (None, None, None, 5 0           bnorm_75[0][0]                   
    __________________________________________________________________________________________________
    conv_76 (Conv2D)                (None, None, None, 1 4718592     leaky_75[0][0]                   
    __________________________________________________________________________________________________
    bnorm_76 (BatchNormalization)   (None, None, None, 1 4096        conv_76[0][0]                    
    __________________________________________________________________________________________________
    leaky_76 (LeakyReLU)            (None, None, None, 1 0           bnorm_76[0][0]                   
    __________________________________________________________________________________________________
    conv_77 (Conv2D)                (None, None, None, 5 524288      leaky_76[0][0]                   
    __________________________________________________________________________________________________
    bnorm_77 (BatchNormalization)   (None, None, None, 5 2048        conv_77[0][0]                    
    __________________________________________________________________________________________________
    leaky_77 (LeakyReLU)            (None, None, None, 5 0           bnorm_77[0][0]                   
    __________________________________________________________________________________________________
    conv_78 (Conv2D)                (None, None, None, 1 4718592     leaky_77[0][0]                   
    __________________________________________________________________________________________________
    bnorm_78 (BatchNormalization)   (None, None, None, 1 4096        conv_78[0][0]                    
    __________________________________________________________________________________________________
    leaky_78 (LeakyReLU)            (None, None, None, 1 0           bnorm_78[0][0]                   
    __________________________________________________________________________________________________
    conv_79 (Conv2D)                (None, None, None, 5 524288      leaky_78[0][0]                   
    __________________________________________________________________________________________________
    bnorm_79 (BatchNormalization)   (None, None, None, 5 2048        conv_79[0][0]                    
    __________________________________________________________________________________________________
    leaky_79 (LeakyReLU)            (None, None, None, 5 0           bnorm_79[0][0]                   
    __________________________________________________________________________________________________
    conv_84 (Conv2D)                (None, None, None, 2 131072      leaky_79[0][0]                   
    __________________________________________________________________________________________________
    bnorm_84 (BatchNormalization)   (None, None, None, 2 1024        conv_84[0][0]                    
    __________________________________________________________________________________________________
    leaky_84 (LeakyReLU)            (None, None, None, 2 0           bnorm_84[0][0]                   
    __________________________________________________________________________________________________
    up_sampling2d_1 (UpSampling2D)  (None, None, None, 2 0           leaky_84[0][0]                   
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, None, None, 7 0           up_sampling2d_1[0][0]            
                                                                     add_19[0][0]                     
    __________________________________________________________________________________________________
    conv_87 (Conv2D)                (None, None, None, 2 196608      concatenate_1[0][0]              
    __________________________________________________________________________________________________
    bnorm_87 (BatchNormalization)   (None, None, None, 2 1024        conv_87[0][0]                    
    __________________________________________________________________________________________________
    leaky_87 (LeakyReLU)            (None, None, None, 2 0           bnorm_87[0][0]                   
    __________________________________________________________________________________________________
    conv_88 (Conv2D)                (None, None, None, 5 1179648     leaky_87[0][0]                   
    __________________________________________________________________________________________________
    bnorm_88 (BatchNormalization)   (None, None, None, 5 2048        conv_88[0][0]                    
    __________________________________________________________________________________________________
    leaky_88 (LeakyReLU)            (None, None, None, 5 0           bnorm_88[0][0]                   
    __________________________________________________________________________________________________
    conv_89 (Conv2D)                (None, None, None, 2 131072      leaky_88[0][0]                   
    __________________________________________________________________________________________________
    bnorm_89 (BatchNormalization)   (None, None, None, 2 1024        conv_89[0][0]                    
    __________________________________________________________________________________________________
    leaky_89 (LeakyReLU)            (None, None, None, 2 0           bnorm_89[0][0]                   
    __________________________________________________________________________________________________
    conv_90 (Conv2D)                (None, None, None, 5 1179648     leaky_89[0][0]                   
    __________________________________________________________________________________________________
    bnorm_90 (BatchNormalization)   (None, None, None, 5 2048        conv_90[0][0]                    
    __________________________________________________________________________________________________
    leaky_90 (LeakyReLU)            (None, None, None, 5 0           bnorm_90[0][0]                   
    __________________________________________________________________________________________________
    conv_91 (Conv2D)                (None, None, None, 2 131072      leaky_90[0][0]                   
    __________________________________________________________________________________________________
    bnorm_91 (BatchNormalization)   (None, None, None, 2 1024        conv_91[0][0]                    
    __________________________________________________________________________________________________
    leaky_91 (LeakyReLU)            (None, None, None, 2 0           bnorm_91[0][0]                   
    __________________________________________________________________________________________________
    conv_96 (Conv2D)                (None, None, None, 1 32768       leaky_91[0][0]                   
    __________________________________________________________________________________________________
    bnorm_96 (BatchNormalization)   (None, None, None, 1 512         conv_96[0][0]                    
    __________________________________________________________________________________________________
    leaky_96 (LeakyReLU)            (None, None, None, 1 0           bnorm_96[0][0]                   
    __________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)  (None, None, None, 1 0           leaky_96[0][0]                   
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, None, None, 3 0           up_sampling2d_2[0][0]            
                                                                     add_11[0][0]                     
    __________________________________________________________________________________________________
    conv_99 (Conv2D)                (None, None, None, 1 49152       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    bnorm_99 (BatchNormalization)   (None, None, None, 1 512         conv_99[0][0]                    
    __________________________________________________________________________________________________
    leaky_99 (LeakyReLU)            (None, None, None, 1 0           bnorm_99[0][0]                   
    __________________________________________________________________________________________________
    conv_100 (Conv2D)               (None, None, None, 2 294912      leaky_99[0][0]                   
    __________________________________________________________________________________________________
    bnorm_100 (BatchNormalization)  (None, None, None, 2 1024        conv_100[0][0]                   
    __________________________________________________________________________________________________
    leaky_100 (LeakyReLU)           (None, None, None, 2 0           bnorm_100[0][0]                  
    __________________________________________________________________________________________________
    conv_101 (Conv2D)               (None, None, None, 1 32768       leaky_100[0][0]                  
    __________________________________________________________________________________________________
    bnorm_101 (BatchNormalization)  (None, None, None, 1 512         conv_101[0][0]                   
    __________________________________________________________________________________________________
    leaky_101 (LeakyReLU)           (None, None, None, 1 0           bnorm_101[0][0]                  
    __________________________________________________________________________________________________
    conv_102 (Conv2D)               (None, None, None, 2 294912      leaky_101[0][0]                  
    __________________________________________________________________________________________________
    bnorm_102 (BatchNormalization)  (None, None, None, 2 1024        conv_102[0][0]                   
    __________________________________________________________________________________________________
    leaky_102 (LeakyReLU)           (None, None, None, 2 0           bnorm_102[0][0]                  
    __________________________________________________________________________________________________
    conv_103 (Conv2D)               (None, None, None, 1 32768       leaky_102[0][0]                  
    __________________________________________________________________________________________________
    bnorm_103 (BatchNormalization)  (None, None, None, 1 512         conv_103[0][0]                   
    __________________________________________________________________________________________________
    leaky_103 (LeakyReLU)           (None, None, None, 1 0           bnorm_103[0][0]                  
    __________________________________________________________________________________________________
    conv_80 (Conv2D)                (None, None, None, 1 4718592     leaky_79[0][0]                   
    __________________________________________________________________________________________________
    conv_92 (Conv2D)                (None, None, None, 5 1179648     leaky_91[0][0]                   
    __________________________________________________________________________________________________
    conv_104 (Conv2D)               (None, None, None, 2 294912      leaky_103[0][0]                  
    __________________________________________________________________________________________________
    bnorm_80 (BatchNormalization)   (None, None, None, 1 4096        conv_80[0][0]                    
    __________________________________________________________________________________________________
    bnorm_92 (BatchNormalization)   (None, None, None, 5 2048        conv_92[0][0]                    
    __________________________________________________________________________________________________
    bnorm_104 (BatchNormalization)  (None, None, None, 2 1024        conv_104[0][0]                   
    __________________________________________________________________________________________________
    leaky_80 (LeakyReLU)            (None, None, None, 1 0           bnorm_80[0][0]                   
    __________________________________________________________________________________________________
    leaky_92 (LeakyReLU)            (None, None, None, 5 0           bnorm_92[0][0]                   
    __________________________________________________________________________________________________
    leaky_104 (LeakyReLU)           (None, None, None, 2 0           bnorm_104[0][0]                  
    __________________________________________________________________________________________________
    conv_81 (Conv2D)                (None, None, None, 2 261375      leaky_80[0][0]                   
    __________________________________________________________________________________________________
    conv_93 (Conv2D)                (None, None, None, 2 130815      leaky_92[0][0]                   
    __________________________________________________________________________________________________
    conv_105 (Conv2D)               (None, None, None, 2 65535       leaky_104[0][0]                  
    ==================================================================================================
    Total params: 62,001,757
    Trainable params: 61,949,149
    Non-trainable params: 52,608
    __________________________________________________________________________________________________


### 4. Prediction

이제 우리의 모델이 잘 동작하는지 확인하기 위해서 training dataset에 있는 10개의 이미지로 테스트 해보자.

(pre-trained된 가중치를 사용했기 때문에 훈련 과정은 필요 없다!)




```python
# pre-trained된 데이터셋에 사용된 parameter
anchors = [[116,90, 156,198, 3737,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# 입력 이미지의 크기 설정
WIDTH, HEIGHT = 416, 416

# class probability threshold 설정
class_threshold = 0.3
```


```python
import os
from matplotlib import pyplot as plt
images = os.listdir('../input/3d-object-detection-for-autonomous-vehicles/train_images')[:10]
```


```python
DATA_PATH = '../input/3d-object-detection-for-autonomous-vehicles/'

# 이미지들을 iterate하면서 detect 수행
for file in images:
    photo_filename = DATA_PATH+ 'train_images/' + file
    
    # load picture 
    image, image_w, image_h = load_image_pixels(photo_filename, (WIDTH, HEIGHT))
    
    # 모델로 이미지 추론값 도출
    yhat = model.predict(image)
    
    # Creat bbox
    boxes = list()
    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, HEIGHT, WIDTH)
    
    # bbox의 크기를 입력 이미지의 크기에 맞춘다.
    correct_yolo_boxes(boxes, image_h, image_w, HEIGHT, WIDTH)
    
    # nms 수행
    do_nms(boxes, 0.5)
    
    # pre-trained에 사용된 label 정의
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck","boat"]
    
    # detect된 객체의 정보
    v_boxes, v_labels, v_scores = get_boxes(boxes,labels, class_threshold)
    
    # 찾은 객체와 score 표시
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
        
    # bbox 시각화
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
```

    truck 94.47240233421326
    car 71.33424878120422
    car 57.38890767097473
    car 95.99141478538513
    car 96.51923775672913
    bus 44.159525632858276
    truck 70.85143327713013
    car 37.07187473773956
    car 35.22671163082123
    truck 32.79102146625519
    car 52.50139832496643
    car 71.62013053894043
    car 41.44318103790283
    car 98.87382388114929





    
![2](https://user-images.githubusercontent.com/77332628/226091699-d269c64a-ebef-4e07-94ad-5e9eccccd1a0.png)
    



    car 90.90200066566467
    car 41.74528121948242
    car 49.315935373306274
    car 86.64723038673401





    
![3](https://user-images.githubusercontent.com/77332628/226091700-682a2c16-fc5b-4915-9304-0d1725f600cb.png)
    



    car 44.00638043880463
    car 99.49303269386292
    car 69.17536854743958
    car 78.7386417388916
    car 64.8408830165863
    car 85.27180552482605
    car 97.52064943313599
    car 83.2088828086853
    car 33.15470218658447





    
![4](https://user-images.githubusercontent.com/77332628/226091701-7dd29ace-7392-4f6c-b8f4-6599c47dd8d8.png)    



    car 43.18581819534302
    car 92.42860078811646
    car 92.89239048957825





    
![5](https://user-images.githubusercontent.com/77332628/226091702-0f411136-27eb-4302-b3d4-22b226b458a3.png)    



    car 65.70577025413513
    truck 54.95043396949768
    car 40.10573327541351
    truck 90.99632501602173
    car 36.974069476127625
    car 54.107075929641724
    car 98.18232655525208
    car 78.2330334186554
    car 62.80314922332764
    car 62.72594928741455





    

![6](https://user-images.githubusercontent.com/77332628/226091705-54b3c4b8-727c-49f4-8ce7-5dcf74c48b6d.png)    



    bus 92.19626784324646
    car 88.94755244255066
    car 87.45279908180237
    truck 35.09664535522461
    truck 39.33030664920807
    car 31.529730558395386
    car 80.00765442848206
    car 38.52341175079346
    car 31.965255737304688
    car 41.98690056800842
    car 99.15739893913269





    
![7](https://user-images.githubusercontent.com/77332628/226091706-757039b1-a07a-45ec-83e1-77f9e3115309.png)    



    car 99.4914710521698
    car 39.73972201347351
    car 32.706499099731445
    car 96.72788977622986
    car 59.8081111907959
    car 32.48153328895569
    car 73.06863069534302
    car 84.5885694026947
    car 84.10609364509583
    car 84.15658473968506
    car 51.734769344329834





    
![8](https://user-images.githubusercontent.com/77332628/226091708-d49765d8-d434-4b3d-8dd1-e1220f7de91a.png)    



    car 99.98246431350708
    car 95.04622220993042
    truck 56.46554231643677
    car 84.31439995765686
    car 39.08815681934357
    car 31.059062480926514
    car 44.72985863685608
    car 91.98101162910461
    car 62.90650963783264
    car 69.01302933692932
    car 94.28300261497498
    car 39.20080363750458
    car 89.67056274414062
    car 50.714272260665894





    
![9](https://user-images.githubusercontent.com/77332628/226091710-b47d968b-14e9-49a6-806a-750966036181.png)    



    car 86.70346140861511
    car 91.92036390304565
    car 68.00507307052612
    car 83.37287306785583
    car 49.55475330352783
    car 32.760241627693176
    car 88.03302049636841
    car 87.4018907546997
    car 92.23202466964722
    car 70.95179557800293
    car 66.62480235099792





    
![10](https://user-images.githubusercontent.com/77332628/226091712-e7bc9762-33a7-44c5-8188-8d756a7613de.png)    



    car 47.68282175064087
    car 99.59219098091125
    car 99.41357374191284
    car 35.1456880569458
    car 69.22145485877991
    car 76.32824182510376
    car 40.37817418575287
    car 97.95092344284058




    

![11](https://user-images.githubusercontent.com/77332628/226091714-64769fde-c053-454c-8ae9-4726d655112a.png)    


직접 모델을 사용해서 실제로 객체를 탐지해보니 흥미로웠다. 물론 nms나 Weight Reader 같은 함수와 클래스의 작동 방법을 완전히 이해하지는 못했지만, 얼추 대략적인 모델의 사용방법을 알 수 있어서 좋았다.
