---
title: "미니프로젝트 4차"
tags:
    - MiniProject
    - CNN
    - Object Detection
    - YOLO
    - Roboflow
date: "2024-11-01"
thumbnail: "/assets/img/aivleproject/proj4/thumb4.jpg"
---

# 이미지 데이터 모델링 얼굴 인식
---
![facerecog](/assets/img/aivleproject/proj4/facerecog.jpg)

* 이미지 데이터를 학습하여 얼굴을 인식하고 분류하는 모델을 설계합니다.
* Pre-Trained Model을 사용하여 높은 성능의 Object Detection이 가능하도록 설계합니다.

# 0. 얼굴 데이터 수집 및 증강
---

## 데이터 수집
```python
import os
import cv2

## 이미지를 저장할 폴더 경로
folder_path = ''
## 이미지 저장 폴더가 없다면 폴더 생성
if not os.path.exists(folder_path) :
    os.makedirs(folder_path)
    print('my_face 폴더를 생성합니다.')

## 웹캠으로 얼굴 사진을 찍어 저장하는 함수
def capture_owner_images(num_images=) : ## num_images에 숫자를 입력한만큼 이미지 저장
    ## 0은 기본 웹캠
    cap = cv2.VideoCapture(0)
    ## haarcascade 알고리즘으로 얼굴 탐지
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ## 위에서 지정한 수만큼 촬영 및 저장하는 반복문
    count = 0
    while count < num_images :
        _, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        ## haarcascade 알고리즘은 흑백 이미지의 명암 차이로 탐지를 하는 것이기에 흑백으로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ## 변환된 프레임에서 얼굴 탐지 시도
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        ## 탐지된 얼굴에서 좌표를 가져오는 반복문
        for (x, y, w, h) in faces :
            ## 프레임에서 얼굴 영역만 가져온다
            face = frame[y:y+h, x:x+w]
            ## 탐지한 얼굴 영역 사이즈 변환 : 괄호를 채우면 됩니다.
            resized_face = cv2.resize(face, ( , ) )
            ## 파일 이름 설정 및 저장
            face_file = f"경로설정/파일명_{count}.jpg"
            cv2.imwrite(face_file, resized_face)
            count += 1
            ## 변환된 얼굴 이미지 출력하여 확인
            cv2.imshow('Captured Face', resized_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
capture_owner_images()
print('저장 완료')
```

## 데이터 증강
```python
import cv2
import keras

import os
import numpy as np

keras.utils.clear_session()

#####################################################
aug_layers = keras.models.Sequential()
aug_layers.add( keras.layers.RandomFlip() )
aug_layers.add( keras.layers.RandomRotation(0.15) )
aug_layers.add( keras.layers.RandomZoom(0.15) )
#####################################################

## augmentation을 적용할 원본 이미지가 있는 경로
img_folder = ''
img_files = os.listdir(img_folder)

## augmentation을 적용한 이미지를 저장할 폴더 경로 및 생성
output_folder = './augmented_images'
if not os.path.exists(output_folder) :
    os.mkdir(output_folder)
    
## 원본 이미지 경로로 만들어진 원본 이미지 리스트를 이용한 반복문
for img_f in img_files :
    ## 원본 이미지 경로
    img_ori = os.path.join(img_folder, img_f)
    ## 이미지 선택 및 array화
    img_ori = keras.utils.load_img(img_ori)
    img_ori = keras.utils.img_to_array(img_ori)
    
    ## 이미지 개별마다 몇 개의 augment를 진행할 것인지에 대한 반복문
    for i in range() :  ## 생성할 이미지 수 입력 필요
        ## 새롭게 저장될 이미지 파일명
        aug_path = os.path.join(output_folder, f'aug_{i}_{img_f}')
        
        ##################################
        ## 위에서 설정한 augment 옵션들을 적용
        img_aug = aug_layers( img_ori )
        ##################################
        
        ## augment 작업을 거치면 자료형이 tensor로 바뀌어서 다시 array로 전환
        ## numpy array가 아니면 저장할 때 에러 발생
        img_aug = keras.utils.img_to_array( img_aug )
        ## cv2의 색상 정보 순서는 BGR순이기에 RGB로 전환
        img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        ## cv2를 이용하여 파일 저장
        cv2.imwrite(aug_path, img_aug)
        print(f'이미지 {img_f}에 대한 {i+1}번째 증강 작업 완료')
    print( f'이미지 {img_f} 증강 완료')
    print('========================')
print(f'이미지 전체 증강 작업 완료')
```

# 1. FaceNet 모델 학습
---
## 데이터셋 준비
```python
tr_idfd, val_idfd = image_dataset_from_directory(tr_data,   ## Training 폴더 경로
                                class_names=['other','my'], ## 클래스 순서 지정
                                batch_size=32,              ## 이미지 덩어리 단위
                                image_size=(160,160),       ## 이미지 리사이즈
                                shuffle=True,               ## 섞어야 올바르게 분할됨
                                seed=2024,                  ## 재현성
                                validation_split=0.3,       ## 데이터 스플릿 비율
                                subset='both',              ## 데이터셋 나눔 방식
                                )
```

## FaceNet Transfer Learning

```python
@register_keras_serializable()
def scaling(x, scale):
    return x * scale

@register_keras_serializable()
def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, momentum=0.995, epsilon=0.001,
                               scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x

@register_keras_serializable()
def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

@register_keras_serializable()
def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                #    K.int_shape(x)[channel_axis],
                   x.shape[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    up = Lambda(scaling,
                # output_shape=K.int_shape(up)[1:],
                output_shape=up.shape[1:],
                arguments={'scale': scale})(up)
    x = add([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x

@register_keras_serializable()
def InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8,
                      weights_path=None):
    inputs = Input(shape=input_shape)
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')

    # 5x Block35 (Inception-ResNet-A block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='Block35',
                                    block_idx=block_idx)

    # Mixed 6a (Reduction-A block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 10x Block17 (Inception-ResNet-B block):
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 5x Block8 (Inception-ResNet-C block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,
                                scale=1.,
                                activation=None,
                                block_type='Block8',
                                block_idx=6)

    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
    # Bottleneck
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name=bn_name)(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')
    if weights_path is not None:
        model.load_weights(weights_path)

    return model
```

## 결과 확인

### Model Train
![result1](/assets/img/aivleproject/proj4/result1.PNG)
### Classification Report
![result2](/assets/img/aivleproject/proj4/result2.PNG)


# 2. YOLO 모델 학습
---
## YOLO Transfer Learning
```python
# Load a model
model = YOLO("yolo11n-cls.pt")

# Train the model
results = model.train(data="/content/dataset", epochs=5, imgsz=640)
```
![result3](/assets/img/aivleproject/proj4/result3.PNG)


# 3. 팀원 데이터 학습 및 FineTuning
---

## Data Labeling
```python
import os

# 라벨을 확인할 이름 리스트
names = ['jaeyub', 'minkyu', 'taegyeong','miso','jonghwan']

# 각 이름별로 라벨 확인
for name in names:
    # 라벨 파일이 저장된 각 경로 설정
    label_paths = [
        f'/content/{name}_face/train/labels',
        f'/content/{name}_face/test/labels',
        f'/content/{name}_face/valid/labels'
    ]

    # 라벨을 저장할 세트 초기화
    unique_labels = set()

    # 각 경로에 있는 .txt 파일을 순회하며 라벨 ID 추출
    for path in label_paths:
        if os.path.exists(path):
            txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]

            for file_name in txt_files:
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r') as file:
                    for line in file:
                        # 라벨 ID 추출
                        label_id = line.split()[0]
                        unique_labels.add(label_id)

    # 고유한 라벨 수와 라벨 목록 출력
    print(f"{name}_face Label : {unique_labels}")
```

## Make Dataset subset
```python
import os
import glob
from collections import defaultdict

# 데이터셋 경로
dataset_path = '/content/dataset'
subsets = ['train', 'test', 'valid']
classes = ['other_face', 'jaeyub_face', 'minkyu_face', 'taegyeong_face', 'miso_face', 'jonghwan_face']

# 클래스별 파일 개수를 저장할 딕셔너리 초기화
class_counts = {subset: defaultdict(int) for subset in subsets}

# 각 subset의 클래스별 파일 개수 확인
for subset in subsets:
    label_path = os.path.join(dataset_path, subset, 'labels')
    label_files = glob.glob(os.path.join(label_path, '*.txt'))

    for label_file in label_files:
        with open(label_file, 'r') as file:
            for line in file:
                class_index = int(line.split()[0])  # 첫 번째 값이 클래스 인덱스
                class_name = classes[class_index]
                class_counts[subset][class_name] += 1

# 결과 출력
for subset in subsets:
    print(f"\n{subset.capitalize()} set class counts:")
    for class_name, count in class_counts[subset].items():
        print(f"  {class_name}: {count} instances")
```

## FineTuning
[발표자료PPT 확인](https://docs.google.com/presentation/d/1XEtH56wZRzJR0SsM4oySK8OZSsKenQJA/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

# 느낀점
---
* ObjectDetection 모델을 튜닝하는 것에 굉장한 어려움을 겪었다.
* 데이터수집 및 annotation을 자동화하는 방법을 좀 더 알아봤으면 좋았을 것 같다.
* 이번에는 데이터 수집 과정이 굉장히 중요했음을 알 수 있었다.