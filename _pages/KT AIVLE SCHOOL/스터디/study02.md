---
title: "2차 스터디 회의"
tags:
    - STUDY
date: "2024-11-10"
thumbnail: "/assets/img/thumbnail/study.jpg"
---

# 회의 주제
---
* DACON 진행상황 공유
* 언어지능 딥러닝 복습

## DACON 진행상황 공유
---
![DACON1](/assets/img/study/dacon1.PNG)
- 진행기간 : 2024-11-03 ~ 2024-11-17
- 시각지능 개별실습을 위한 개별과제 부여


### 전재엽
[코드 확인]()
- 전처리 과정에서 csv파일을 이용한 Labeled_Dataset을 구성
- 모델(ResNetV50)에서 Transfer-Learning 진행
- Augmentation Layers를 TestImage에 맞춰 3개의 Aug_Layers를 추가함
    - RandomZoom
    - RandomRotation
    - RandomFlip ( Horizontal )
- Epochs = 10 으로 진행한 결과 **Accuracy = 0.68** 의 성능을 보임


### 이성훈
[코드 확인]()
- 모델(ResNetV50)에서 Transfer-Learning 진행
- EarlyStopping을 주고 학습을 시킨 결과 멈추지 않고 계속해서 Epoch가 진행됨.
- Val_Acc 가 0.2에서 개선되지 않아 **Keras가 아닌 PyTorch 모델 구현을 고려함**
- 코드 공유 게시판을 참고하며 성능이 좋은 모델들을 가져왔지만 PyTorch의 기초지식이 부족하여 많은 시도를 해보지 못함
- swin_v2 모델이 성능이 뛰어났던 것으로 확인함.

### 이상화
[코드 확인]()
- 모델(ResNetV101)에서 Transfer-Learning 진행
- sample_submission.csv에서 요구하는 이미지에서 Class를 예측하는 모델을 인지하고 진행함
- 저해상도 Train Data를 사용했으나 성능이 크게 향상되지 않음
- **ImageNet에서 새 이미지만을 학습한 모델의 권한 요청을 시도하고 활용할 계획을 가짐**

### 오진석
[코드 확인]()
- 전처리 과정에서 csv파일을 이용한 Labeled_Dataset을 구성
- 모델(???????)에서 Transfer-Learning 진행
- **Augmentation Layers를 base_model 뒤에 배치하였으나, 피드백을 통해 모델 구조를 새로 수정함**
- 처음부터 accuracy = 0.5 에서 시작하여 0.8 이상까지 상승하여 바로 test_image를 사용하여 제출을 시도해봄
- 하지만 Test F1-Score 는 0.06 으로 저조한 성적이 나와 개선할 필요가 있음.

### 송명재
[코드 확인]()
- 공부를 위해 YOLO모델을 사용하기로 함
- Train Image의 쓸데없는 배경이 학습에 도움되지 않을 것이라고 판단. 새를 먼저 객체탐지(Object-Detection)한 후 Crop하는 과정을 거쳐 전처리 과정을 거침
- ReduceLROnPlateau() 메서드 사용 : 학습 중에 주기적으로 검증 데이터셋의 손실을 모니터링하고, 미리 정의된 조건에 따라 학습률을 감소시킴.
