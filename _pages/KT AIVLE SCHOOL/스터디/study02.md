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
* 공모전 주제 선정

## DACON 진행상황 공유
---
![DACON1](/assets/img/study/dacon1.PNG)
- 진행기간 : 2024-11-03 ~ 2024-11-17
- 시각지능 개별실습을 위한 개별과제 부여


### 전재엽
[코드 확인](https://colab.research.google.com/drive/1P2mQOq6D0vX9Lqb5ykraQsMiJ8BhhDRL?usp=sharing)
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
- 저해상도 Train Data를 사용했으나 성능이 크게 향상되지 않아 Upscale Train Data 사용
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

---
### 공통 주제
1. Upscale_Data 활용방안에 대해 : 별도로 주어진 Upscale_image를 어떻게 활용할 것인지
2. Keras VS PyTorch : 수업중 진행한 익숙한 Keras를 사용할 지, 공모전이서 주로 사용되는 PyTorch를 배우면서 진행할 지
3. Ensemble : TOP5 의 앙상블 기법이 상당히 많음. 여러 모델을 사용하며 앙상블하는 방법에 대한 고민
4. GPU : 학습시간이 굉장히 오래걸려 여러 시도를 하기 힘들었던 것 같음.

---
## 공모전 주제 선정
---
### 2024 청소년데이터 분석활용 공모전 
![Cont1](/assets/img/study/cont1.PNG)

### 2024 문화체육관광 데이터 활용대회
![Cont2](/assets/img/study/cont2.PNG)

4명,4명으로 나누어 각자 원하는 공모전에 도전할 계획을 세우며 마무리했습니다.
