---
title: "미니프로젝트 3차"
tags:
    - MiniProject
    - DNN
    - Keras
date: "2024-10-18"
thumbnail: "/assets/img/aivleproject/proj3/thumb3.PNG"
---

# 스마트폰 센서 기반 모션 분류
---
![sensor](/assets/img/aivleproject/proj3/sensor.PNG)

* 561개의 센서 데이터와 6개의 행동 패턴으로 이루어진 데이터 활용
* **선택과 집중**을 위해 변수 중요도 추출을 통해 필요한 센서만 사용
* **정적행동** 과 **동적행동** 분류 모델과 세부 행동 분류 모델 설계
* 데이터 파이프라인 설계

# EDA (탐색적 데이터 분석)
---

## RandomForest 변수 중요도 추출

### 모델 Predict
![Predict](/assets/img/aivleproject/proj3/eda1.PNG)

### 변수중요도 분석
![FeatureImportance](/assets/img/aivleproject/proj3/eda2.png)

### 상위 변수 데이터분석
![eda3](/assets/img/aivleproject/proj3/eda3.png)
![eda4](/assets/img/aivleproject/proj3/eda4.png)
![eda5](/assets/img/aivleproject/proj3/eda5.png)

### 이진분류 및 다중분류 센서별 중요도 시각화
![eda6](/assets/img/aivleproject/proj3/eda6.png)
* 정적&동적 분류와 세부행동 분류의 센서별 중요도가 차이가 있음.
* 따라서 정적 및 동적 분류 모델을 먼저 만든 후
* 세부행동 분류하는 모델을 추가로 만들기로 계획함.


# 모델링
---

## 여러 모델 실험

### Baseline Model
![model1](/assets/img/aivleproject/proj3/model1.png)

### Model + HiddenLayer 
![model2](/assets/img/aivleproject/proj3/model2.png)

### Model + HiddenLayer + Dropout + EarlyStopping
![model3](/assets/img/aivleproject/proj3/model3.png)

### Model + HL + Dropout + ES + BatchNorm
![model4](/assets/img/aivleproject/proj3/model4.png)

### PCA 차원축소 + Model
![model5](/assets/img/aivleproject/proj3/model5.png)


## 모델 성능 비교
![model6](/assets/img/aivleproject/proj3/model6.png)
```python
# PCA 차원축소

from sklearn.decomposition import PCA

pca = PCA(n_components=100)
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)

validation_losses = []
validation_accuracies = []

for i in range(5):
    print(f'{i+1}번째 반복수행중 . . . ')
    clear_session()
    nfeatures_p = x_train_pca.shape[1]

    model5 = Sequential([Input(shape=(nfeatures_p,)),
                         Dense(6, activation='softmax')])
    
    model5.compile(optimizer=Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    history5 = model5.fit(x_train_pca, y_train, epochs=100, validation_split=.2, verbose=0).history

    validation_loss = history5['val_loss'][-1]
    validation_accuracy = history5['val_accuracy'][-1]
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)
```


# 느낀점
---
* 딥러닝 모델 설계를 하면서 다양한 시도를 해야해서 어려웠다.
* DropOut 비율과 HiddenLayer 수 조절하는 게 굉장히 어려웠다.
* 성능이 왜 상승하는 지 알 수가 없으니 좀 어려웠다.