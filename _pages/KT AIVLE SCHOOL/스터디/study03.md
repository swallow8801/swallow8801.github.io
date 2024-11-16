---
title: "3차 스터디 회의"
tags:
    - STUDY
date: "2024-11-15"
thumbnail: "/assets/img/thumbnail/study.jpg"
---

# 회의 주제
---
* 6차 미니프로젝트("시계열 데이터 기반 상품 판매량 예측") 리뷰

## 미니프로젝트 리뷰
---

### 27조 
- 인원 : 김상우, 남지윤
- [발표자료](https://docs.google.com/presentation/d/1loOAjxrzyYfRfvmB7xvi6fiXxznS5Mb-/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- 페르소나 : 재고관리 입장에서 프로젝트를 진행함
- Oil Price 데이터 삭제
- 공휴일을 찾아 판매량 패턴을 분석
- 11월 ThanksGivingDay 대비하기 때문에 우유 판매량이 높을 것이다.
    - -> 공휴일/공휴일D-1/공휴일D-2 칼럼추가
- 가장 최근 4개월을 Validation으로 설정. 11월을 포함시키기 위함.

- **모델링 평가**
    - 농산물(42번) 결과가 제일 안좋았고 CNN이 예측을 더 잘하는 것 같았음. 시계열 데이터의 이상치 예측을 잘하는 것으로 추정
    - Base Model의 성능이 괜찮아서, 튜닝으로 인한 성능 향상이 거의 없었고, 더 떨어지는 경우도 있었다.
    - **GridSearch를 사용하여 최적의 HyperParameter를 찾는 방법을 시도함** 다만 시간이 오래걸렸다.
- **비즈니스 평가**
    - 공휴일 4/3/2일전에 발주량을 많이 해야할 것.
    - 11월은 전체적으로 판매량이 높아 안전 재고로 남기는 것이 좋아보였음.

---
### 28조
- 인원 : 전재엽, 오진석, 송명재
- [발표자료](https://docs.google.com/presentation/d/1C9mm9pbhM1d3nky-IPDG_GyIPwj706UT/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- 휴일과 주말은 비슷한 추이를 보여, 동시에 묶어서 파생변수를 생성하기로 함.
- Oil Price 데이터 삭제. 상관계수 0.12로 불필요한 데이터로 판단함.
- **Holt-Winters : 데이터의 계절성 특성을 확인하고 ACF,PACF 분석을 통한 자기상관성 분석을 진행함**
- 동일 지역 매장 판매 추세가 비슷하므로 활용하는 방법을 고민해봄.
    - -> 하지만 다중공산성 문제로 인해 가설 기각됨


- **모델링 평가**
    - 우유(12번) 결과가 제일 안좋았음
    - Base Model의 성능이 괜찮은 편이었으나, 직접 파라미터 조정을 하며 튜닝을 한 결과 개선된 모습을 보였다.
- **비즈니스 평가**
    - **유통기한 데이터를 활용하여 낭비될 수 있는 재고량을 추정하여 더 좋은 예측 시스템을 구성했을 것 같다.**
    - 계절적 특성 및 시계열 특성 활용을 위한 이동평균 등 다양한 변수 활용을 하지 못해 아쉬웠다.

---
### 29조
- 인원 : 이성훈, 이상화, 황태언
- [발표자료](https://docs.google.com/presentation/d/1oxQgknJEIxAe8R-NHMRNDL_Uqu2Sx1g4/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- 7/14일 이동평균 칼럼 추가, 5일/7일 전 칼럼 추가, 주말여부/계절/휴일 칼럼 추가

- **아쉬운점 및 개선사항**
    - 1월1일 판매량 0 에 대한 처리를 어떻게 할지 좀 더 고민해볼 수 있었는데 아쉬웠다.
    - CNN,LSTM 개량된 것을 짜보고 싶은데 아쉬웠다.
    - etc



## 미니프로젝트 토의사항
---

### 1. 유가데이터를 왜 넣어줬을까?
유가데이터와 연관성을 찾을 수 있었거나, 혹은 관련상품은 존재했을 수 있다고 판단된다.

### 2. 상품별로 다른 전처리 방식을 통한 서로 다른 데이터셋을 구성하는 방안
상품들의 특징에 따라 필요로 하는 변수나 불필요한 변수가 존재할 수 있으므로 서로 다른 전처리방식을 사용하는 것이 좋았을 것 같다.

### 3. 계절적 특성을 반영한 모델 구성 및 전처리
이동평균이나 과거 데이터 등 여러 변수를 적절히 활용하는 아이디어가 필요했을 것이다.

### 4. 변수를 적절히 줄이는 방법도 고려해야 했을 것이다.
불필요한 변수가 학습에 방해되는 경우도 있었다. 28조의 경우, 7개의 column으로도 충분히 좋은 성능을 보였다.

### 5. 데이터 분석의 중요도
가설 검증을 위한 데이터 분석이 굉장히 중요했던 것 같고, 전처리 과정에 중요한 역할을 했다.


## 추가적으로 공부해볼 것
---

### 1. 딥러닝모델 GridSearch
![DNN GridSearch](/assets/img/study/GridSearch.PNG)
- 딥러닝 모델 역시 GridSearch를 통해 최적의 파라미터를 찾을 수 있다.
- 은닉층, learning_rate, batch_size, epochs 등을 조절할 수 있다.


### 2. Holt-Winter Exponential Smoothing
이게 뭘까요? 계절적 특성 분석을 위해 사용했던 것입니다.

### 3. ACF,PACF 자기상관성 분석
이게 뭘까요? 자기상관성은 뭐고 어떻게 분석하는 걸까요?

### 4. ARIMA, SARIMA
계절적 특성을 지닌 데이터를 처리하는 데에 적합한 모델이라고 하는데 뭘까요?

### 5. Infinite MAPE
- 실제값이 0이 있을 경우 MAPE가 무한대의 값으로 수렴하게 되는 문제가 발생한다.

#### 성능평가지표 및 함수 소개
![Inf_mape](/assets/img/study/inf_mape.PNG)
- SMAPE(Symmetric Mean Absolute Percentage Error)

$\text{Log MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \log(y_i + \epsilon) - \log(\hat{y}_i + \epsilon) \right|$
- Log MAPE (로그값을 취한 MAPE)

$\text{SMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$
- Filtered MAPE (0값을 제외한 MAPE)

#### 출력결과
![Inf_mape_result](/assets/img/study/inf_mape_result.PNG)
