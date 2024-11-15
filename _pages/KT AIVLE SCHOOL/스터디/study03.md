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

### 29조
- 인원 : 이성훈, 이상화, 황태언
- [발표자료](https://docs.google.com/presentation/d/1oxQgknJEIxAe8R-NHMRNDL_Uqu2Sx1g4/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)
