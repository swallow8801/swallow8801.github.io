---
title: "4차 스터디 회의"
tags:
    - STUDY
date: "2024-11-24"
thumbnail: "/assets/img/thumbnail/study.jpg"
---

# 회의 주제
---
* 6차 미니프로젝트("응급상황 자동인식 및 응급실 연계") 리뷰

## 미니프로젝트 리뷰
---

### 27조 
- 인원 : 김상우, 남지윤
- [발표자료](https://docs.google.com/presentation/d/1ybrdQeGm5jyrk16X6x9ZkvaxyGKa4OWB/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- 중증도 카테고리 포함 구어체, 일상 구어체, 영문 구어체 3가지의 데이터를 수집하여 진행
- 다문화가정이나 외국인 신고까지 고려하기 위함
- NAVER MAPS API 사용하며 **소요시간**과 **도착예정시간**까지 가져와 제공함


---
### 28조
- 인원 : 전재엽, 오진석, 송명재
- [발표자료](https://docs.google.com/presentation/d/18kKphT6Bjw0cQ2I4G5OQU6RYXJ-wL3MA/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- 의학용어가 포함된 데이터와 일상용어 데이터를 활용
- 모델에 맞는 형태를 위해 요약된 상태로 데이터를 수집하여 사용함.
- Data Augmentation에 대한 고민으로 **Back Translation** 원어->외국어->원어 의 방식을 채택했으나 시간이 부족하여 적용하지 못함 

---
### 29조
- 인원 : 이성훈, 이상화, 황태언
- [발표자료](https://docs.google.com/presentation/d/1WFqawxF-Bfp58FrPya1G4Vy3gerhJYLN/edit?usp=drive_link&ouid=110582999063746602025&rtpof=true&sd=true)

#### 설명
- KTAS등급에 맞게 데이터 정제를 해야함을 느낌
- **LIME Algorithm** : 각각의 형태소별로 다른 가중치를 가지게 하거나 민감성을 가지는 형용사에 대한 조치가 필요함을 느낌




## 미니프로젝트 토의사항
---

### 1. KTAS 3등급 vs 4등급
- 대부분의 조에서 1,2,3등급은 응급실연계, 4,5등급은 조언제공 등으로 서비스를 제공했음
- 다만 3등급과 4등급을 잘못 분류하는 것은 매우 BAD
- **그렇다면 미니프로젝트3차와 같이 응급여부를 먼저 분류하는 모델을 사용한 후 세부등급을 분류하는 방향은 어떨까?**

### 2. BERT 모델 Tuning
- 이번에는 Fine-Tuning 할만한게 많지않음. HyperParameter조정으로 인한 모델의 성능향상이 크지 않았음
- **데이터 수집의 중요성이 더 느껴지는 프로젝트였다.**

### 3. 모델 Valid성능과 실성능
- 데이터를 어떻게 수집하냐에 따라 validation성능이 달라졌고 실제성능은 그렇지 않은 경우도 존재함.
- **모델의 Validation성능만을 보고 모델 자체에 대해서 평가할 수 없다는 생각이 듬.**


## 추가내용 및 후기
---

### LIME 알고리즘
개별적으로 해보길 권장!

### Data Augmentation
데이터 수집이 힘든 프로젝트였던 만큼 증강이 의미있는 프로젝트였다. BackTranslation외에 적용해볼만한 증강기법?

### ...
연달아 이은 미니프로젝트로 다들 고생한거 같아 최대한 짧게 진행했고 다들 고생 많이했어요!
