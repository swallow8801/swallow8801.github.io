---
title: "미니프로젝트 2차"
tags:
    - MiniProject
    - MachineLearning
date: "2024-10-08"
thumbnail: "/assets/img/aivleproject/proj2/thumb2.jpg"
---

# 신규 아파트 주차 수요 예측
---
![parking](/assets/img/aivleproject/proj2/parking.png)

* 기존 아파트와 실차량수 데이터를 활용하여 단지별 실차량수를 예측합니다.
* 예측한 실차량수를 통해 주차 수요를 예측하는 모델을 설계합니다.


# 데이터 전처리
---
## 원본 데이터
![data0](/assets/img/aivleproject/proj2/data00.PNG)

## 전처리 데이터
![data](/assets/img/aivleproject/proj2/data.PNG)

* 결측치 처리
* 데이터 분리 및 상세 집계
* 전용면적 구간 분리


# EDA (탐색적 데이터 분석)
---
## 상관계수 히트맵 
![Heatmap](/assets/img/aivleproject/proj2/heatmap.png)

## 지역별 실차량수 평균 그래프
![eda1](/assets/img/aivleproject/proj2/eda1.png)

## 난방방식별 실차량수 평균 그래프
![eda2](/assets/img/aivleproject/proj2/eda2.png)


# 머신러닝 모델링
---
## 모델 설계
* 간단한 머신러닝 모델을 5개 골라서 진행
* 최적의 파라미터를 찾기 위해 **GridSearchCV** 를 같이 진행

## 모델 성능 비교
![compare](/assets/img/aivleproject/proj2/modelcompare.png)

## 파이프라인 구축
```python
def data_pipeline(data):
    apt01 = data.copy()

    # 결측치 처리
    for idx, value in enumerate(new_data.isna().sum()):
        if value > 0:
            col_name = apt01.columns[idx]
            apt01[col_name] = apt01[col_name].fillna(apt01[col_name].mode()[0])
    
    # 변수 추가
    apt01['준공연도'] = apt01['준공일자'].astype(str).str[:4].astype(int)
    
    # 총면적 계산
    apt01['총면적'] = (apt01['전용면적'] + apt01['공용면적']) * apt01['전용면적별세대수']

    # 불필요한 변수 제거
    drop_cols = ['단지명', '단지내주차면수', '준공일자']
    apt01.drop(columns=drop_cols, inplace=True)
    
    # 단지 데이터
    apart_cols = ['단지코드', '총세대수', '지역', '준공연도', '건물형태', '난방방식', '승강기설치여부']
    data01 = apt01.loc[:, apart_cols]
    data01 = data01.drop_duplicates()
    data01 = data01.reset_index(drop=True)

    # 상세 데이터
    apart_cols_02 = ['단지코드', '총면적', '전용면적별세대수', '전용면적', '공용면적', '임대보증금', '임대료']
    data02 = apt01.loc[:, apart_cols_02]
    df_area = data02.groupby('단지코드')['총면적'].sum()
    
    bins = [10, 40, 80, 200]
    labels = ['원룸', '투룸', '투룸이상']

    data02['전용면적구간'] = pd.cut(data02['전용면적'], bins=bins, labels=labels)
    temp = data02.groupby(['단지코드','전용면적구간'])['전용면적별세대수'].sum().reset_index()
    
    df_pivot = temp.pivot(index='단지코드', columns='전용면적구간', values='전용면적별세대수')
    df_pivot.columns.name = None
    df_pivot.reset_index(inplace=True)
    
    df_rent = data02.groupby('단지코드')[['임대보증금', '임대료']].mean()
    
    base_data = pd.merge(data01, df_area, on='단지코드' ,how='left')
    base_data = pd.merge(base_data, df_pivot, on='단지코드' ,how='left')
    base_data = pd.merge(base_data, df_rent, on='단지코드' ,how='left')

    # 난방방식 매핑
    map_dict = {
    '개별가스난방': '개별',
    '개별유류난방': '개별',
    '지역난방': '지역',
    '지역가스난방': '지역',
    '지역유류난방': '지역',
    '중앙가스난방': '중앙',
    '중앙난방': '중앙',
    '중앙유류난방': '중앙'
    }
    base_data['난방방식'] = base_data['난방방식'].replace(map_dict)
    
    base_data = pd.get_dummies(base_data, columns=['지역', '건물형태', '난방방식'], drop_first=True, dtype=int)
    
    drop_cols2 = [ '승강기설치여부', '단지코드']
    base_data.drop(columns=drop_cols2, inplace=True)

    names = set(data['지역'].unique()) - set(new_data['지역'].unique())
    for name in names:
        base_data[f'지역_{name}'] = 0

    return base_data
```


## 결과 확인
![result](/assets/img/aivleproject/proj2/predict.png)



# 느낀점
---

- 간단한 머신러닝 모델을 설계하면서 예측해보니까 신기했다.
- 전처리 파이프라인을 만드는 게 생각보다 어려웠다.
- 최적의 파라미터를 찾는 GridSearchCV 시간이 오래걸리긴 했다.