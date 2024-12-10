---
title: "미니프로젝트 1차"
tags:
    - MiniProject
    - EDA
date: "2024-09-25"
thumbnail: "/assets/img/aivleproject/proj1/thumb1.jpeg"
---

# 서울시 생활정보 기반 대중교통 수요 분석
---
![station](/assets/img/aivleproject/proj1/busstation.jpg)

* 사용 데이터
  * 서울시 버스노선별 정류장별 승하차 인원 정보
  * 서울시 버스정류장 위치정보
  * 서울시 구별 이동 데이터
  * 서울시 구별 주민 등록 데이터
  * 서울시 구별 등록 업종 상위 10개 데이터

* 개요
  * 서울시 생활정보 데이터를 기반으로 데이터 분석을 시도함.
  * 공공 데이터를 활용해 버스노선 관련 인사이트를 도출함. 



# EDA (탐색적 데이터 분석)
---

## 단변량 분석

### 함수 선언
```python
def eda(col, min_max=min):
    if min_max == min:
        third_extreme = df[col].nsmallest(3).iloc[-1]
        mask = df[col] <= third_extreme
    elif min_max == max:
        third_extreme = df[col].nlargest(3).iloc[-1]
        mask = df[col] >= third_extreme

    plt.figure(figsize=(10, 6))

    sns.barplot(x='자치구', y=col, data=df, palette=['red' if m else 'blue' for m in mask])

    plt.axhline(y=df[col].mean(), color='green', linestyle='--', label=f"{col} 평균")

    plt.title(f"{col} 자치구별 비교 ({'최소값' if min_max == 'min' else '최대값'})")
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```
### 자치구별 노선수
![eda1](/assets/img/aivleproject/proj1/eda1.png)

### 자치구별 이둥인구(합)
![eda2](/assets/img/aivleproject/proj1/eda2.png)

### 자치구별 총 이동인구
![eda3](/assets/img/aivleproject/proj1/eda3.png)


## 이변량 분석

### 가설 검증을 위한 Heatmap 시각화
![eda5](/assets/img/aivleproject/proj1/eda5.png)

### 상관계수 분석 및 산점도/회귀선 시각화
```python
def paint_scatter(x, y):\
    # x, y는 분석에 사용할 칼럼 이름
    # 산점도 그리기
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x}과 {y}의 관계')
    # 산점도의 각 점에 자치구 이름 표시
    for i, row in df.iterrows():
        plt.text(row[x], row[y], row['자치구'])
    # 회귀선 그리기
    sns.regplot(x=x, y=y, data=df, line_kws={'color': 'red'})
    plt.show()
    
    # 상관계수와 p-value
    print(spst.pearsonr(df[x], df[y]))
```
![eda4](/assets/img/aivleproject/proj1/eda4.png)


### 전체 데이터 상관계수 히트맵 시각화
![eda6](/assets/img/aivleproject/proj1/eda6.png)


## 결론 도출

### 버스 노선 혹은 정류장이 필요한 구는?
* 강동구
* 송파구
* 강서구

### 이유는?
* 가설 검증 및 시각화를 통해 하위 구를 선정했고 가장 많이 해당되었기 때문이다.



# 느낀점
---

- 첫 미니프로젝트라서 소통이 어색했다.
- 데이터 분석만으로는 인사이트 도출이 힘들었으나 다양한 의견을 주고받을 수 있어서 좋았다.
- 다음에는 강의장에서 대면으로 회의를 진행해보면 좋겠다!