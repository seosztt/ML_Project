# 주제

`특정 도시 월 평균 양파 가격 예측`

- 프로젝트 범위 정의

  - 2021년 10월~2022년 8월 도시별 월 평균 양파 도매가격을 머신러닝을 활용하여 예측
- 작물 선택 : 양파
  - 작물에 대한 도메인 지식이 없으므로 계절에 따른 변화가 뚜렷한 품목 선택
  - 소비량이 많은 대중적 품목 선택
- 수행도구, 데이터소개
  - EDA : Numpy, Pandas, Matplotlib, Seaborn
  - 머신러닝 : Sklearn



# 데이터 수집

- [ 농넷 | 농산물유통종합정보시스템](http://nongnet.or.kr/)
  - 품종, 생산지, 도매시장, 판매회사, 월별 평균 도매 가격
- [기상청](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)
  - 2013년~ 2021년 월별 도시별 기후 데이터
- [ 농식품수출정보](https://www.kati.net/statistics/monthlyPerformanceByProduct.do)
  - 2013년~2021년 월별 양파 수입 수출량



# EDA

- 양파값에 대한 스터디를 통한 변수 후보 탐색

  - 기온, 강수, 물류가격, 수확량, 수출입량, 원자재가격, 재배기간, 코로나 등
- 변수 선정
  - 월별, 시도산지별 거래량, 거래금액을 통해 거래가격(원/kg) 산출
  - 품종, 도매시장, 판매회사
  - 생산량 데이터: 전년도 연간 재배면적, 생산성, 총 생산량
  - 기후 데이터: 전년도 월별 평균기온, 최고기온, 최저기온, 강수량, 일조량.
  - 전년도 월별 양파 수입/수출량
  - 예측을 위해서는 feature의 미래 예상값을 입력해야 하므로 기후, 생산량, 수출입 데이터는 한해 전의 데이터를 feature로 하였다.

# 모델링

## prophet

- 페이스북에서 공개한 시계열 예측 라이브러리

- prophet 알고리즘의 파라미터는 매우 직관적이기 때문에 시계열 데이터에 대한 지식이 부족하더라도 쉽게 사용할 수 있다.

- 하려는 작업에 대한 개괄을 파악하기 위해 일 평균 도매가격 데이터만 넣어서 모델을 만들어봤다.

  ![prophet](https://github.com/seosztt/ML_Project/blob/master/image/prophet.png?raw=true)

- 사용 데이터 :2014년 1월 3일부터 2021년 10월 7일까지 일 평균 양파 도매가격 (원/kg)

| 사용 데이터 | 2014년 1월 3일부터 2021년 10월 7일까지 일 평균 양파 도매가격 (원/kg) |
| ----------- | ------------------------------------------------------------ |
| train data  | 2014년 1월 3일 ~ 2020년 6월 19일                             |
| test data   | 2020년 6월 20일 ~ 2021년 10월 7일                            |
| result      | - RMSE : 328.1<br />- NMAE : 0.2886                          |



## regression

- 전처리한 데이터셋을 sklearn의 [LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]에 학습시켜 RandomForestRegressor에서 가장 좋은 결과를 얻었다.

- 코드는 RandomForestRegressor를 예시로 작성하였으며, 아래의 과정을 import한 모델만 바꿔가며 반복 수행하였다.

```python
X_train=train.drop(columns=['price'])
y_train=train['price']

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

RF=RandomForestRegressor(n_estimators = 2000,
                            max_depth = 5,
                            criterion ='mse',
                            n_jobs = -1,
                            random_state = 2045)
scores=cross_val_score(RF,
                       X_train,
                       y_train,
                       scoring='neg_mean_squared_error',
                       cv=5)
np.sqrt(-scores.mean()) # 교차검증rmse: 203.44
```



## feature selection

### permutation importance

- 특정 feature의 데이터를 shuffle 했을 때 모델의 예측 성능이 얼마나 감소했는지 보여준다.

![PI_example](https://github.com/seosztt/ML_Project/blob/master/image/PI_example.png?raw=true)

- Weight가 음수라는 건 feature의 데이터를 shuffle 했을 때 모델의 성능이 개선되었다는 뜻이다. Weight의 편차 범위 전체가 음수인 feature를 주로 제거하는 방식으로 5차에 걸쳐 제거했다.
- 73개 였던 feature의 수가 1차 23개, 2차 11개, 3차 4개, 4차 3개, 5차 2개 제거하여 최종 31개로 줄었다.
- 각 차수마다 교차검증을 했으나  RMSE의 개선이 뚜렷하게 나타나지는 않았다.
- 31개 feature가 남은 모델로 Test를 해본 결과 RMSE는 170이 나왔다.
- 제거 후 남은 기후 feature를 살펴보면
  - 판매 시기(month, year)가 영향이 컸으며 품종, 수입량, 도매시장, 수확량이 영향이 커서 경험적 직관에 부합하는 결과가 나온듯 하다.
  - 여름엔 최고기온이, 겨울엔 최저기온이 남아있어 직관에 부합하는 결과가 나왔다. 평균 기온은 7월과 10,12,1월이 남아있어서 한 여름과 한겨울의 평균기온이 영향이 있는 듯하다.
  - 일조량은 1,2,4,5,7,10,11월이 남아있어서 봄가을의 일조량이 가격과 관계가 있는 듯하고, 강수량은, 3,6,7,8,10월이 남아있으어서 주로 여름철 강수량이 영향을 미치는 듯 하나 6,7,8월의 weight가 낮다.


# Program

![program](https://github.com/seosztt/ML_Project/blob/master/image/program.png?raw=true)

- 도시와 년월을 입력하면 그 도시 해당 년월의 월 평균 도매 가격 예측값이 나오는 프로그램을 작성했다.
- kind, corp, market은 groupby city 최빈값을 입력값으로 하였다.
- 2021년 10월 11월 12월의 날씨 정보는 각 월 해당 city의 2014~20년 값의 평균을 사용하였다.

# 향후 개선사항

- 분석 모델을 비즈니스와 어떻게 연관지을 것인가
- 비지도 학습 연관분석하여 scope 선정 시 참고



# 후기

시작할 때는 Regression 모델을 통해 미래의 어떤 값을 예측한다는 게 논리적 결함이 많을 것 같아 scope을 매우 한정적으로 정해야 할 것으로 여겼다. 그러나 이번 프로젝트를 통해 배운 점은 우리가 하는 모델링이 논리적 정합성을 따지는 일이 아니라는 것이다. 논리적 결함이 있어보여도 일단 모델을 만들어보는 게 미래 예측값 모델링을 하는 방식이다. 그 모델의 성능은 논리적 정합성이 말해주는 게 아니다. 그것은 미래가 되어 모델이 실제로 얼마나 예측을 잘했는지를 봐야 비로소 알 수 있다. 논리적 결함이 있어보였던 모델이 실제로 예측을 잘 할 수도 있다. 그러므로 지금 단계에서 할 수 있는 일은 어떤 최선의 모델을 만드는 것이 아니라 여러 모델을 만들어보고 검증을 통해서 성능이 낮을 것으로 예상되는 모델을 제외하는 과정이지 않을까 생각해봤다. 프로젝트는 마무리되지만 각 모델에 대한 평가는 새로 시작될 것이다.

