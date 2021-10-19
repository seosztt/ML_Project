# 주제: 특정 도시 월 평균 양파 가격 예측

- 이 프로젝트는 필요나 불편이 있어서 문제를 해결하려는 게 아니라 배웠던 걸 실습해보는 과정이다.
- 따라서 난이도가 적절한가,  재밌을 것 같은가를 기준으로 선정하였다.
- 조 이름 선정 : CSP(Crop Sale Pridiction)
- 프로젝트 범위 정의

  - 2021년 10월~2022년 8월 도시별 월 평균 양파 도매가격 예측
- 작물 선택 : 양파
  - 작물에 대한 도메인 지식이 없으므로 가격의 변화 추이 및 변동폭 기준 계절성과 상관관계가 높은 품목 선택
  - 소비량이 많은 대중적 품목 선택
- 수행도구, 데이터소개
  - EDA : Numpy, Pandas, Matplotlib, Seaborn
  - 머신러닝 : Sklearn 패키지 내 Regression 모델 활용



# 데이터 수집

- [ 농넷 | 농산물유통종합정보시스템](http://nongnet.or.kr/)
  - 품종, 생산지, 도매시장, 판매회사, 월별 평균 도매 가격
- [기상청](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)
  - 2013년~ 2021년 월별 도시별 기후 데이터 
- [ 농식품수출정보](https://www.kati.net/statistics/monthlyPerformanceByProduct.do)
  - 2013년~2021년 월별 양파 수입 수출량



# 전처리, EDA

- 변수에 대한 가설 설정 및 변수 선정

  - 선택 품종에 대한 사전 스터디를 통한 가설 변수 설정

    ex) 기온, 강수, 물류가격, 수확량, 수출입량, 원자재가격, 재배기간, 코로나 등

- **월별 거래가격(y)에 영향을 미치는 요인 탐색**

  - 전년도 재배면적
  - 10h당 생산량 : 기온, 강수, 일조량
  - 추가 공급량 : 수출입량 

- **데이터셋**

  - 월별/시도산지별/거래량/거래금액 : 거래가격(원/kg) 산출
  - 생산량 정보 : 연간 재배면적, 연간 생산성, 총 생산량
  - 기후 정보 : 월별 평균기온, 최고기온, 최저기온, 강수량, 일조량
  - 월별 양파 수입/수출 물량 

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

## RandomForest Regression

### permutation importance

- 특정 feature의 데이터를 shuffle 했을 때 모델의 예측 성능이 얼마나 감소했는지 보여준다.

![PI_example](https://github.com/seosztt/ML_Project/blob/master/image/PI_example.png?raw=true)

- Weight가 음수라는 건 feature의 데이터를 shuffle 했을 때 모델의 성능이 개선되었다는 뜻이다. Weight의 편차 범위 전체가 음수인 feature를 주로 제거하는 방식으로 5차에 걸쳐 제거했다.
- 73개 였던 feature의 수가 1차 23개, 2차 11개, 3차 4개, 4차 3개, 5차 2개 제거하여 최종 31개로 줄었다.
- 각 차수마다 교차검증을 했으나  RMSE의 개선이 뚜렷하게 나타나지는 않았다.
- 31개 feature가 남은 모델로 Test를 해본 결과 RMSE는 170이 나왔다.

# Program

![program](https://github.com/seosztt/ML_Project/blob/master/image/program.png?raw=true)

- 도시와 년월을 입력하면 그 도시 해당 년월의 월 평균 도매 가격 예측값이 나오는 프로그램을 작성했다.

# 향후 개선사항

- 분석 모델을 비즈니스와 어떻게 연관지을 것인가
- 비지도 학습 연관분석하여 scope 선정 시 참고

# 후기

시작할 때는 Regression 모델을 통해 미래의 어떤 값을 예측한다는 게 논리적 결함이 많을 것 같아 scope을 매우 한정적으로 정해야 할 것으로 여겼다. 그러나 이번 프로젝트를 통해 배운 점은 우리가 하는 모델링이 논리적 정합성을 따지는 일이 아니라는 것이다. 논리적 결함이 있어보여도 일단 모델을 만들어보는 게 미래 예측값 모델링을 하는 방식이다. 그 모델의 성능은 논리적 정합성이 말해주는 게 아니다. 그것은 미래가 되어 모델이 실제로 얼마나 예측을 잘했는지를 봐야 비로소 알 수 있다. 논리적 결함이 있어보였던 모델이 실제로 예측을 잘 할 수도 있다. 그러므로 지금 단계에서 할 수 있는 일은 어떤 최선의 모델을 만드는 것이 아니라 여러 모델을 만들어보고 검증을 통해서 성능이 낮을 것으로 예상되는 모델을 제외하는 과정이지 않을까 생각해봤다. 프로젝트는 마무리되지만 각 모델에 대한 평가는 새로 시작될 것이다.

