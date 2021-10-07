# 주제 선정

- 이 프로젝트는 필요나 불편이 있어서 문제를 해결하려는 게 아니라 배웠던 걸 실습해보는 과정이다.
- 따라서 난이도가 적절한가,  재밌을 것 같은가를 기준으로 선정하였다.
- 최종 주제 선정 : 농산물 가격 예측 https://www.dacon.io/competitions/official/235801/data
- 조 이름 선정 : CSP(Crop Sale Pridiction)

- 프로젝트 범위 정의
- 작물 선택
  - 가격의 변화 추이 및 변동폭 기준 계절성과 상관관계가 높은 품목 우선
  - 소비량이 많은 대중적 품목 선택

- 변수에 대한 가설 설정 및 1차 선정

  -  선택 품종에 대한 사전 스터디를 통한 가설 변수 설정

    ex. 기온, 강수, 물류가격, 수확량, 수출입량, 원자재가격, 재배기간, 코로나 등

- 데이터 수집 및 전처리
  - 품종별 일일 거래량 및 가격 :[ 농넷 | 농산물유통종합정보시스템](http://nongnet.or.kr/)
  - 독립변수용 외부데이터 : 기후정보([기상청](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)),[ 농식품수출정보](https://www.kati.net/statistics/monthlyPerformanceByProduct.do) 등
- 가격예측 모델을 위한 머신러닝 모델 탐색 및 1차 테스트
- 모델 설계, 평가, 검증 반복을 통한 예측 정확도 개선
- 수행도구, 데이터소개
  - EDA : Numpy, Pandas, Matplotlib, Seaborn
  - 머신러닝 : Sklearn 패키지 내 Regression 모델 활용

# 데이터 수집, 전처리, EDA

- 분석 모델을 비즈니스와 어떻게 연관지을 것인가?
- 기온, 강수량, 토양의 질 등 가격에 영향을 미치는 다른 변수 데이터를 찾는다.
- 파생변수 생성?
- 비지도 학습 연관분석하여 작물 선택 시 참고?

