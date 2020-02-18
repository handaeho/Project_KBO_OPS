# 2019 KBO타자 성적 예측

# 타자의 OPS(On base Plus Slugging, 출루율 + 장타율)을 예측해보자.
# OPS => ‘얼마나 자주 루상에 나가느냐’ 와 ‘얼마나 많은 루를 나가느냐’.
# 따라서 출루율이나 장타율과 달리 OPS는 두개를 한꺼번에 보여줄수 있다는 면에서 득점과 가장 밀접한 지표라고 볼 수 있다.

# Host : ①데이콘 ②서울대학교 통계연구소 ③수원대 DS & ML 센터 ④한국야구학회 ⑤기타

# Background : 2016년 관중수가 800만명을 돌파한 프로야구는 명실공히 한국 프로스포츠 최고의 인기 종목.
# 프로야구의 인기와 더불어 데이터 분석에 대한 인식이 높아짐에 따라 국내 여러 구단에서 데이터 사이언스 역할의 수요가 늘고 있다.
# 특히 야구에서는 특정 선수의 성적 변동성이 해마다 매우 크기 때문에 내년 성적을 예측하기 까다로운 부분이 많다.
# 정말 못 할 것이라고 생각했던 선수도 막상 내년에는 잘하고, 많은 지표가 리그 상위권이었던 선수가 내년에는 그렇지 않은 경우가 많다.

# Evaluation : WeightedRootMeanSquaredError(WRSME)
# 선수들의 타수를 가중치로 둔 RMSE. 예측 공식은 WRSME = sqrt(∑{(실제값 - 예측값)^2 * 타수} / ∑타수)

# [Files]
# ① Regular_Season_Batter.csv : KBO에서 활약한 타자들의 역대 정규시즌 성적을 포함하여 몸무게, 키 ,생년월일 등의 기본정보
# ② Regular_Season_Batter_Day_by_Day.csv: KBO에서 활약한 타자들의 일자 별 정규시즌 성적
# ③ Pre_Season_Batter.csv : KBO에서 활약한 타자들의 역대 시범경기(정규시즌 직전에 여는 연습경기) 성적
# ④ submission.csv : 참가자들이 예측해야 할 타자의 이름과 아이디 목록

#######################################################################################################################
# 필요 패키지 import
import pandas as pd # 데이터 프레임 형태를 다루는 패키지
import matplotlib.pyplot as plt # 시각화 패키지
import seaborn as sns # 시각화 패키지
plt.style.use('fivethirtyeight')
import os # 데이터 로드시, 디렉토리 설정 패키지
import numpy as np # 선형대수(행렬 등) 계산 패키지

# 지정 작업의 현재 디렉토리 변경(CSV 파일 다운로드 위치)
os.chdir("/home/daeho/다운로드/drive-download-20191022T014909Z-001")

# 정규시즌 타자 데이터
regular = pd.read_csv("Regular_Season_Batter.csv")
regular = regular.loc[~regular['OPS'].isnull(), ] # OPS == NULL 제외

# 예측할 타자 데이터
submission = pd.read_csv("submission01.csv")

# 2019년 OPS 예측을 위해, 타자 정보가 저장될 데이터 프레임 생성
agg={} # agg 딕셔너리 생성. 딕셔너리 : key : value를 가짐. key 호출시, 해당 value가 나옴.
for i in regular.columns: # i가 정규시즌 데이터 컬럼 수만큼 증가 하는동안
    agg[i] = [] # agg 딕셔너리에 i(regular 컬럼)를 키로, 그 컬럼에 해당하는 값을 value로 추가

# i는 행, j는 열 iloc[index]: 컬럼명이 아니라 index 숫자로 데이터를 가져옴. 반환할때도 시리즈가 아닌 숫자만 가져온다.
for i in submission['batter_name'].unique(): # i(행)가 예측할 타자들 수만큼 증가 하는동안 -----> unique() : 배열 내에서 중복을 제거, 유니크한 값을 보여줌
   for j in regular.columns: # j(열)가 정규시즌 데이터 컬럼 수만큼 증가하는 동안
        if j in ['batter_id', 'batter_name', 'height/weight', 'year_born', 'position', 'starting_salary']:# j(열)가 해당 정보면
            agg[j].append(regular.loc[regular['batter_name'] == i, j].iloc[0]) # agg 딕셔너리에 j가 가리키는 스탯 컬럼 이름을 key로, i가 가리키는 타자의 이름의 j가 가리키는 스탯 데이터를 value로 추가하고(숫자형, iloc[])
        elif j == 'year': # j(열)가 연도면
            agg[j].append(2019) # 2019를 추가
        else: # 그것도 아니면
            agg[j].append(0) # 0을 붙여라
regular = pd.concat([regular, pd.DataFrame(agg)]) # 정규시즌 데이터에 agg 데이터 프레임을 붙인다.

# 정규시즌 데이터 확인
pd.set_option('display.max_columns', 500) # 컬럼 생략없이 모두 나오게
print(regular.head(10)) # 상위 10개 출력
#    batter_id batter_name  year team    avg    G   AB   R   H  2B  3B  HR      TB   #
# 0          0        가르시아  2018   LG  0.339   50  183  27  62   9   0   8    95
# 1          1         강경학  2011   한화  0.000    2    1   0   0   0   0   0    0
# 2          1         강경학  2014   한화  0.221   41   86  11  19   2   3   1   30
# 3          1         강경학  2015   한화  0.257  120  311  50  80   7   4   2  101
# 4          1         강경학  2016   한화  0.158   46  101  16  16   3   2   1   26
# 5          1         강경학  2017   한화  0.214   59   84  17  18   2   1   0   22
# 6          1         강경학  2018   한화  0.278   77  245  42  68  11   1   5   96
# 7          2         강구성  2013   NC  0.000    2    2   0   0   0   0   0     0
# 8          2         강구성  2015   NC  0.200    4    5   0   1   1   0   0     2
# 9          2         강구성  2016   NC  0.000    2    3   0   0   0   0   0     0
#
#    RBI  SB  CS  BB  HBP  SO  GDP    SLG    OBP   E height/weight  \
# 0   34   5   0   9    8  25    3  0.519  0.383   9    177cm/93kg
# 1    0   0   0   0    0   1    0  0.000  0.000   1    180cm/72kg
# 2    7   0   0  13    2  28    1  0.349  0.337   6    180cm/72kg
# 3   27   4   3  40    5  58    3  0.325  0.348  15    180cm/72kg
# 4    7   0   0   8    2  30    5  0.257  0.232   7    180cm/72kg
# 5    4   1   1   8    1  19    1  0.262  0.290   4    180cm/72kg
# 6   27   6   3  38    4  59    7  0.392  0.382   2    180cm/72kg
# 7    0   0   0   0    0   0    0  0.000  0.000   0    180cm/82kg
# 8    0   0   0   0    0   0    0  0.400  0.200   0    180cm/82kg
# 9    0   0   0   0    0   1    0  0.000  0.000   0    180cm/82kg
#
#         year_born        position                                         career   starting_salary   OPS
# 0  1985년 04월 12일  내야수(우투우타)         쿠바 Ciego de Avila Maximo Gomez Baez(대)             NaN  0.902
# 1  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.000
# 2  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.686
# 3  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.673
# 4  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.489
# 5  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.552
# 6  1992년 08월 11일  내야수(우투좌타)                       광주대성초-광주동성중-광주동성고        10000만원  0.774
# 7  1993년 06월 09일  외야수(우투좌타)                       관산초-부천중-야탑고-NC-상무           9000만원  0.000
# 8  1993년 06월 09일  외야수(우투좌타)                       관산초-부천중-야탑고-NC-상무           9000만원  0.600
# 9  1993년 06월 09일  외야수(우투좌타)                       관산초-부천중-야탑고-NC-상무           9000만원  0.000

#######################################################################################################################

# 야구는 여러 영향(날씨, 컨디션, 운 등)을 은근히 많이 받는 스포츠이다. 특히 '운'이 따르기도 하는 스포츠인데,
# A 선수가 평소라면 안타나 홈런이 될 타구가 번번히 상대 수비에 막히거나, 파울이 되었다.
# 그런데 B 선수는 평소라면 아웃이 되어야 하는 타구가 운 좋게도 기록되지 않는 상대 실책등으로 안타나 출루가 되었다.
# 그렇다면 단순하게 기록만 보고서 B 선수가 A 선수보다 잘한다고 할 수 있을까?

# 운이 좋았던 기록 걸러내기. -----> 자기상관도 분석. 즉, 자기 스스로와의 상관도를 분석해, 상관도가 높다면 그 스탯은 '실력', 상관도가 낮다면 그 스탯은 '운'
def get_self_corr(var,regular = regular): # get_self_corr 함수 선언
    x = [] # x 배열
    y = [] # y 배열
    regular1 = regular.loc[regular['AB'] >= 50,] # 50타석 이상 선수 추출해, regularl에
    for name in regular1['batter_name'].unique(): # 타자 수만큼 반복
        a = regular1.loc[regular1['batter_name'] == name, ].sort_values('year') # 연도순으로 정렬 후, 타자이름을 a에 할당.
        k = [] # k 배열
        for i in a['year'].unique(): # 정렬후, 중복이 제거된 연도만큼 반복(..., 2007, 2008, 2009, ..., 2017)
            if (a['year'] == i+1).sum() == 1:
                k.append(i) # k 배열에 i를 채움
        for i in k: # 채워진 k 배열 수만큼
            x.append(a.loc[a['year'] == i, var].iloc[0]) # x 배열에 k 배열의 연도와 a 배열의 연도가 같은, 밑에서 전달 받은 스탯 값을 추가.
            y.append(a.loc[a['year'] == i+1, var].iloc[0]) # y 배열에 k 배열의 연도+1과 a 배열의 연도가 같은, 밑에서 전달받은 스탯 값을 추가.
    plt.scatter(x, y) # 시각화
    plt.title(var)
    plt.show() # 그래프 출력
    print(pd.Series(x).corr(pd.Series(y))**2) # y 변수에 대한 x의 corr. x와 y는 올해와 다음해의 같은 스탯값. 즉 자기 상관계수 연산후, 출력

# 안타(H)는 단타, 2루타, 3루타, 홈런을 모두 합친 기록. 따라서 단타를 구하기 위해 H에서 2B, 3B, HR을 빼야함.
regular['1B'] = regular['H'] - regular['2B'] - regular['3B'] - regular['HR']

# 기록 별, 자기상관도 파악
for i in ['avg','1B','2B','3B']:
    get_self_corr(i) # get_self_corr 함수의 인자 var에 i가 가리키는 스탯 이름 전달
# 타격의 지표는 전반적으로 자기상관도가 낮다.(avg(약 0.17), 1B(약 0.35), 2B(약 0.32), 3B(약0.20))

for i in ['HR', 'BB']:
    get_self_corr(i)
# 볼넷(약 0.45)과 홈런(약 0.55)은 자기상관도가 높은 편. 즉, 실력을 나타낼수 있는 지표가 된다.

# 위의 결과로 보아 운적인 요소에 많이 영향을 받는 스탯: 1B,2B,3B / 운적인 요소에 많이 영향을 받지 않는 스탯: HR,BB

# 영향을 많이 받는 스탯에서 영향을 덜 받는 스탯을 제외해, 진짜 '운'을 보자.
regular['1b_luck'] = regular['1B'] / (regular['AB'] - regular['HR'] - regular['SO'])
regular['2b_luck'] = regular['2B'] / (regular['AB'] - regular['HR'] - regular['SO'])
regular['3b_luck'] = regular['3B'] / (regular['AB'] - regular['HR'] - regular['SO'])

# regular 데이터에 '운' 지표를 추가하고 스탯 컬럼을 'lag' 변수화
# lag 변수 : 원하는 값을 구하기 위해 기반이 되는 변수. 예를 들어 현재의 OPS를 구하는 기반이 되는 1년전 타율은 lag_2_avg, 3년전 OPS는 lag_3_OPS등으로 나타낸다.

# i = 0이면 1번 행인 '가르시아'의 정보가 나오는데, 우리가 원하는 직전년도인 2018년도에는 활동하지 않아, 데이터가 없다. 따라서 시리즈가 0이 되어 out of bound 에러 발생
# 따라서 len(~~~) == 0. 즉, 직전년도의 스탯 기록이 없으면 nan을 삽입하고, 아니라면 그 해당 년도의 스탯을 삽입

for j in ['avg', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'SLG', 'OBP',
          'E', '1b_luck', '2b_luck', '3b_luck']: # j는 각 스탯 컬럼 범위동안 반복.
    lag_1_avg = [] # lag_1_avg 데이터 프레임 생성
    for i in range(len(regular)): # i는 regular 크기만큼 반복
        if len(regular.loc[(regular['batter_name'] == regular['batter_name'].iloc[i]) &
               (regular['year'] == regular['year'].iloc[i] - 1)][j]) == 0: # 해당 타자 이름과 그 타자의 번호가 같고, 직전년도의 j가 가리키는 스탯이 0이면(기록이 없으면),
            lag_1_avg.append(np.nan) # lag_1_avg 데이터 프레임에 nan 넣음.
        else:
            lag_1_avg.append(regular.loc[(regular['batter_name'] == regular['batter_name'].iloc[i]) &
                             (regular['year'] == regular['year'].iloc[i] - 1)][j].iloc[0]) # lag_1_avg 데이터 프레임에 직전년도 스탯을 채움.

    regular['lag_1_' + j] = lag_1_avg # 원래 regular 데이터에 lag_1_스탯명 컬럼을 추가
    print(j)
print(regular.columns)

# 통산 커리어 구하기
def get_total_career(name,year,var):
    if (len(regular.loc[(regular['batter_name'] == name) & (regular['year'] < year-1), 'H']) != 0): # 직전년도 전까지 안타가 있으면
        return regular.loc[(regular['batter_name'] == name) & (regular['year'] < year-1), var].sum() # 안타 수 총합 계산
    else:
        return np.nan # 안타가 없으면 nan 값 채움

for i in ['G', 'AB', 'R', 'H','2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO']:
    regular['total_'+i] = regular.apply(lambda x: get_total_career(x['batter_name'], x['year'], i), axis=1)
    # 타자 이름과 연도, i가 가리키는 스탯을 get_total,career 함수의 name, year, var로 전달해서, regular에 'total_각 스탯 이름(i)'으로 저장
    print(i)
print(regular.columns)

# RandomForestRegressor Model => 개별 트리모델을 여러개 만들어, 서로 앙상블 시키는 모델
from sklearn.ensemble import RandomForestRegressor
#
# train = regular.loc[regular['year'] <= 2017,] # 2017년도까지의 데이터를 학습 데이터 셋으로
# test = regular.loc[regular['year'] == 2018,] # 2018년도 데이터를 테스트 데이터 셋으로
# y_train = train['OPS'] # OPS를 예측하기 위해서 2017년도까지의 OPS를 학습.
# X_train = train[[x for x in regular.columns if ('lag' in x)|('total' in x)]]
# # x for x in ~~~> 모든 문자를 반복(regular 컬럼을 모두 반복), lag 컬럼 또는 total 컬럼에 해당 값이 숫자면 X_train 목록에 넣음
#
# y_test = test['OPS'] # 학습된 데이터로 OPS 예측을 테스트 하기 위해로 OPS를 레이블
# X_test = test[[x for x in regular.columns if ('lag' in x) | ('total' in x)]]
# # x for x in ~~~> 모든 문자를 반복(regular 컬럼을 모두 반복), lag 컬럼 또는 total 컬럼에 해당 값이 숫자면 X_test 목록에 넣음
#
# # RandomForestRegressor 모델 생성
# rf = RandomForestRegressor(n_estimators = 500)
# rf.fit(X_train.fillna(-1), y_train, sample_weight = train['AB'])  # X_train에 na값은 -1로 대체. 가중치는 타석 수(1타석 1홈런보다 100타석 10홈런이 OPS는 작아도 가치 더 크니까)
# # 위에서 언급한 WRMSE는 타석 수를 가중치로 부여. 즉, 더 많이 타석에 들어온 선수에게 가중치를 부여한다.
#
# # 모델 성능 평가
# pred = rf.predict(X_test.fillna(-1))
#
# # 테스트 데이터 셋의 OPS를 real, 타석 수를 ab에 할당
# real = test['OPS']
# ab = test['AB']
#
# # 예측된 OPS와 실제 OPS간의 차이 분석
# from sklearn.metrics import mean_squared_error
# final_predict = mean_squared_error(real,pred,sample_weight=ab)**0.5 # 위에서 언급한 RMSE 예측 공식 이용
# print(final_predict) # 오차율 0.12514610344125332
# print("ACC =" + 1 - final_predict + "%")

# 2018년도까지의 데이터로 2019년(올해) 타자 OPS 예측하기
train = regular.loc[regular['year'] <= 2018, ] # 2018년도까지의 데이터를 학습 데이터 셋으로
test = regular.loc[regular['year'] == 2019, ] # 예측할 2019년(올해)의 데이터를 테스트 데이터 셋으로

y_train = train['OPS'] # 학습 데이터 셋의 OPS를 레이블로
X_train = train[[x for x in regular.columns if ('lag' in x) | ('total' in x)]] # OPS를 제외한 스탯들을 넣어 모델에 대입

rf = RandomForestRegressor(n_estimators = 500) # 개별 트리 500개를 만들고 앙상블
rf.fit(X_train.fillna(-1), y_train, sample_weight = train['AB']) # OPS를 제외한 스탯들의 NA값은 -1로 대체 후, 타석 수를 가중치로

test = regular.loc[regular['year'] == 2019, ] # 2019년의 데이터를 행으로 가지고 있는 테스트 배열 생성.
pred = rf.predict(test[[x for x in regular.columns if ('lag' in x) | ('total' in x)]].fillna(-1))
# 만들어진 모델 rf_01에 lag 컬럼 또는 total 컬럼의 테스트 데이터를 넣어 예측 데이터 생성. NA값은 -1로 대체.

print(pred) # 예측된 데이터 출력

# [0.70013743 0.68574905 0.3878261  0.36712156 0.79469327 0.95277682
#  0.46924199 0.63231229 0.56305413 0.60328684 0.64634392 0.72806247
#  0.88602982 0.68466606 0.63491605 0.74686319 0.85797697 0.65890992
#  0.55661762 0.78750502 0.54957674 0.61089874 0.70691016 0.76450856
#  0.49126188 0.68991049 0.69717173 0.66978386 0.5304121  0.71814013
#  0.72581306 0.7480059  0.70637041 0.64137584 0.62147452 0.28560706
#  0.65141973 0.63728402 0.3696927  0.80969778 1.03231109 0.64072877
#  0.88540021 0.62102208 0.63286872 0.6171699  0.670066   0.72375088
#  0.58172502 0.75211812 0.64639057 0.81747285 0.79494055 0.88183927
#  0.50911407 0.67052371 0.71843069 0.50668551 0.89781348 0.54106435
#  0.52267776 0.78758416 0.90359417 0.53570005 0.7864195  0.73081537
#  0.61557191 1.00960344 1.03041674 1.01534364 0.64645591 0.78130493
#  0.69974589 0.8289153  0.83217676 0.83377098 0.607757   0.59068455
#  0.75161815 0.73101582 1.02998221 0.82259671 0.67597698 0.60115228
#  0.81437114 0.76507988 0.55132764 0.66236124 0.66800475 0.58824064
#  0.74745685 0.76318702 0.564209   0.61085075 0.5090088  0.61655196
#  0.58030187 0.55767201 0.6900797  0.80016379 0.84396224 0.83092861
#  0.7259991  0.71536299 0.60888429 0.69925219 0.88273201 0.63523018
#  0.79011893 0.62158998 0.70898389 0.86770559 0.49389505 0.75370892
#  0.58982999 0.66035068 0.6110644  0.63246101 0.9153976  0.79766865
#  0.68253184 0.91615016 0.53592929 0.65161494 0.83161453 0.90904659
#  0.56677904 0.82916519 0.74614423 0.81036082 0.64242118 0.66291496
#  0.8994422  0.56813386 0.80006064 0.5551825  0.63412892 0.61366153
#  0.57191046 0.68144276 1.01136662 0.61998376 0.72866425 0.77156961
#  0.6108624  0.61668725 0.53048672 0.89553371 0.61193516 0.79435007
#  0.46129487 0.67219062 0.60686627 0.8920538  0.65505888 0.31408921
#  0.86369679 0.56043778 0.76977615 0.70780286 0.56646932 0.60853485
#  0.74869964 0.53373832 0.73313646 0.76091695 0.67940783 0.83278755
#  0.43392315 0.74996701 0.65689954 0.71752859 0.60376316 0.34675246
#  0.6879713  0.52418094 0.60844234 0.65928442 0.94086792 0.57786636
#  0.82424835 0.58901469 0.66165405 0.68646991 0.70292628 0.71364169
#  0.65496021 0.7819025  0.67678365 0.620065   0.70492387 0.61773325
#  0.73130711 0.63772496 0.57049292 0.55909834 0.59056053 0.64257052
#  0.91715964 0.77418239 0.83352501 0.66945864 0.66222481 0.60601055
#  0.66756436 0.4531535  0.65514486 0.92877536 0.60392748 0.91185277
#  0.70167393 0.75694429 0.66500923 0.89410422 0.70042526 0.51097354
#  0.99650873 0.61006587 0.67962798 0.79778468 0.59368419 0.95865148
#  0.66021723 0.55523598 0.6196814  0.91183848 0.57483459]

# 예측한 OPS를 csv파일로 저장.
pd.DataFrame({'batter_id': test['batter_id'], 'batter_name': test['batter_name'],
              'OPS': pred}).to_csv("submission01.csv", index=False)


# Random Forest Model
# 다수의 결정 트리들을 학습하는 앙상블 모델. 하나의 트리는 계층 구조로 이루어진 노드들과 엣지의 집합.  트리에서는 모든 모드가 input만을 가진다.
# train 단계에서는 종단 노드(터미널 노드)에 대한 매개변수와 내부노드 관련 분할 함수의 매개변수를 최적화.
# 데이터 포인트 v의 훈련 집합 S0 및 실제 데이터 레이블(ground truth label)이 주어졌을 때, 트리의 매개변수는 정의한 목적 함수를 최소화 하도록 선택된다.
# 트리의 성장을 언제 멈출지 결정하기 위해 미리 정의된 여러가지 멈춤 조건이 적용된다.
# T개의 트리로 구성된 하나의 포레스트의 경우, 일반적으로 훈련 과정은 각 트리에 대해 독립적으로 T번 반복된다.
# 랜덤 트리 또는 포레스트에서 주목할 사실은 랜덤성(randomness)이 오직 훈련 과정에만 존재한다는 것이다.
# 트리가 형성된 후 고정되어 있다면 테스트 단계에서는 완전히 결정론적인 특성을 보인다.
# 즉, 동일한 입력 데이터에 대해 항상 동일한 결과를 낸다.