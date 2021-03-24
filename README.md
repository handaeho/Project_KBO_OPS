# Project_KBO_OPS ~ By Han Dae Ho

2019 KBO타자 성적 예측

타자의 OPS(On base Plus Slugging, 출루율 + 장타율)을 예측해보자.
OPS => ‘얼마나 자주 루상에 나가느냐’ 와 ‘얼마나 많은 루를 나가느냐’.
따라서 출루율이나 장타율과 달리 OPS는 두개를 한꺼번에 보여줄수 있다는 면에서 득점과 가장 밀접한 지표라고 볼 수 있다.

Host : ①데이콘 ②서울대학교 통계연구소 ③수원대 DS & ML 센터 ④한국야구학회 ⑤기타

Background : 2016년 관중수가 800만명을 돌파한 프로야구는 명실공히 한국 프로스포츠 최고의 인기 종목.
프로야구의 인기와 더불어 데이터 분석에 대한 인식이 높아짐에 따라 국내 여러 구단에서 데이터 사이언스 역할의 수요가 늘고 있다.
특히 야구에서는 특정 선수의 성적 변동성이 해마다 매우 크기 때문에 내년 성적을 예측하기 까다로운 부분이 많다.
정말 못 할 것이라고 생각했던 선수도 막상 내년에는 잘하고, 많은 지표가 리그 상위권이었던 선수가 내년에는 그렇지 않은 경우가 많다.

- Evaluation : WeightedRootMeanSquaredError(WRSME)
- 선수들의 타수를 가중치로 둔 RMSE. 예측 공식은 WRSME = sqrt(∑{(실제값 - 예측값)^2 * 타수} / ∑타수)

[Files]
  - ① Regular_Season_Batter.csv : KBO에서 활약한 타자들의 역대 정규시즌 성적을 포함하여 몸무게, 키 ,생년월일 등의 기본정보
  - ② Regular_Season_Batter_Day_by_Day.csv: KBO에서 활약한 타자들의 일자 별 정규시즌 성적
  - ③ Pre_Season_Batter.csv : KBO에서 활약한 타자들의 역대 시범경기(정규시즌 직전에 여는 연습경기) 성적
  - ④ submission.csv : 참가자들이 예측해야 할 타자의 이름과 아이디 목록
