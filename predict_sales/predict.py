import pandas as pd
import os
from datetime import datetime, timedelta
from prophet import Prophet
import json
import numpy as np

class SalesPredictor:
    def __init__(self, data_dir, target_sales=23549):
        self.data_dir = data_dir
        self.target_sales = target_sales
        self.model_weekday = None
        self.model_weekend = None
        self.holidays = None
        self.data = None
        self.current_month_actual = None # 지난달까지의 실제 청약건수 + 이번달 어제까지의 실제 청약건수
        self.current_date = datetime.now() # 현재 시간

    def load_data(self):
        """데이터 로드 및 전처리"""
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        dfs = []
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep='\t')
                df['일자'] = pd.to_datetime(df['일자'])
                dfs.append(df)
        
        if dfs:
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("데이터 파일을 찾을 수 없습니다.")

        # 데이터 전처리
        self.data['datetime'] = self.data['일자'] + pd.to_timedelta(self.data['시간대'], unit='h')
        self.data = self.data.sort_values('datetime')

        # 특징 추가
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['dayofweek'] = self.data['datetime'].dt.dayofweek
        self.data['month'] = self.data['datetime'].dt.month
        
        if self.holidays is not None and not self.holidays.empty: # holidays가 None이 아니고 비어있지 않은 경우에만 is_holiday 특징 추가
            self.data['is_holiday'] = self.data['datetime'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
        else:
            self.data['is_holiday'] = 0 # 공휴일 데이터 없으면 모두 0
        self.data['is_peak_hour'] = self.data['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        self.data['is_weekend'] = self.data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        self.data['week_of_month'] = ((self.data['datetime'].dt.day - 1) // 7) + 1
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)

    def load_holidays(self):
        """공휴일 데이터 로드"""
        holidays_file = os.path.join(self.data_dir, "holidays.json")
        try:
            with open(holidays_file) as f:
                holidays_data = json.load(f) # holidays_data 변수에 로드
            self.holidays = pd.DataFrame({
                'holiday': [name for name in holidays_data.values()],
                'ds': pd.to_datetime(list(holidays_data.keys())), # holidays_data 사용
                'lower_window': 0,
                'upper_window': 1
            })
        except FileNotFoundError:
            print("공휴일 파일을 찾을 수 없습니다. 공휴일 없이 진행합니다.")
            self.holidays = pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    def train(self):
        """주중/주말 모델 학습"""
        # 현재 년도의 1월부터 현재 월까지의 데이터 추출 (학습용)
        current_year = self.current_date.year
        current_month = self.current_date.month
        
        training_data = self.data[
            (self.data['datetime'].dt.year == current_year) & 
            (self.data['datetime'].dt.month <= current_month)
        ]
        
        if training_data.empty:
            # 현재 년도 데이터가 없으면 이전 년도 데이터 사용
            training_data = self.data[
                (self.data['datetime'].dt.month >= 1) & 
                (self.data['datetime'].dt.month <= 12)
            ]

        # 주중 데이터 (월~금)
        weekday_data = training_data[training_data['is_weekend'] == 0]

        # 주말 데이터 (토~일)
        weekend_data = training_data[training_data['is_weekend'] == 1]

        # 주중 전용 학습 데이터
        prophet_weekday = weekday_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})

        # 주말 전용 학습 데이터
        prophet_weekend = weekend_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})

        # 주중 전용 모델 학습
        if not prophet_weekday.empty:
            self.model_weekday = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=self.holidays,
                changepoint_prior_scale=0.05,
                holidays_prior_scale=10
            )
            self.model_weekday.add_seasonality(name='hourly', period=1, fourier_order=15)
            self.model_weekday.add_regressor('hour')
            self.model_weekday.add_regressor('dayofweek')
            self.model_weekday.add_regressor('month')
            self.model_weekday.add_regressor('is_holiday')
            self.model_weekday.add_regressor('is_peak_hour')
            self.model_weekday.add_regressor('week_of_month')
            self.model_weekday.add_regressor('sin_hour')
            self.model_weekday.add_regressor('cos_hour')
            self.model_weekday.fit(prophet_weekday)

        # 주말 전용 모델 학습
        if not prophet_weekend.empty:
            self.model_weekend = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=self.holidays,
                changepoint_prior_scale=0.05,
                holidays_prior_scale=10
            )
            self.model_weekend.add_seasonality(name='hourly', period=1, fourier_order=15)
            self.model_weekend.add_regressor('hour')
            self.model_weekend.add_regressor('dayofweek')
            self.model_weekend.add_regressor('month')
            self.model_weekend.add_regressor('is_holiday')
            self.model_weekend.add_regressor('is_peak_hour')
            self.model_weekend.add_regressor('week_of_month')
            self.model_weekend.add_regressor('sin_hour')
            self.model_weekend.add_regressor('cos_hour')
            self.model_weekend.fit(prophet_weekend)

    def calculate_current_month_actual(self):
        """현재 월의 실제 청약 건수 계산 (현재 날짜의 어제까지)"""
        current_year = self.current_date.year
        current_month = self.current_date.month
        
        # 어제까지의 실제 데이터
        yesterday = self.current_date - timedelta(days=1)
        
        current_month_data_until_yesterday = self.data[
            (self.data['datetime'].dt.year == current_year) & 
            (self.data['datetime'].dt.month == current_month) & 
            (self.data['datetime'].dt.date <= yesterday.date())
        ]
        
        self.current_month_actual = current_month_data_until_yesterday['건수'].sum() if not current_month_data_until_yesterday.empty else 0
        
        print(f"현재 월({current_month}월) 실제 청약 건수 ({yesterday.strftime('%Y-%m-%d')}까지): {self.current_month_actual}건")

    def get_actual_data_for_date_and_hour(self, target_date, end_hour=23):
        """특정 날짜의 특정 시간까지의 실제 데이터 합계 반환"""
        # target_date는 이미 date 객체이므로 .date()를 다시 호출할 필요 없음
        actual_data = self.data[
            (self.data['datetime'].dt.date == target_date) & 
            (self.data['datetime'].dt.hour < end_hour)
        ]
        return actual_data['건수'].sum() if not actual_data.empty else 0

    def predict(self, start_date, end_date, today_full_day_estimated_sales=None):
        """지정된 기간 동안 예측 수행 (실제 데이터가 있으면 우선 사용)"""
        # start_date와 end_date가 datetime 객체로 들어오면, 시간 정보를 포함하여 범위 설정
        future = pd.DataFrame({
            'ds': pd.date_range(start=start_date.replace(hour=8), end=end_date.replace(hour=23), freq='h'),
        })
        future['hour'] = future['ds'].dt.hour
        future['dayofweek'] = future['ds'].dt.dayofweek
        future['month'] = future['ds'].dt.month
        
        if self.holidays is not None and not self.holidays.empty:
            future['is_holiday'] = future['ds'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
        else:
            future['is_holiday'] = 0

        future['is_peak_hour'] = future['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future['is_weekend'] = future['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future['week_of_month'] = ((future['ds'].dt.day - 1) // 7) + 1
        future['sin_hour'] = np.sin(2 * np.pi * future['hour'] / 24)
        future['cos_hour'] = np.cos(2 * np.pi * future['hour'] / 24)

        # 일별 예측 수행
        daily_predictions = []
        today_date_obj = self.current_date.date() # datetime.date 객체
        
        # pd.date_range는 Timestamp 객체를 반환하므로, 비교할 때 .date()를 사용
        for date_iter in pd.date_range(start=start_date.date(), end=end_date.date(), freq='D'):
            # 오늘 날짜인 경우, predict_today에서 계산된 오늘의 총 예상 건수를 사용
            if date_iter.date() == today_date_obj and today_full_day_estimated_sales is not None:
                daily_predictions.append({
                    '날짜': date_iter.strftime('%Y-%m-%d'),
                    '예측값': today_full_day_estimated_sales,
                    '데이터타입': '예측(오늘 전체)'
                })
                continue
                
            # 과거 날짜인 경우 실제 데이터 사용 (단, 현재 날짜 이전이어야 함)
            # 예측 기간에 현재 날짜 이전의 과거 날짜가 포함될 경우
            if date_iter.date() < today_date_obj: # .date() 추가
                actual_sales = self.get_actual_data_for_date_and_hour(date_iter.date(), end_hour=24) # .date() 추가
                if actual_sales > 0:
                    daily_predictions.append({
                        '날짜': date_iter.strftime('%Y-%m-%d'),
                        '예측값': actual_sales,
                        '데이터타입': '실제'
                    })
                    continue # 다음 날짜로 넘어감
            
            # 미래 날짜이거나 오늘 날짜인데 today_full_day_estimated_sales가 없는 경우 예측
            day_data = future[future['ds'].dt.date == date_iter.date()] # .date() 추가
            if day_data.empty:
                continue
            
            if day_data['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
                forecast_day = self.model_weekend.predict(day_data)
            elif day_data['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
                forecast_day = self.model_weekday.predict(day_data)
            else:
                # 모델이 없으면 평균값 사용 (학습 데이터가 충분하지 않을 때 대비)
                # 이 부분은 시간대별 예측이 아닌 일별 예측이므로, 일별 평균을 사용하는 것이 더 적절
                avg_daily = self.data.groupby(self.data['datetime'].dt.date)['건수'].sum().mean()
                daily_total = int(avg_daily)
                daily_predictions.append({
                    '날짜': date_iter.strftime('%Y-%m-%d'),
                    '예측값': daily_total,
                    '데이터타입': '예측(평균)'
                })
                continue
                
            daily_total = forecast_day['yhat'].round().astype(int).sum()
            daily_predictions.append({
                '날짜': date_iter.strftime('%Y-%m-%d'),
                '예측값': daily_total,
                '데이터타입': '예측'
            })

        return pd.DataFrame(daily_predictions)

    def predict_today(self, target_time=None):
        """오늘 시간대별 예측 (target_time 이후부터)"""
        if target_time is None:
            target_time = self.current_date
            
        today = target_time.date() # target_time은 datetime 객체이므로 .date()를 호출하여 date 객체로 변환
        
        # 오늘의 실제 데이터 (0시부터 현재 시간 직전까지)
        actual_sales_so_far_today = self.get_actual_data_for_date_and_hour(today, end_hour=target_time.hour)
        
        # 현재 시간부터 23시까지 예측
        start_hour = target_time.replace(minute=0, second=0, microsecond=0)
        end_hour = target_time.replace(hour=23, minute=0, second=0, microsecond=0)

        future_today = pd.DataFrame({
            'ds': pd.date_range(start=start_hour, end=end_hour, freq='h'),
        })
        
        # 예측할 시간대가 없는 경우 (예: 23시 이후에 실행)
        if future_today.empty:
            # 오늘의 총 실제 데이터 (0시부터 23시까지 모두)
            total_actual_today = self.get_actual_data_for_date_and_hour(today, end_hour=24) 
            predicted_sales_today = pd.DataFrame({
                'ds': [target_time.replace(hour=23)], # 23시 기준으로 한 줄 추가
                'yhat': [0], # 예측값은 0
                '예측값': [0],
                '날짜': [today.strftime("%Y-%m-%d")],
                '시간대': ["23시"],
                '누적_예측값': [0],
                '누적_건수': [self.current_month_actual + total_actual_today],
                '누적_달성율(%)': [( (self.current_month_actual + total_actual_today) / self.target_sales * 100).round(1)]
            })
            return predicted_sales_today, total_actual_today # 오늘 실제 + 0(예측)
            
        future_today['hour'] = future_today['ds'].dt.hour
        future_today['dayofweek'] = future_today['ds'].dt.dayofweek
        future_today['month'] = future_today['ds'].dt.month
        
        if self.holidays is not None and not self.holidays.empty:
            future_today['is_holiday'] = future_today['ds'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
        else:
            future_today['is_holiday'] = 0

        future_today['is_peak_hour'] = future_today['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future_today['is_weekend'] = future_today['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future_today['week_of_month'] = ((future_today['ds'].dt.day - 1) // 7) + 1
        future_today['sin_hour'] = np.sin(2 * np.pi * future_today['hour'] / 24)
        future_today['cos_hour'] = np.cos(2 * np.pi * future_today['hour'] / 24)

        # 주중/주말 여부에 따라 모델 선택
        if future_today['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
            forecast_today = self.model_weekend.predict(future_today)
        elif future_today['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
            forecast_today = self.model_weekday.predict(future_today)
        else:
            # 모델이 없으면 시간대별 평균 사용 (학습 데이터가 충분하지 않을 때 대비)
            # 여기서는 '건수'의 시간대별 평균을 사용하도록 변경
            hourly_avg = self.data.groupby(self.data['datetime'].dt.hour)['건수'].mean().reset_index()
            hourly_avg.columns = ['hour', 'avg_sales']
            
            forecast_today = pd.merge(future_today, hourly_avg, on='hour', how='left')
            forecast_today['yhat'] = forecast_today['avg_sales'].fillna(0) # 결측치는 0으로 채움

        predicted_sales_today_df = forecast_today[['ds', 'yhat']].copy()
        predicted_sales_today_df.loc[:, '예측값'] = predicted_sales_today_df['yhat'].round().astype(int)
        predicted_sales_today_df.loc[:, '날짜'] = predicted_sales_today_df['ds'].dt.strftime("%Y-%m-%d")
        predicted_sales_today_df.loc[:, '시간대'] = predicted_sales_today_df['ds'].dt.hour.apply(lambda x: f"{x}시")
        
        # 오늘 남은 시간의 예측 합계
        total_predicted_from_current_time = predicted_sales_today_df['예측값'].sum()
        
        # 오늘의 총 예상 판매량 = 오늘 실제 (현재시간까지) + 오늘 예측 (남은시간)
        today_full_day_estimated_sales = actual_sales_so_far_today + total_predicted_from_current_time

        # 시간대별 누적 계산 (월초부터 누적)
        cumulative_predicted_so_far = 0 # 현재 시간 이후 예측값의 누적합
        cumulative_list = []
        for index, row in predicted_sales_today_df.iterrows():
            cumulative_predicted_so_far += row['예측값']
            # 전체 누적 건수 = (어제까지 실제) + (오늘 실제: 현재시간까지) + (오늘 예측: 현재시간부터 해당 시간까지)
            cumulative_list.append(self.current_month_actual + actual_sales_so_far_today + cumulative_predicted_so_far)
            
        predicted_sales_today_df['누적_건수'] = cumulative_list
        predicted_sales_today_df['누적_달성율(%)'] = (predicted_sales_today_df['누적_건수'] / self.target_sales * 100).round(1)

        return predicted_sales_today_df, today_full_day_estimated_sales # 오늘 전체 예상 건수도 함께 반환

if __name__ == "__main__":
    # 예측 모드 실행
    # 실제 데이터가 있는 디렉토리로 변경해야 합니다.
    predictor = SalesPredictor(data_dir="/Users/linakorea/Project/ai-project/predict_sales/data/")
    predictor.load_holidays()
    predictor.load_data()
    predictor.calculate_current_month_actual()
    predictor.train()

    # 현재 시간 사용
    now = predictor.current_date
    today_str = now.strftime("%Y-%m-%d")
    
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 오늘 시간대별 예측 및 오늘의 총 예상 판매량 계산
    predicted_sales_today_df, today_full_day_estimated_sales = predictor.predict_today(now)
    
    if not predicted_sales_today_df.empty:
        print(f"\n{today_str} 시간대별 청약 건수 예측 ({now.hour}시~23시):")
        print(predicted_sales_today_df[['날짜', '시간대', '예측값', '누적_건수', '누적_달성율(%)']].to_string(index=False))

        # 23시 기준의 '오늘 마감까지의 달성률'
        today_23hr_cumulative_sales = predicted_sales_today_df['누적_건수'].iloc[-1]
        achievement_rate_today_23hr = (today_23hr_cumulative_sales / predictor.target_sales) * 100
        print(f"\n{today_str} 마감까지 예상 총 건수: {today_full_day_estimated_sales}건")
        print(f"{now.month}월 목표 달성율 (오늘 마감까지): {achievement_rate_today_23hr:.1f}%")

    # 이번 달 말일까지 일별 예측
    from calendar import monthrange
    
    current_year = now.year
    current_month = now.month
    last_day = monthrange(current_year, current_month)[1]
    end_of_month = datetime(current_year, current_month, last_day)
    
    # predict 함수에 오늘의 총 예상 판매량 전달
    daily_predictions = predictor.predict(start_date=now, end_date=end_of_month, today_full_day_estimated_sales=today_full_day_estimated_sales)

    if not daily_predictions.empty:
        # 누적 계산 시작점 (오늘 23시까지의 누적 건수)
        # predicted_sales_today_df가 비어있지 않다면, 23시 기준의 누적 건수 (today_23hr_cumulative_sales)부터 시작
        # 만약 predict_today_df가 비어있다면 (예: 23시 이후 실행), current_month_actual (어제까지 실제)부터 시작
        cumulative_sales = today_23hr_cumulative_sales if 'today_23hr_cumulative_sales' in locals() else predictor.current_month_actual

        cumulative_count_list = []
        achievement_rate_list = []

        for idx, row in daily_predictions.iterrows():
            if row['날짜'] == today_str:
                # 오늘 날짜는 이미 위에서 계산된 23시 기준의 누적 건수를 사용
                cumulative_count_list.append(cumulative_sales)
            else:
                cumulative_sales += row['예측값']
                cumulative_count_list.append(cumulative_sales)
            
            achievement_rate_list.append((cumulative_sales / predictor.target_sales * 100).round(1))

        daily_predictions['누적_건수'] = cumulative_count_list
        daily_predictions['누적_달성율(%)'] = achievement_rate_list

        print(f"\n{now.strftime('%Y-%m-%d')}부터 {current_month}월 말일까지 일별 청약 건수 예측:")
        print(daily_predictions[['날짜', '예측값', '데이터타입', '누적_건수', '누적_달성율(%)']].to_string(index=False))
        
        # 5월 전체 목표 달성률 (월말 최종 예측값)
        if not daily_predictions.empty:
            total_month_sales_overall = daily_predictions['누적_건수'].iloc[-1]
            achievement_rate_month_overall = (total_month_sales_overall / predictor.target_sales) * 100
            print(f"\n{current_month}월 전체 목표 달성율 (실제 + 예측): {achievement_rate_month_overall:.1f}%")
    else:
        print("\n예측할 데이터가 없습니다.")