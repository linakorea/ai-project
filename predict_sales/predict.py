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
        self.current_month_actual = None
        self.current_date = datetime.now()

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
        self.data['is_holiday'] = self.data['datetime'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
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
                holidays_data = json.load(f)
            self.holidays = pd.DataFrame({
                'holiday': [name for name in holidays_data.values()],
                'ds': pd.to_datetime(list(holidays_data.keys())),
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
        """현재 월의 실제 청약 건수 계산 (현재 날짜까지)"""
        current_year = self.current_date.year
        current_month = self.current_date.month
        current_day = self.current_date.day
        
        # 현재 월의 1일부터 어제까지의 실제 데이터
        yesterday = self.current_date - timedelta(days=1)
        
        current_month_data = self.data[
            (self.data['datetime'].dt.year == current_year) & 
            (self.data['datetime'].dt.month == current_month) & 
            (self.data['datetime'].dt.date <= yesterday.date())
        ]
        
        self.current_month_actual = current_month_data['건수'].sum() if not current_month_data.empty else 0
        
        print(f"현재 월({current_month}월) 실제 청약 건수 ({yesterday.strftime('%Y-%m-%d')}까지): {self.current_month_actual}건")

    def get_actual_data_for_date(self, target_date):
        """특정 날짜의 실제 데이터가 있는지 확인하고 반환"""
        actual_data = self.data[
            (self.data['datetime'].dt.date == target_date.date())
        ]
        
        if not actual_data.empty:
            return actual_data['건수'].sum()
        return None

    def predict(self, start_date, end_date, today_total_prediction=None):
        """지정된 기간 동안 예측 수행 (실제 데이터가 있으면 우선 사용)"""
        future = pd.DataFrame({
            'ds': pd.date_range(start=start_date.replace(hour=8), end=end_date.replace(hour=23), freq='h'),
        })
        future['hour'] = future['ds'].dt.hour
        future['dayofweek'] = future['ds'].dt.dayofweek
        future['month'] = future['ds'].dt.month
        future['is_holiday'] = future['ds'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
        future['is_peak_hour'] = future['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future['is_weekend'] = future['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future['week_of_month'] = ((future['ds'].dt.day - 1) // 7) + 1
        future['sin_hour'] = np.sin(2 * np.pi * future['hour'] / 24)
        future['cos_hour'] = np.cos(2 * np.pi * future['hour'] / 24)

        # 일별 예측 수행
        daily_predictions = []
        today_date = self.current_date.date()
        
        for date in pd.date_range(start=start_date.date(), end=end_date.date(), freq='D'):
            # 오늘 날짜인 경우 시간대별 예측 결과 사용
            if date.date() == today_date and today_total_prediction is not None:
                daily_predictions.append({
                    '날짜': date.strftime('%Y-%m-%d'),
                    '예측값': today_total_prediction,
                    '데이터타입': '예측(오늘)'
                })
                continue
                
            # 실제 데이터가 있는지 먼저 확인
            actual_sales = self.get_actual_data_for_date(date)
            
            if actual_sales is not None:
                # 실제 데이터가 있으면 사용
                daily_predictions.append({
                    '날짜': date.strftime('%Y-%m-%d'),
                    '예측값': actual_sales,
                    '데이터타입': '실제'
                })
            else:
                # 실제 데이터가 없으면 예측
                day_data = future[future['ds'].dt.date == date.date()]
                if day_data.empty:
                    continue
                
                if day_data['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
                    forecast_day = self.model_weekend.predict(day_data)
                elif day_data['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
                    forecast_day = self.model_weekday.predict(day_data)
                else:
                    # 모델이 없으면 평균값 사용
                    avg_daily = self.data.groupby(self.data['datetime'].dt.date)['건수'].sum().mean()
                    daily_total = int(avg_daily)
                    daily_predictions.append({
                        '날짜': date.strftime('%Y-%m-%d'),
                        '예측값': daily_total,
                        '데이터타입': '예측(평균)'
                    })
                    continue
                    
                daily_total = forecast_day['yhat'].round().astype(int).sum()
                daily_predictions.append({
                    '날짜': date.strftime('%Y-%m-%d'),
                    '예측값': daily_total,
                    '데이터타입': '예측'
                })

        return pd.DataFrame(daily_predictions)

    def predict_today(self, target_time=None):
        """오늘 시간대별 예측 (target_time 이후부터)"""
        if target_time is None:
            target_time = self.current_date
            
        today = target_time.date()
        
        # 오늘의 실제 데이터가 있는지 확인 (현재 시간 이전)
        today_actual_data = self.data[
            (self.data['datetime'].dt.date == today) & 
            (self.data['datetime'].dt.hour < target_time.hour)
        ]
        
        actual_sales_so_far = today_actual_data['건수'].sum() if not today_actual_data.empty else 0
        
        # 현재 시간부터 23시까지 예측
        start_hour = target_time.replace(minute=0, second=0, microsecond=0)
        end_hour = target_time.replace(hour=23, minute=0, second=0, microsecond=0)

        future_today = pd.DataFrame({
            'ds': pd.date_range(start=start_hour, end=end_hour, freq='h'),
        })
        
        if future_today.empty:
            return pd.DataFrame()
            
        future_today['hour'] = future_today['ds'].dt.hour
        future_today['dayofweek'] = future_today['ds'].dt.dayofweek
        future_today['month'] = future_today['ds'].dt.month
        future_today['is_holiday'] = future_today['ds'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
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
            # 모델이 없으면 시간대별 평균 사용
            hourly_avg = self.data.groupby(self.data['datetime'].dt.hour)['건수'].mean()
            forecast_today = pd.DataFrame({
                'ds': future_today['ds'],
                'yhat': future_today['hour'].map(hourly_avg).fillna(0)
            })

        predicted_sales_today = forecast_today[['ds', 'yhat']].copy()
        predicted_sales_today.loc[:, '예측값'] = predicted_sales_today['yhat'].round().astype(int)
        predicted_sales_today.loc[:, '날짜'] = predicted_sales_today['ds'].dt.strftime("%Y-%m-%d")
        predicted_sales_today.loc[:, '시간대'] = predicted_sales_today['ds'].dt.hour.apply(lambda x: f"{x}시")
        predicted_sales_today.loc[:, '누적_예측값'] = predicted_sales_today['예측값'].cumsum()
        predicted_sales_today.loc[:, '누적_건수'] = self.current_month_actual + actual_sales_so_far + predicted_sales_today['누적_예측값']
        predicted_sales_today.loc[:, '누적_달성율(%)'] = (predicted_sales_today['누적_건수'] / self.target_sales * 100).round(1)

        return predicted_sales_today

if __name__ == "__main__":
    # 예측 모드 실행
    predictor = SalesPredictor(data_dir="/Users/linakorea/Project/ai-project/predict_sales/data/")
    predictor.load_holidays()
    predictor.load_data()
    predictor.calculate_current_month_actual()
    predictor.train()

    # 현재 시간 사용
    now = predictor.current_date
    today_str = now.strftime("%Y-%m-%d")
    
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 오늘 시간대별 예측
    predicted_sales_today = predictor.predict_today(now)
    
    if not predicted_sales_today.empty:
        print(f"\n{today_str} 시간대별 청약 건수 예측 ({now.hour}시~23시):")
        print(predicted_sales_today[['날짜', '시간대', '예측값', '누적_건수', '누적_달성율(%)']].to_string(index=False))

        total_predicted_sales_today = predicted_sales_today['예측값'].sum()
        print(f"\n{today_str} 마감까지 예측 건수 ({now.hour}시~23시): {total_predicted_sales_today}건")

        total_current_month_sales = predictor.current_month_actual + total_predicted_sales_today
        achievement_rate_today = (total_current_month_sales / predictor.target_sales) * 100
        print(f"{now.month}월 목표 달성율 (실제 + 오늘 예측): {achievement_rate_today:.1f}%")

    # 이번 달 말일까지 일별 예측
    from calendar import monthrange
    
    current_year = now.year
    current_month = now.month
    last_day = monthrange(current_year, current_month)[1]
    end_of_month = datetime(current_year, current_month, last_day)
    
    # 오늘의 총 예측값 계산 (실제 + 예측)
    today_actual = predictor.get_actual_data_for_date(now)
    if today_actual is None:
        today_actual = 0
    
    today_total_prediction = today_actual + (total_predicted_sales_today if not predicted_sales_today.empty else 0)
    
    daily_predictions = predictor.predict(start_date=now, end_date=end_of_month, today_total_prediction=today_total_prediction)

    if not daily_predictions.empty:
        cumulative_sales = predictor.current_month_actual
        cumulative_count_list = []
        achievement_rate_list = []

        for idx, row in daily_predictions.iterrows():
            # 오늘 날짜인 경우 실제 누적값을 사용
            if row['날짜'] == today_str and not predicted_sales_today.empty:
                final_cumulative = predicted_sales_today['누적_건수'].iloc[-1]  # 시간대별 예측의 마지막 누적값
                cumulative_count_list.append(final_cumulative)
                achievement_rate_list.append((final_cumulative / predictor.target_sales * 100).round(1))
            else:
                cumulative_sales += row['예측값']
                cumulative_count_list.append(cumulative_sales)
                achievement_rate_list.append((cumulative_sales / predictor.target_sales * 100).round(1))

        daily_predictions['누적_건수'] = cumulative_count_list
        daily_predictions['누적_달성율(%)'] = achievement_rate_list

        print(f"\n{now.strftime('%Y-%m-%d')}부터 {current_month}월 말일까지 일별 청약 건수 예측:")
        print(daily_predictions[['날짜', '예측값', '데이터타입', '누적_건수', '누적_달성율(%)']].to_string(index=False))
        
        # 오늘의 누적값을 기준으로 월말 예측 재계산
        if not predicted_sales_today.empty:
            today_final_cumulative = predicted_sales_today['누적_건수'].iloc[-1]
            remaining_days_prediction = daily_predictions[daily_predictions['날짜'] != today_str]['예측값'].sum()
            total_month_sales_overall = today_final_cumulative + remaining_days_prediction
        else:
            total_month_sales_overall = predictor.current_month_actual + daily_predictions['예측값'].sum()
            
        achievement_rate_month_overall = (total_month_sales_overall / predictor.target_sales) * 100
        print(f"\n{current_month}월 전체 목표 달성율 (실제 + 예측): {achievement_rate_month_overall:.1f}%")
    else:
        print("\n예측할 데이터가 없습니다.")