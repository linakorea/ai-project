import pandas as pd
import os
from datetime import datetime, timedelta
from prophet import Prophet
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys

class SalesPredictor:
    # 클래스 상수 정의
    PEAK_HOURS = [14, 16]
    BUSINESS_START_HOUR = 8
    BUSINESS_END_HOUR = 23
    
    def __init__(self, data_dir: str, target_sales: int = 24000):
        self.data_dir = Path(data_dir)
        self.target_sales = target_sales
        self.model_weekday: Optional[Prophet] = None
        self.model_weekend: Optional[Prophet] = None
        self.holidays: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None
        self.total_actual_may: Optional[int] = None

    def load_data(self) -> None:
        """데이터 로드 및 전처리"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        if not files:
            raise FileNotFoundError(f"데이터 파일(.txt)을 찾을 수 없습니다: {self.data_dir}")

        dfs = []
        for file in files:
            file_path = self.data_dir / file
            try:
                df = pd.read_csv(file_path, sep='\t')
                df['일자'] = pd.to_datetime(df['일자'])
                dfs.append(df)
            except Exception as e:
                print(f"파일 '{file}' 처리 중 오류 발생: {str(e)}")
                continue

        if not dfs:
            raise ValueError("처리 가능한 데이터 파일이 없습니다.")

        self.data = pd.concat(dfs, ignore_index=True)

        # 데이터 전처리
        self.data['datetime'] = self.data['일자'] + pd.to_timedelta(self.data['시간대'], unit='h')
        self.data = self.data.sort_values('datetime')

        # 특징 추가
        self._add_features()

    def _add_features(self) -> None:
        """데이터 특징 추가"""
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['dayofweek'] = self.data['datetime'].dt.dayofweek
        self.data['month'] = self.data['datetime'].dt.month
        self.data['is_holiday'] = self.data['datetime'].dt.date.isin(self.holidays['ds'].dt.date if self.holidays is not None else []).astype(int)
        self.data['is_peak_hour'] = self.data['hour'].apply(lambda x: 1 if x in self.PEAK_HOURS else 0)
        self.data['is_weekend'] = self.data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        self.data['week_of_month'] = ((self.data['datetime'].dt.day - 1) // 7) + 1
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)

    def load_holidays(self) -> None:
        """공휴일 데이터 로드"""
        holidays_file = self.data_dir / "holidays.json"
        try:
            with open(holidays_file) as f:
                holidays_data = json.load(f)
        except FileNotFoundError:
            print(f"경고: 공휴일 데이터 파일을 찾을 수 없습니다: {holidays_file}")
            holidays_data = {}
        except json.JSONDecodeError:
            print(f"경고: 공휴일 데이터 파일 형식이 잘못되었습니다: {holidays_file}")
            holidays_data = {}

        self.holidays = pd.DataFrame({
            'holiday': [name for name in holidays_data.values()],
            'ds': pd.to_datetime(list(holidays_data.keys())),
            'lower_window': 0,
            'upper_window': 1
        }) if holidays_data else pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    def train(self) -> None:
        """주중/주말 모델 학습"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")

        # 1~5월 데이터 추출
        data_jan_may = self.data[(self.data['datetime'].dt.month >= 1) & (self.data['datetime'].dt.month <= 5)]

        # 주중/주말 데이터 분리
        weekday_data_jan_may = data_jan_may[data_jan_may['is_weekend'] == 0]
        weekend_data_jan_may = data_jan_may[data_jan_may['is_weekend'] == 1]

        # 학습 데이터 준비
        prophet_columns = ['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 
                         'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']
        prophet_weekday = weekday_data_jan_may[prophet_columns].rename(columns={'datetime': 'ds', '건수': 'y'})
        prophet_weekend = weekend_data_jan_may[prophet_columns].rename(columns={'datetime': 'ds', '건수': 'y'})

        # 모델 설정
        model_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays': self.holidays,
            'changepoint_prior_scale': 0.05,
            'holidays_prior_scale': 10
        }

        # 주중 모델 학습
        self.model_weekday = self._train_model(prophet_weekday, **model_params)

        # 주말 모델 학습
        self.model_weekend = self._train_model(prophet_weekend, **model_params)

    def _train_model(self, data: pd.DataFrame, **kwargs) -> Prophet:
        """Prophet 모델 학습을 위한 헬퍼 메서드"""
        model = Prophet(**kwargs)
        model.add_seasonality(name='hourly', period=1, fourier_order=15)
        
        # 리그레서 추가
        regressors = ['hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 
                     'week_of_month', 'sin_hour', 'cos_hour']
        for regressor in regressors:
            model.add_regressor(regressor)
            
        model.fit(data)
        return model

    def calculate_actual_may(self) -> None:
        """5월 1일~20일 실제 청약 건수 계산"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            
        may_actual = self.data[
            (self.data['datetime'].dt.year == 2025) & 
            (self.data['datetime'].dt.month == 5) & 
            (self.data['datetime'].dt.day <= 20)
        ]
        self.total_actual_may = may_actual['건수'].sum()

    def _prepare_future_df(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """예측을 위한 미래 데이터프레임 준비"""
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=start_date.replace(hour=self.BUSINESS_START_HOUR),
                end=end_date.replace(hour=self.BUSINESS_END_HOUR),
                freq='h'
            ),
        })
        
        # 특징 추가
        future['hour'] = future['ds'].dt.hour
        future['dayofweek'] = future['ds'].dt.dayofweek
        future['month'] = future['ds'].dt.month
        future['is_holiday'] = future['ds'].dt.date.isin(self.holidays['ds'].dt.date if self.holidays is not None else []).astype(int)
        future['is_peak_hour'] = future['hour'].apply(lambda x: 1 if x in self.PEAK_HOURS else 0)
        future['is_weekend'] = future['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future['week_of_month'] = ((future['ds'].dt.day - 1) // 7) + 1
        future['sin_hour'] = np.sin(2 * np.pi * future['hour'] / 24)
        future['cos_hour'] = np.cos(2 * np.pi * future['hour'] / 24)
        
        return future

    def predict(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """지정된 기간 동안 예측 수행"""
        if self.model_weekday is None or self.model_weekend is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")

        future = self._prepare_future_df(start_date, end_date)

        # 일별 예측 수행
        daily_predictions = []
        for date in pd.date_range(start=start_date.date(), end=end_date.date(), freq='D'):
            day_data = future[future['ds'].dt.date == date.date()]
            if day_data.empty:
                continue
                
            forecast_day = (self.model_weekend if day_data['is_weekend'].iloc[0] == 1 
                          else self.model_weekday).predict(day_data)
            
            daily_total = forecast_day['yhat'].round().astype(int).sum()
            daily_predictions.append({
                '날짜': date.strftime('%Y-%m-%d'),
                '예측값': daily_total
            })

        return pd.DataFrame(daily_predictions)

    def predict_today(self, now: datetime) -> pd.DataFrame:
        """오늘 시간대별 예측"""
        if self.model_weekday is None or self.model_weekend is None:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")
        if self.total_actual_may is None:
            raise ValueError("5월 실제 데이터가 계산되지 않았습니다. calculate_actual_may()를 먼저 실행하세요.")

        start_hour = now.replace(minute=0, second=0, microsecond=0)
        end_hour = now.replace(hour=self.BUSINESS_END_HOUR, minute=0, second=0, microsecond=0)

        future_today = self._prepare_future_df(start_hour, end_hour)
        
        # 주중/주말 여부에 따라 모델 선택
        forecast_today = (self.model_weekend if future_today['is_weekend'].iloc[0] == 1 
                        else self.model_weekday).predict(future_today)

        # 결과 데이터프레임 준비
        predicted_sales_today = pd.DataFrame({
            'ds': forecast_today['ds'],
            '예측값': forecast_today['yhat'].round().astype(int),
            '날짜': forecast_today['ds'].dt.strftime("%Y-%m-%d"),
            '시간대': forecast_today['ds'].dt.hour.apply(lambda x: f"{x}시")
        })
        
        predicted_sales_today['누적_예측값'] = predicted_sales_today['예측값'].cumsum()
        predicted_sales_today['누적_달성율(%)'] = (
            (self.total_actual_may + predicted_sales_today['누적_예측값']) / 
            self.target_sales * 100
        ).round(1)

        return predicted_sales_today

if __name__ == "__main__":
    try:
        # 예측 모델 실행
        predictor = SalesPredictor(data_dir="/Users/linakorea/Project/ai-project/predict_sales/data/")
        predictor.load_holidays()
        predictor.load_data()
        predictor.calculate_actual_may()
        predictor.train()

        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        predicted_sales_today = predictor.predict_today(now)

        # 오늘 시간대별 예측 출력
        print(f"\n{today_str} 시간대별 청약 건수 예측 ({now.hour}시~{predictor.BUSINESS_END_HOUR}시):")
        print(predicted_sales_today[['날짜', '시간대', '예측값', '누적_달성율(%)']].to_string(index=False))

        total_predicted_sales_today = predicted_sales_today['예측값'].sum()
        print(f"\n{today_str} 마감까지 예측 건수 ({now.hour}시~{predictor.BUSINESS_END_HOUR}시): {total_predicted_sales_today}건")

        total_may_sales = predictor.total_actual_may + total_predicted_sales_today
        achievement_rate_today = (total_may_sales / predictor.target_sales) * 100
        print(f"5월 목표 달성율 (1~20일 실제 + 오늘 예측): {achievement_rate_today:.1f}%")

        # 말일까지 일별 예측
        end_of_month = datetime(2025, 5, 31)
        daily_predictions = predictor.predict(start_date=now, end_date=end_of_month)

        cumulative_sales = predictor.total_actual_may
        for idx, row in daily_predictions.iterrows():
            cumulative_sales += row['예측값']
            daily_predictions.loc[idx, '누적_달성율(%)'] = (cumulative_sales / predictor.target_sales * 100).round(1)

        print("\n2025-05-21부터 말일까지 일별 청약 건수 예측:")
        print(daily_predictions.to_string(index=False, formatters={'예측값': lambda x: f'\t{x}'}))
        
        total_may_sales = predictor.total_actual_may + daily_predictions['예측값'].sum()
        achievement_rate_month = (total_may_sales / predictor.target_sales) * 100
        print(f"\n5월 전체 목표 달성율 (1~20일 실제 + 21~31일 예측): {achievement_rate_month:.1f}%")
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        sys.exit(1)