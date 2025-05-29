# app.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from prophet import Prophet
import json
import numpy as np
from calendar import monthrange
import pytz # pytz 라이브러리 추가

# --- SalesPredictor 클래스 정의 ---
class SalesPredictor:
    def __init__(self, data_dir, target_sales=23234):
        # data_dir은 외부에서 주입되는 완전한 절대 경로를 기대합니다.
        self.data_dir = data_dir
        self.target_sales = target_sales
        self.model_weekday = None
        self.model_weekend = None
        self.holidays = None
        self.data = None
        self.current_month_actual = None
        # 한국 시간(KST)으로 현재 시간 설정
        self.current_date = datetime.now(pytz.timezone('Asia/Seoul'))

    def load_data(self):
        """데이터 로드 및 전처리"""
        if not os.path.exists(self.data_dir):
            st.error(f"오류: 데이터 디렉토리 '{self.data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
            st.stop()

        try:
            dir_contents = os.listdir(self.data_dir)
            files = [f for f in dir_contents if f.endswith('.txt')]
        except Exception as e:
            st.error(f"'{self.data_dir}' 디렉토리 목록 읽기 중 오류 발생: {e}")
            st.stop()

        dfs = []
        if not files:
            st.error(f"오류: '{self.data_dir}'에서 '.txt' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
            raise ValueError("데이터 파일을 찾을 수 없습니다.")

        for file in files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep='\t')
                df['일자'] = pd.to_datetime(df['일자'])
                dfs.append(df)
            else:
                st.warning(f"경고: 파일 '{file_path}'가 존재하지 않습니다. 건너킵니다.")

        if dfs:
            self.data = pd.concat(dfs, ignore_index=True)
            self.data = self.data.sort_values('일자')
        else:
            st.error(f"오류: '{self.data_dir}'에서 유효한 데이터를 로드할 수 없습니다.")
            raise ValueError("데이터 파일을 로드할 수 없습니다.")

        # 데이터 전처리 및 특징 추가
        # 기존 데이터가 타임존 정보가 없는 날짜/시간이므로, 먼저 KST로 로컬라이즈
        self.data['datetime'] = self.data['일자'] + pd.to_timedelta(self.data['시간대'], unit='h')
        self.data['datetime'] = self.data['datetime'].dt.tz_localize('Asia/Seoul') # KST로 명시적 지정
        self.data = self.data.sort_values('datetime')
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['dayofweek'] = self.data['datetime'].dt.dayofweek
        self.data['month'] = self.data['datetime'].dt.month

        if self.holidays is not None and not self.holidays.empty:
            self.data['is_holiday'] = self.data['datetime'].dt.date.isin(self.holidays['ds'].dt.date).astype(int)
        else:
            self.data['is_holiday'] = 0
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
                'ds': pd.to_datetime(list(holidays_data.keys())).tz_localize('Asia/Seoul'), # 공휴일도 KST로 설정
                'lower_window': 0,
                'upper_window': 1
            })
        except FileNotFoundError:
            st.warning(f"경고: 경로: '{holidays_file}'에서 공휴일 파일을 찾을 수 없습니다. 공휴일 없이 진행합니다.")
            self.holidays = pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])
        except json.JSONDecodeError:
            st.error(f"오류: 공휴일 파일 '{holidays_file}'이 유효한 JSON 형식이 아닙니다.")
            self.holidays = pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

    def train(self):
        """주중/주말 모델 학습"""
        current_year = self.current_date.year
        current_month = self.current_date.month

        training_data = self.data[
            (self.data['datetime'].dt.year == current_year) &
            (self.data['datetime'].dt.month <= current_month)
        ].copy() # SettingWithCopyWarning 방지

        weekday_data = training_data[training_data['is_weekend'] == 0].copy()
        weekend_data = training_data[training_data['is_weekend'] == 1].copy()

        # Prophet 입력 데이터프레임의 'ds' 열에 타임존 정보 제거 (Prophet은 naive datetime을 선호)
        prophet_weekday = weekday_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})
        prophet_weekday['ds'] = prophet_weekday['ds'].dt.tz_localize(None) # 타임존 제거
        prophet_weekend = weekend_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})
        prophet_weekend['ds'] = prophet_weekend['ds'].dt.tz_localize(None) # 타임존 제거

        # 공휴일 데이터도 타임존 제거 (Prophet 학습 시 필요)
        holidays_for_prophet = self.holidays.copy()
        if not holidays_for_prophet.empty:
            holidays_for_prophet['ds'] = holidays_for_prophet['ds'].dt.tz_localize(None)


        if not prophet_weekday.empty:
            self.model_weekday = Prophet(
                yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                holidays=holidays_for_prophet if not holidays_for_prophet.empty else None,
                changepoint_prior_scale=0.05, holidays_prior_scale=10
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

        if not prophet_weekend.empty:
            self.model_weekend = Prophet(
                yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                holidays=holidays_for_prophet if not holidays_for_prophet.empty else None,
                changepoint_prior_scale=0.05, holidays_prior_scale=10
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

        yesterday_kst_date = (self.current_date - timedelta(days=1)).date()

        current_month_data_until_yesterday = self.data[
            (self.data['datetime'].dt.year == current_year) &
            (self.data['datetime'].dt.month == current_month) &
            (self.data['datetime'].dt.date <= yesterday_kst_date) # KST 기준으로 어제까지
        ]

        self.current_month_actual = current_month_data_until_yesterday['건수'].sum() if not current_month_data_until_yesterday.empty else 0


    def get_actual_data_for_date_and_hour(self, target_date, end_hour=23):
        """특정 날짜의 특정 시간까지의 실제 데이터 합계 반환"""
        # target_date는 naive date 객체이므로, data['datetime']의 date 부분을 비교
        # data['datetime']은 KST 타임존 정보를 가지고 있으므로, date만 비교
        actual_data = self.data[
            (self.data['datetime'].dt.date == target_date) &
            (self.data['datetime'].dt.hour < end_hour)
        ]
        return actual_data['건수'].sum() if not actual_data.empty else 0

    def predict(self, start_date, end_date, today_full_day_estimated_sales=None):
        """지정된 기간 동안 예측 수행 (실제 데이터가 있으면 우선 사용)"""
        # start_date와 end_date는 date 객체로 들어오므로, datetime 객체로 변환
        start_datetime = datetime(start_date.year, start_date.month, start_date.day, 8, 0, 0, 0)
        end_datetime = datetime(end_date.year, end_date.month, end_date.day, 23, 0, 0, 0)

        # KST로 타임존 설정 후 naive datetime으로 변환 (Prophet 예측용)
        start_date_kst = pytz.timezone('Asia/Seoul').localize(start_datetime)
        end_date_kst = pytz.timezone('Asia/Seoul').localize(end_datetime)

        future = pd.DataFrame({
            'ds': pd.date_range(start=start_date_kst, end=end_date_kst, freq='h'),
        })
        future['ds'] = future['ds'].dt.tz_localize(None) # Prophet 입력에 맞게 naive datetime으로 변환

        future['hour'] = future['ds'].dt.hour
        future['dayofweek'] = future['ds'].dt.dayofweek
        future['month'] = future['ds'].dt.month

        # 공휴일 데이터도 Prophet 예측 입력에 맞게 naive datetime으로 변환하여 사용
        holidays_for_prediction = self.holidays.copy()
        if not holidays_for_prediction.empty:
            holidays_for_prediction['ds'] = holidays_for_prediction['ds'].dt.tz_localize(None)

        if not holidays_for_prediction.empty:
            future['is_holiday'] = future['ds'].dt.date.isin(holidays_for_prediction['ds'].dt.date).astype(int)
        else:
            future['is_holiday'] = 0

        future['is_peak_hour'] = future['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future['is_weekend'] = future['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future['week_of_month'] = ((future['ds'].dt.day - 1) // 7) + 1
        future['sin_hour'] = np.sin(2 * np.pi * future['hour'] / 24)
        future['cos_hour'] = np.cos(2 * np.pi * future['hour'] / 24)

        daily_predictions = []
        # `current_date`는 KST 타임존 정보를 포함하고 있으므로 `.date()`로 비교
        today_date_obj = self.current_date.date()

        # start_date와 end_date는 이미 date 객체이므로 .date()를 다시 호출할 필요 없음
        for date_iter in pd.date_range(start=start_date, end=end_date, freq='D'):
            if date_iter.date() == today_date_obj and today_full_day_estimated_sales is not None:
                daily_predictions.append({
                    '날짜': date_iter.strftime('%Y-%m-%d'),
                    '예측값': today_full_day_estimated_sales,
                    '데이터타입': '예측(오늘 전체)'
                })
                continue

            if date_iter.date() < today_date_obj:
                actual_sales = self.get_actual_data_for_date_and_hour(date_iter.date(), end_hour=24)
                if actual_sales > 0:
                    daily_predictions.append({
                        '날짜': date_iter.strftime('%Y-%m-%d'),
                        '예측값': actual_sales,
                        '데이터타입': '실제'
                    })
                    continue

            # Prophet 예측을 위해 future 데이터프레임에서 naive datetime으로 필터링
            day_data = future[future['ds'].dt.date == date_iter.date()]
            if day_data.empty:
                continue

            if day_data['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
                forecast_day = self.model_weekend.predict(day_data)
            elif day_data['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
                forecast_day = self.model_weekday.predict(day_data)
            else:
                avg_daily = self.data.groupby(self.data['datetime'].dt.date)['건수'].sum().mean()
                daily_total = int(avg_daily)
                daily_predictions.append({
                    '날짜': date_iter.strftime('%Y-%m-%d'),
                    '예측값': daily_total,
                    '데이터타입': '예측(평균)'
                })
                continue

            # NaN 값을 0으로 처리 후 반올림하고 정수로 변환
            daily_total = np.nan_to_num(forecast_day['yhat']).round().astype(int).sum()
            daily_predictions.append({
                '날짜': date_iter.strftime('%Y-%m-%d'),
                '예측값': daily_total,
                '데이터타입': '예측'
            })

        return pd.DataFrame(daily_predictions)

    def predict_today(self, target_time=None):
        """오늘 시간대별 예측 (8시부터 23시까지, 실제 데이터 우선 사용)"""
        if target_time is None:
            target_time = self.current_date # KST 타임존 정보 포함

        today = target_time.date() # Naive date object for comparison

        # Define the full prediction range for today (8 AM to 11 PM)
        start_of_day_8am = datetime(today.year, today.month, today.day, 8, 0, 0)
        end_of_day_11pm = datetime(today.year, today.month, today.day, 23, 0, 0)

        # Localize these to KST and then remove timezone for Prophet
        start_of_day_8am_kst = pytz.timezone('Asia/Seoul').localize(start_of_day_8am)
        end_of_day_11pm_kst = pytz.timezone('Asia/Seoul').localize(end_of_day_11pm)

        future_today_full_range = pd.DataFrame({
            'ds': pd.date_range(start=start_of_day_8am_kst, end=end_of_day_11pm_kst, freq='h'),
        })
        future_today_full_range['ds'] = future_today_full_range['ds'].dt.tz_localize(None) # Prophet needs naive datetime

        # Add regressors for Prophet
        future_today_full_range['hour'] = future_today_full_range['ds'].dt.hour
        future_today_full_range['dayofweek'] = future_today_full_range['ds'].dt.dayofweek
        future_today_full_range['month'] = future_today_full_range['ds'].dt.month

        holidays_for_prediction = self.holidays.copy()
        if not holidays_for_prediction.empty:
            holidays_for_prediction['ds'] = holidays_for_prediction['ds'].dt.tz_localize(None)

        if not holidays_for_prediction.empty:
            future_today_full_range['is_holiday'] = future_today_full_range['ds'].dt.date.isin(holidays_for_prediction['ds'].dt.date).astype(int)
        else:
            future_today_full_range['is_holiday'] = 0

        future_today_full_range['is_peak_hour'] = future_today_full_range['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future_today_full_range['is_weekend'] = future_today_full_range['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future_today_full_range['week_of_month'] = ((future_today_full_range['ds'].dt.day - 1) // 7) + 1
        future_today_full_range['sin_hour'] = np.sin(2 * np.pi * future_today_full_range['hour'] / 24)
        future_today_full_range['cos_hour'] = np.cos(2 * np.pi * future_today_full_range['hour'] / 24)

        # Predict for the full range (8 AM to 11 PM)
        if future_today_full_range['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
            forecast_today_full_range = self.model_weekend.predict(future_today_full_range)
        elif future_today_full_range['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
            forecast_today_full_range = self.model_weekday.predict(future_today_full_range)
        else:
            # Fallback if no model is trained
            hourly_avg = self.data.groupby(self.data['datetime'].dt.hour)['건수'].mean().reset_index()
            hourly_avg.columns = ['hour', 'avg_sales']
            forecast_today_full_range = pd.merge(future_today_full_range, hourly_avg, on='hour', how='left')
            forecast_today_full_range['yhat'] = forecast_today_full_range['avg_sales'].fillna(0)

        # Get actual data for today, hour by hour
        actual_data_today = self.data[self.data['datetime'].dt.date == today].copy()
        actual_data_today['hour_only'] = actual_data_today['datetime'].dt.hour
        actual_sales_by_hour = actual_data_today.groupby('hour_only')['건수'].sum().to_dict()

        # Combine actual and predicted data
        combined_predictions = []
        current_cumulative_sales = self.current_month_actual # Start with actual sales up to yesterday

        for index, row in forecast_today_full_range.iterrows():
            hour = row['ds'].hour
            # NaN 값을 0으로 처리 후 반올림하고 정수로 변환
            predicted_value = int(np.nan_to_num(row['yhat']).round())
            value_to_use = predicted_value
            data_type = '예측'

            # Use actual data if available for past/current hours
            # 현재 시간까지는 실제 데이터가 있다면 사용하고, 없다면 예측 데이터를 사용
            if hour <= target_time.hour:
                if hour in actual_sales_by_hour:
                    value_to_use = actual_sales_by_hour[hour]
                    data_type = '실제'
                # else: # No actual data for a past/current hour, use prediction (default)
                #     data_type = '예측(실제없음)' # Optional: add a specific type for clarity

            current_cumulative_sales += value_to_use

            combined_predictions.append({
                'ds': row['ds'], # Add the 'ds' column here for charting
                '날짜': today.strftime("%Y-%m-%d"),
                '시간대': f"{hour}시",
                '예측값': value_to_use,
                '누적_건수': current_cumulative_sales,
                '달성율(%)': (current_cumulative_sales / self.target_sales * 100).round(1),
                '데이터타입': data_type
            })

        predicted_sales_today_df = pd.DataFrame(combined_predictions)
        # Calculate today's full estimated sales based on the combined actuals/predictions for today
        total_full_day_estimated_sales = predicted_sales_today_df['예측값'].sum()

        return predicted_sales_today_df, total_full_day_estimated_sales

# --- Streamlit 앱 시작 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정

# 전문적인 디자인 시스템 CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        /* LINA Primary Colors (inferred from PDF) */
        --lina-black: #1A1A1A;
        --lina-dark-blue: #1A437C;
        --lina-digital-blue: #3498DB; /* Primary blue for highlights, values */
        
        /* LINA Secondary Color (inferred from PDF) */
        --lina-yellow: #F39C12; /* Accent color */

        /* LINA Non-Chromatic Colors (inferred from PDF) */
        --lina-gray: #6C7A89;
        --lina-light-gray: #D0D3D4;
        --lina-pale-gray: #EAECEE; /* Very light background for elements */
        
        /* Derived Neutral Colors (for consistency with LINA palette) */
        --neutral-50: #FDFDFD; /* LINA Pale Gray inspired */
        --neutral-100: #F8F9FA; /* Slightly darker than 50, LINA Light Gray inspired */
        --neutral-200: #F0F2F5;
        --neutral-300: #E2E4E8;
        --neutral-400: #C4C8CC;
        --neutral-500: #A8AEB2;
        --neutral-600: #8C9498;
        --neutral-700: #70787C;
        --neutral-800: #545C60;
        --neutral-900: #384044;
        
        /* Text Colors */
        --text-primary: var(--neutral-900);
        --text-secondary: var(--neutral-700);
        --text-tertiary: var(--neutral-500);
        --text-inverse: #ffffff;
        
        /* Background Colors */
        --bg-primary: #ffffff;
        --bg-secondary: var(--neutral-50); /* Overall app background */
        --bg-tertiary: var(--neutral-100); /* For summary cards, etc. */
        --bg-card: #ffffff;
        --bg-overlay: rgba(15, 23, 42, 0.8);
        
        /* Border & Divider Colors */
        --border-primary: var(--neutral-200);
        --border-secondary: var(--neutral-300);
        --border-focus: var(--lina-digital-blue);
        
        /* Shadows */
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
        
        /* Spacing System */
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
        --spacing-2xl: 48px;
        --spacing-3xl: 64px;
        
        /* Border Radius */
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --radius-2xl: 24px;
        
        /* Typography */
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-base: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.25rem;
        --font-size-2xl: 1.5rem;
        --font-size-3xl: 1.875rem;
        --font-size-4xl: 2.25rem;
        --font-size-5xl: 3rem;
        
        --line-height-tight: 1.25;
        --line-height-normal: 1.5;
        --line-height-relaxed: 1.625;
    }

    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        line-height: var(--line-height-normal);
        font-size: var(--font-size-base);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background-color: var(--bg-secondary);
    }

    /* 메인 앱 컨테이너 */
    .stApp {
        background-color: var(--bg-secondary);
        min-height: 100vh;
        padding: var(--spacing-lg);
    }

    /* 메인 컨테이너 */
    .st-emotion-cache-z5fcl4, .st-emotion-cache-1c7y2vl {
        max-width: 1200px;
        margin: 0 auto;
        background-color: var(--bg-card);
        border-radius: var(--radius-xl);
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-lg);
        padding: var(--spacing-3xl) var(--spacing-2xl);
        margin-bottom: var(--spacing-2xl);
    }

    /* 타이포그래피 시스템 */
    h1 {
        font-size: var(--font-size-5xl);
        font-weight: 800;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: var(--spacing-3xl);
        line-height: var(--line-height-tight);
        letter-spacing: -0.025em;
        position: relative;
    }

    h1::after {
        content: '';
        position: absolute;
        bottom: -16px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, var(--lina-digital-blue), var(--lina-yellow)); /* LINA colors for gradient */
        border-radius: 2px;
    }

    h2 {
        font-size: var(--font-size-3xl);
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xl);
        line-height: var(--line-height-tight);
        border-bottom: 1px solid var(--border-primary);
        padding-bottom: var(--spacing-md);
    }

    h3 {
        font-size: var(--font-size-2xl);
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-lg);
        line-height: var(--line-height-tight);
        position: relative;
        padding-left: var(--spacing-lg);
    }

    h3::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 24px;
        background-color: var(--lina-digital-blue); /* LINA Digital Blue for accent */
        border-radius: 2px;
    }

    /* 예측 요약 카드 */
    .forecast-summary-st {
        text-align: center;
        margin-bottom: var(--spacing-3xl);
        background-color: var(--bg-tertiary);
        padding: var(--spacing-3xl) var(--spacing-2xl);
        border-radius: var(--radius-2xl);
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-md);
        position: relative;
    }

    .forecast-summary-st h2 {
        font-size: var(--font-size-4xl);
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: var(--spacing-lg);
        border-bottom: none;
        padding-bottom: 0;
    }

    .forecast-summary-st p {
        font-size: var(--font-size-lg);
        color: var(--text-secondary);
        margin-bottom: var(--spacing-md);
        font-weight: 500;
    }

    .forecast-summary-st .highlight {
        font-size: var(--font-size-5xl);
        font-weight: 900;
        color: var(--lina-digital-blue); /* LINA Digital Blue for highlight */
        margin: 0 var(--spacing-md);
        display: inline-block;
    }

    /* 메트릭 카드 */
    [data-testid="stMetric"] {
        background-color: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: var(--spacing-2xl) var(--spacing-xl);
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-sm);
        text-align: center;
        margin-bottom: var(--spacing-lg);
        transition: all 0.2s ease-in-out;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--border-focus);
    }

    [data-testid="stMetricLabel"] {
        font-size: var(--font-size-sm);
        font-weight: 600;
        color: var(--text-tertiary);
        margin-bottom: var(--spacing-md);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricValue"] {
        font-size: var(--font-size-4xl) !important;
        font-weight: 800 !important;
        color: var(--lina-digital-blue) !important; /* LINA Digital Blue for metric values */
        margin: var(--spacing-md) 0 !important;
        line-height: var(--line-height-tight) !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: var(--font-size-sm);
        font-weight: 500;
        padding: var(--spacing-xs) var(--spacing-md);
        background-color: var(--bg-tertiary);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-secondary);
        display: inline-block;
        margin-top: var(--spacing-md);
    }

    /* 테이블 스타일 */
    .stDataFrame {
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        margin-bottom: var(--spacing-2xl);
        border: 1px solid var(--border-primary);
        background-color: #FFFFFF !important; /* 전체 테이블 배경을 흰색으로 변경 (강제 적용) */
        /* overflow-x: hidden !important; */ /* 이 속성은 하위 요소에 의해 재정의될 수 있습니다. */
    }

    /* st.dataframe의 실제 테이블 컨테이너에 overflow-x: hidden을 강제 적용 */
    .stDataFrame > div:first-child > div:first-child {
        overflow-x: hidden !important;
    }

    .stDataFrame div[data-testid="stDataFrameResizable"] {
        overflow-x: hidden !important;
    }

    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'Inter', sans-serif;
        table-layout: fixed; /* 컬럼 너비 고정하여 내용이 넘치지 않도록 함 */
    }

    .stDataFrame th {
        background-color: #FFFFFF !important; /* 헤더를 흰색으로 (강제 적용) */
        color: #1A1A1A !important; /* 텍스트 색상 명시적 설정 */
        font-weight: 600;
        padding: var(--spacing-lg) var(--spacing-xl);
        text-align: left;
        font-size: var(--font-size-lg); /* 폰트 크기 증가 */
        letter-spacing: 0.025em;
        border-bottom: 1px solid var(--border-primary);
    }

    .stDataFrame td {
        padding: var(--spacing-md) var(--spacing-xl);
        text-align: left;
        border-bottom: 1px solid var(--border-primary);
        color: #1A1A1A !important; /* 텍스트 색상 명시적 설정 */
        font-size: var(--font-size-base); /* 폰트 크기 증가 */
        font-weight: 500;
        background-color: #FFFFFF !important; /* 모든 셀의 배경을 흰색으로 (강제 적용) */
    }

    .stDataFrame tbody tr:last-child td {
        border-bottom: none;
    }

    .stDataFrame tbody tr:nth-child(even) {
        background-color: #FFFFFF !important; /* 줄무늬를 흰색으로 (짝수 행, 강제 적용) */
    }

    .stDataFrame tbody tr:hover {
        background-color: var(--neutral-100); /* 호버 효과는 유지 */
        transition: background-color 0.2s ease;
    }

    /* 차트 컨테이너 */
    .st-emotion-cache-1c7y2vl {
        padding: 25px;
        background-color: var(--bg-card);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        margin-bottom: var(--spacing-2xl);
        border: 1px solid var(--border-primary);
    }

    /* 사이드바 */
    .st-emotion-cache-vk3305 {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-primary);
        box-shadow: var(--shadow-sm);
        padding: var(--spacing-2xl);
        border-radius: var(--radius-lg);
        margin: var(--spacing-lg);
    }

    .st-emotion-cache-vk3305 h2 {
        color: var(--text-primary);
        border-bottom: none;
        margin-bottom: 25px;
        font-size: 1.6em;
        font-weight: 700;
        padding-bottom: var(--spacing-md);
    }

    .st-emotion-cache-vk3305 label {
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: var(--spacing-md);
        font-size: var(--font-size-sm);
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* 입력 필드 */
    .stSelectbox > div > div {
        background-color: var(--bg-card);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        transition: all 0.2s ease-in-out;
    }

    .stSelectbox > div > div:focus-within {
        border-color: var(--border-focus);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    .stNumberInput > div > div > input {
        background-color: var(--bg-card);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        padding: var(--spacing-md);
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--border-focus);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, var(--lina-digital-blue) 0%, #3498DB 100%); /* LINA Digital Blue for buttons */
        color: var(--text-inverse);
        border: none;
        border-radius: var(--radius-md);
        padding: var(--spacing-md) var(--spacing-xl);
        font-weight: 600;
        font-size: var(--font-size-sm);
        transition: all 0.2s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        box-shadow: var(--shadow-sm);
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--lina-dark-blue) 0%, var(--lina-digital-blue) 100%); /* Darker LINA blue on hover */
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }

    /* 알림 및 상태 표시 */
    .stAlert {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-primary);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-lg);
    }

    .stSuccess {
        background-color: #f0fdf4;
        border-color: #22c55e;
        color: #166534;
    }

    .stWarning {
        background-color: #fffbeb;
        border-color: var(--lina-yellow); /* LINA Yellow for warnings */
        color: #92400e;
    }

    .stError {
        background-color: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }

    /* 구분선 */
    hr {
        border: none;
        height: 1px;
        background-color: var(--border-primary);
        margin: var(--spacing-3xl) 0;
    }

    /* 캡션 */
    .st-emotion-cache-10qj07y {
        text-align: center;
        color: var(--text-tertiary);
        font-size: var(--font-size-sm);
        margin-top: var(--spacing-2xl);
        font-style: italic;
    }

    /* 로딩 상태 */
    .stSpinner > div {
        border-color: var(--lina-digital-blue) !important; /* LINA Digital Blue for spinner */
        border-right-color: transparent !important;
    }

    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background-color: var(--bg-tertiary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: var(--neutral-400);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background-color: var(--neutral-500);
    }

    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .st-emotion-cache-z5fcl4, .st-emotion-cache-1c7y2vl {
            margin: var(--spacing-md);
            padding: var(--spacing-xl) var(--spacing-lg);
        }
        
        h1 {
            font-size: var(--font-size-4xl);
        }
        
        h2 {
            font-size: var(--font-size-2xl);
        }
        
        h3 {
            font-size: var(--font-size-xl);
        }
        
        [data-testid="stMetricValue"] {
            font-size: var(--font-size-3xl) !important;
        }
        
        .forecast-summary-st .highlight {
            font-size: var(--font-size-4xl);
        }
        
        .forecast-summary-st {
            padding: var(--spacing-xl) var(--spacing-lg);
        }
        
        .stDataFrame th,
        .stDataFrame td {
            padding: var(--spacing-sm) var(--spacing-md);
            font-size: var(--font-size-xs);
        }
    }

    @media (max-width: 480px) {
        .stApp {
            padding: var(--spacing-md);
        }
        
        .st-emotion-cache-z5fcl4, .st-emotion-cache-1c7y2vl {
            margin: var(--spacing-sm);
            padding: var(--spacing-lg) var(--spacing-sm);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 월별 청약 건수 예측 대시보드")
st.markdown("---")

# 목표 청약 건수 입력 (사이드바)
st.sidebar.header("설정")
target_sales_input = st.sidebar.number_input(
    "월 목표 청약 건수:",
    min_value=1000,
    max_value=100000,
    value=23234, # 변경된 목표값 반영
    step=100
)

# --- ★★★ 가장 중요한 부분: 데이터 디렉토리 절대 경로 지정 ★★★ ---
# Streamlit Cloud 환경을 위한 경로 설정
# GitHub 저장소의 루트에 `data/` 폴더가 있다고 가정
# 예: GitHub repo name이 'my-streamlit-app'인 경우
# fixed_data_dir = "/mount/src/my-streamlit-app/data/"
# 사용자께서 제공해주신 기존 app.py의 경로 설정 로직을 따릅니다.
fixed_data_dir = "/mount/src/ai-project/predict_sales/data/" # 기본 경로 설정

# 로컬 개발 환경을 위한 대체 경로 (Streamlit Cloud에서는 이 부분이 실행되지 않음)
if not os.path.exists(fixed_data_dir):
    # 로컬에서 실행 시 'data/' 폴더가 현재 스크립트와 같은 디렉토리에 있을 경우
    fixed_data_dir = 'data/'
    if not os.path.exists(fixed_data_dir):
        st.error(f"오류: 로컬 환경에서 데이터 디렉토리 '{fixed_data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        st.stop()


# 예측기 인스턴스 생성 (절대 경로 전달)
predictor = SalesPredictor(data_dir=fixed_data_dir, target_sales=target_sales_input)

# 데이터 로드 및 모델 학습
try:
    predictor.load_holidays()
    predictor.load_data()
    predictor.calculate_current_month_actual()
    predictor.train()
    # st.success("데이터 로드 및 모델 학습 완료!") # 로딩 성공 메시지는 주석 처리
except ValueError as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.warning("경로 및 파일 존재 여부를 다시 확인해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"예측기 초기화 또는 학습 중 알 수 없는 오류가 발생했습니다: {e}")
    st.stop()

# 현재 시간은 SalesPredictor 클래스 생성 시 이미 한국 시간으로 설정되어 있음
now = predictor.current_date
today_str = now.strftime("%Y-%m-%d")

st.write(f"현재 시간: **{now.strftime('%Y-%m-%d %H:%M:%S (KST)')}**") # KST 추가
st.markdown("---")

# --- 주요 정보 카드 (대시보드 상단) ---
predicted_sales_today_df, today_full_day_estimated_sales = predictor.predict_today(now)

# 오늘 마감까지의 누적 판매량 및 달성률 계산 (카드에 표시하기 위함)
today_23hr_cumulative_sales = 0
achievement_rate_today_23hr = 0
if not predicted_sales_today_df.empty:
    today_23hr_cumulative_sales = predicted_sales_today_df['누적_건수'].iloc[-1]
    achievement_rate_today_23hr = (today_23hr_cumulative_sales / predictor.target_sales) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"**{now.month}월 현재까지 실제 청약 건수**",
        value=f"{predictor.current_month_actual:,.0f}건"
    )
with col2:
    st.metric(
        label=f"**{today_str} 마감까지 예상 총 건수**",
        value=f"{today_full_day_estimated_sales:,.0f}건"
    )
with col3:
    st.metric(
        label=f"**{now.month}월 목표 달성율 (오늘 마감까지)**",
        value=f"{achievement_rate_today_23hr:.1f}%",
        delta=f"{predictor.target_sales - today_23hr_cumulative_sales:,.0f}건 남음"
    )

st.markdown("---")

# --- 오늘 시간대별 예측 섹션 ---
st.header(f"📅 {today_str} 청약 건수 예측")

if not predicted_sales_today_df.empty:
    # 예측 요약 섹션
    st.markdown(
        f"""
        <div class="forecast-summary-st">
            <h2>오늘의 예상 청약 요약</h2>
            <p>총 예상 청약 건수: <span class="highlight">{today_full_day_estimated_sales:,.0f}</span>건</p>
            <p>최다 예상 시간대: <span class="highlight">{predicted_sales_today_df['시간대'].iloc[predicted_sales_today_df['예측값'].idxmax()]}</span> (<span class="highlight">{predicted_sales_today_df['예측값'].max():,.0f}</span>건)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(f"시간대별 청약 건수 예측 (8시~23시):") # 문구 수정
    # 시간대별 예측 데이터프레임 표시
    st.dataframe(predicted_sales_today_df[['날짜', '시간대', '예측값', '누적_건수', '달성율(%)']].style.format({
        '누적_건수': "{:,.0f}",
        '달성율(%)': "{:.1f}%"
    }), use_container_width=True, hide_index=True, height=(len(predicted_sales_today_df) * 35 + 38)) # 높이 조정

    # 시간대별 청약 건수 그래프 (곡선)
    st.subheader("시간대별 청약 건수 그래프")
    # 그래프 X축 순서가 맞도록 '시간대'를 숫자로 변환 후 정렬하거나, 'ds'를 인덱스로 사용
    predicted_sales_today_df['시간_정렬'] = predicted_sales_today_df['ds'].dt.hour
    chart_data = predicted_sales_today_df.sort_values('시간_정렬').set_index('시간대')['예측값']
    st.line_chart(chart_data)

else:
    st.write("오늘 예측 데이터가 없습니다.")

st.markdown("---")

# --- 이번 달 말일까지 일별 예측 섹션 (전체 통계) ---
st.header(f"🗓️ {now.month}월 말일까지 일별 청약 건수 예측")

current_year = now.year
current_month = now.month
last_day = monthrange(current_year, current_month)[1]
# end_of_month는 naive datetime으로 생성하여 predict 함수에 전달
end_of_month = datetime(current_year, current_month, last_day, 23, 59, 59) # 시분초 포함

# predict 함수에 date 객체를 직접 전달하도록 수정
daily_predictions = predictor.predict(start_date=now.date(), end_date=end_of_month.date(), today_full_day_estimated_sales=today_full_day_estimated_sales)

# total_month_sales_overall과 achievement_month_overall을 초기화합니다.
total_month_sales_overall = predictor.current_month_actual # 기본값으로 현재 월 실제 데이터 사용
achievement_rate_month_overall = (total_month_sales_overall / predictor.target_sales * 100).round(1)

if not daily_predictions.empty:
    cumulative_sales = today_23hr_cumulative_sales if 'today_23hr_cumulative_sales' in locals() else predictor.current_month_actual

    cumulative_count_list = []
    achievement_rate_list = []

    for idx, row in daily_predictions.iterrows():
        if row['날짜'] == today_str:
            # 오늘 날짜는 이미 predict_today에서 누적된 값으로 시작
            cumulative_count_list.append(cumulative_sales)
        else:
            cumulative_sales += row['예측값']
            cumulative_count_list.append(cumulative_sales)

        achievement_rate_list.append((cumulative_sales / predictor.target_sales * 100).round(1))

    daily_predictions['누적_건수'] = cumulative_count_list
    daily_predictions['달성율(%)'] = achievement_rate_list

    st.dataframe(daily_predictions[['날짜', '예측값', '데이터타입', '누적_건수', '달성율(%)']].style.format({
        '예측값': "{:,.0f}",
        '누적_건수': "{:,.0f}",
        '달성율(%)': "{:.1f}%"
    }), use_container_width=True, hide_index=True, height=(len(daily_predictions) * 35 + 38)) # 높이 조정

    # if not daily_predictions.empty: # 이 조건문은 이제 불필요합니다.
    total_month_sales_overall = daily_predictions['누적_건수'].iloc[-1]
    achievement_rate_month_overall = (total_month_sales_overall / predictor.target_sales) * 100
    st.metric(label=f"**{current_month}월 전체 목표 달성율 (실제 + 예측)**", value=f"{achievement_rate_month_overall:.1f}%")
else:
    st.write("이번 달 남은 기간에 대한 예측 데이터가 없습니다.")
    st.metric(label=f"**{current_month}월 전체 목표 달성율 (실제 + 예측)**", value=f"{achievement_rate_month_overall:.1f}%") # 예측 데이터가 없을 때도 메트릭 표시

st.markdown("---")

# --- 분석 리포트 섹션 (가장 하단) ---
st.header("분석 리포트")
st.markdown("""
이곳에는 청약 데이터에 대한 심층적인 분석 리포트, 트렌드, 예측 모델의 정확도 등에 대한 내용이 표시될 예정입니다.
<ul>
    <li>요일별 청약 트렌드</li>
    <li>캠페인 효과 분석</li>
    <li>과거 데이터 기반의 예측 정확도</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by Streamlit and Prophet for sales prediction.")
