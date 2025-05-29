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
    def __init__(self, data_dir, target_sales=23549):
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
        # 데이터프레임의 'datetime' 열을 한국 시간으로 변환
        self.data['datetime'] = self.data['일자'] + pd.to_timedelta(self.data['시간대'], unit='h')
        self.data['datetime'] = self.data['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul') # UTC로 인식 후 KST로 변환
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
            st.warning(f"경로: '{holidays_file}'에서 공휴일 파일을 찾을 수 없습니다. 공휴일 없이 진행합니다.")
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
        ]

        if training_data.empty:
            training_data = self.data[
                (self.data['datetime'].dt.month >= 1) &
                (self.data['datetime'].dt.month <= 12)
            ]

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

        # `astimezone(pytz.utc).date()` 를 사용하여 비교 시 타임존 일관성 유지
        yesterday_utc = (self.current_date - timedelta(days=1)).astimezone(pytz.utc).date()


        current_month_data_until_yesterday = self.data[
            (self.data['datetime'].dt.year == current_year) &
            (self.data['datetime'].dt.month == current_month) &
            (self.data['datetime'].dt.tz_convert('Asia/Seoul').dt.date <= (self.current_date - timedelta(days=1)).date()) # KST 기준으로 어제까지
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
        # start_date와 end_date를 KST로 타임존 설정 후 naive datetime으로 변환 (Prophet 예측용)
        future_start_kst = start_date.replace(hour=8, minute=0, second=0, microsecond=0).astimezone(pytz.timezone('Asia/Seoul'))
        future_end_kst = end_date.replace(hour=23, minute=0, second=0, microsecond=0).astimezone(pytz.timezone('Asia/Seoul'))

        future = pd.DataFrame({
            'ds': pd.date_range(start=future_start_kst, end=future_end_kst, freq='h'),
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

        for date_iter in pd.date_range(start=start_date.date(), end=end_date.date(), freq='D'):
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
            target_time = self.current_date # KST 타임존 정보 포함

        today = target_time.date() # Naive date 객체

        # get_actual_data_for_date_and_hour 함수는 naive date를 받으므로, target_time.hour를 그대로 사용
        actual_sales_so_far_today = self.get_actual_data_for_date_and_hour(today, end_hour=target_time.hour)

        # Prophet 예측을 위한 future 데이터프레임 생성 시 naive datetime 사용
        start_hour_naive = target_time.replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
        end_hour_naive = target_time.replace(hour=23, minute=0, second=0, microsecond=0).replace(tzinfo=None)


        future_today = pd.DataFrame({
            'ds': pd.date_range(start=start_hour_naive, end=end_hour_naive, freq='h'),
        })


        if future_today.empty:
            total_actual_today = self.get_actual_data_for_date_and_hour(today, end_hour=24)
            predicted_sales_today = pd.DataFrame({
                'ds': [target_time.replace(hour=23, tzinfo=None)], # ds는 naive로 저장
                'yhat': [0],
                '예측값': [0],
                '날짜': [today.strftime("%Y-%m-%d")],
                '시간대': ["23시"],
                '누적_예측값': [0],
                '누적_건수': [self.current_month_actual + total_actual_today],
                '달성율(%)': [( (self.current_month_actual + total_actual_today) / self.target_sales * 100).round(1)]
            })
            return predicted_sales_today, total_actual_today

        future_today['hour'] = future_today['ds'].dt.hour
        future_today['dayofweek'] = future_today['ds'].dt.dayofweek
        future_today['month'] = future_today['ds'].dt.month

        holidays_for_prediction = self.holidays.copy()
        if not holidays_for_prediction.empty:
            holidays_for_prediction['ds'] = holidays_for_prediction['ds'].dt.tz_localize(None)


        if not holidays_for_prediction.empty:
            future_today['is_holiday'] = future_today['ds'].dt.date.isin(holidays_for_prediction['ds'].dt.date).astype(int)
        else:
            future_today['is_holiday'] = 0


        future_today['is_peak_hour'] = future_today['hour'].apply(lambda x: 1 if x in [14, 16] else 0)
        future_today['is_weekend'] = future_today['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        future_today['week_of_month'] = ((future_today['ds'].dt.day - 1) // 7) + 1
        future_today['sin_hour'] = np.sin(2 * np.pi * future_today['hour'] / 24)
        future_today['cos_hour'] = np.cos(2 * np.pi * future_today['hour'] / 24)

        if future_today['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
            forecast_today = self.model_weekend.predict(future_today)
        elif future_today['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
            forecast_today = self.model_weekday.predict(future_today)
        else:
            # 학습 데이터가 없는 경우, 임시로 평균값 사용
            # 실제 데이터의 'datetime'은 KST 타임존이므로, hour 비교는 문제가 없음.
            hourly_avg = self.data.groupby(self.data['datetime'].dt.hour)['건수'].mean().reset_index()
            hourly_avg.columns = ['hour', 'avg_sales']

            forecast_today = pd.merge(future_today, hourly_avg, on='hour', how='left')
            forecast_today['yhat'] = forecast_today['avg_sales'].fillna(0)


        predicted_sales_today_df = forecast_today[['ds', 'yhat']].copy()
        predicted_sales_today_df.loc[:, '예측값'] = predicted_sales_today_df['yhat'].round().astype(int)
        predicted_sales_today_df.loc[:, '날짜'] = predicted_sales_today_df['ds'].dt.strftime("%Y-%m-%d")
        predicted_sales_today_df.loc[:, '시간대'] = predicted_sales_today_df['ds'].dt.hour.apply(lambda x: f"{x}시")

        total_predicted_from_current_time = predicted_sales_today_df['예측값'].sum()
        today_full_day_estimated_sales = actual_sales_so_far_today + total_predicted_from_current_time

        cumulative_predicted_so_far = 0
        cumulative_list = []
        for index, row in predicted_sales_today_df.iterrows():
            cumulative_predicted_so_far += row['예측값']
            cumulative_list.append(self.current_month_actual + actual_sales_so_far_today + cumulative_predicted_so_far)

        predicted_sales_today_df['누적_건수'] = cumulative_list
        predicted_sales_today_df['달성율(%)'] = (predicted_sales_today_df['누적_건수'] / self.target_sales * 100).round(1)

        return predicted_sales_today_df, today_full_day_estimated_sales

# --- Streamlit 앱 시작 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정

# 커스텀 CSS (이전 디자인의 주요 스타일을 Streamlit에 적용)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        line-height: 1.6;
        font-size: 1em; /* 기본 폰트 크기 조정 */
    }

    /* Streamlit 기본 여백 제거 및 배경색 설정 */
    .stApp {
        background-color: #F8F8F8; /* 아주 연한 회색 배경 (흰색에 가까움) */
    }

    /* 컨테이너 스타일 (이전 HTML의 .container와 유사) */
    .st-emotion-cache-z5fcl4, .st-emotion-cache-1c7y2vl { /* Streamlit main content block & chart container */
        max-width: 1200px;
        margin: 20px auto;
        background-color: #FFFFFF; /* 흰색 배경 */
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        padding: 25px;
    }

    /* 제목 스타일 */
    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
        font-weight: 700;
    }

    h2 {
        color: #2c3e50;
        font-size: 1.6em; /* h2 폰트 크기 조정 */
        margin-bottom: 20px;
    }

    h3 {
        color: #2c3e50;
        font-size: 1.2em; /* h3 폰트 크기 조정 */
        margin-bottom: 15px;
    }

    /* 예측 요약 섹션 스타일 */
    .forecast-summary-st {
        text-align: center;
        margin-bottom: 40px;
        background-color: #e8f5fd; /* 연한 파란색 배경 */
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }

    .forecast-summary-st h2 {
        color: #2980b9;
        font-size: 1.6em; /* 요약 h2 폰트 크기 조정 */
        margin-bottom: 10px;
    }

    .forecast-summary-st p {
        font-size: 1.1em; /* 요약 p 폰트 크기 조정 */
        color: #444;
    }

    .forecast-summary-st .highlight {
        font-size: 2em; /* 요약 강조 폰트 크기 조정 */
        font-weight: 700;
        color: #2980b9;
        margin: 0 5px;
    }

    /* Streamlit Metric 위젯 스타일 (카드 형태) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF; /* 흰색 배경 */
        border-radius: 12px; /* 둥근 모서리 */
        padding: 20px; /* 패딩 증가 */
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); /* 그림자 강화 */
        text-align: center;
        margin-bottom: 20px; /* 카드 간 간격 */
        border: 1px solid #eee; /* 옅은 테두리 */
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1em; /* 라벨 폰트 크기 조정 */
        font-weight: 600;
        color: #555;
        margin-bottom: 8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5em !important; /* 값 폰트 크기 조정 */
        font-weight: 700;
        color: #3498db;
        margin-top: 5px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1em; /* 델타 폰트 크기 조정 */
        color: #28a745; /* 긍정적인 변화 */
        font-weight: 500;
        margin-top: 10px;
    }

    /* 테이블 스타일 (st.dataframe) */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px; /* 테이블 하단 여백 */
    }

    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
        background-color: #FFFFFF;
    }

    .stDataFrame th {
        background-color: #f2f2f2;
        font-weight: 600;
        color: #555;
        padding: 15px 20px;
        text-align: left;
        font-size: 1em; /* 테이블 헤더 폰트 크기 조정 */
    }

    .stDataFrame td {
        padding: 15px 20px;
        text-align: left;
        border-bottom: 1px solid #EEE;
        color: #333; /* 테이블 셀 텍스트 색상 명확히 */
        font-size: 0.95em; /* 테이블 셀 폰트 크기 조정 */
    }

    .stDataFrame tbody tr:last-child td {
        border-bottom: none;
    }

    .stDataFrame tbody tr:nth-child(even) {
        background-color: #FAFAFA;
    }

    .stDataFrame tbody tr:hover {
        background-color: #E6F7FF;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }

    /* 차트 컨테이너 스타일 */
    .st-emotion-cache-1c7y2vl { /* Streamlit chart container */
        padding: 20px;
        background-color: #FFF;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px; /* 차트 하단 여백 */
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
    value=23549,
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

    st.subheader(f"시간대별 청약 건수 예측 ({now.hour}시~23시):")
    # 시간대별 예측 데이터프레임 표시
    st.dataframe(predicted_sales_today_df[['날짜', '시간대', '예측값', '누적_건수', '달성율(%)']].style.format({
        '누적_건수': "{:,.0f}",
        '달성율(%)': "{:.1f}%"
    }), use_container_width=True, hide_index=True, height=(len(predicted_sales_today_df) + 1) * 35 + 3)

    # 시간대별 청약 건수 그래프 (곡선)
    st.subheader("시간대별 청약 건수 그래프")
    st.line_chart(predicted_sales_today_df.set_index('시간대')['예측값'])
else:
    st.write("오늘 예측 데이터가 없습니다.")

st.markdown("---")

# --- 이번 달 말일까지 일별 예측 섹션 (전체 통계) ---
st.header(f"🗓️ {now.month}월 말일까지 일별 청약 건수 예측")

current_year = now.year
current_month = now.month
last_day = monthrange(current_year, current_month)[1]
end_of_month = datetime(current_year, current_month, last_day)

daily_predictions = predictor.predict(start_date=now, end_date=end_of_month, today_full_day_estimated_sales=today_full_day_estimated_sales)

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
    }), use_container_width=True, hide_index=True, height=(len(daily_predictions) + 1) * 35 + 3)

    if not daily_predictions.empty:
        total_month_sales_overall = daily_predictions['누적_건수'].iloc[-1]
        achievement_rate_month_overall = (total_month_sales_overall / predictor.target_sales) * 100
        st.metric(label=f"**{current_month}월 전체 목표 달성율 (실제 + 예측)**", value=f"{achievement_rate_month_overall:.1f}%")
else:
    st.write("이번 달 남은 기간에 대한 예측 데이터가 없습니다.")

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