import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from prophet import Prophet
import json
import numpy as np
from calendar import monthrange

# --- SalesPredictor 클래스 정의 ---
class SalesPredictor:
    def __init__(self, data_dir, target_sales=23549):
        # 이제 data_dir은 외부에서 주입되는 완전한 절대 경로를 기대합니다.
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
        # st.write(f"현재 작업 디렉토리 (os.getcwd()): `{os.getcwd()}`")
        # st.write(f"설정된 data_dir: `{self.data_dir}` (절대 경로로 간주)")

        if not os.path.exists(self.data_dir):
            st.error(f"오류: 데이터 디렉토리 '{self.data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
            st.stop() # 디렉토리가 없으면 앱 중단
        else:
            st.success(f"데이터 디렉토리 '{self.data_dir}' 존재 확인!")

        try:
            # os.listdir()에 절대 경로를 직접 전달
            dir_contents = os.listdir(self.data_dir)
            # st.write(f"'{self.data_dir}' 내용: {dir_contents}")
            files = [f for f in dir_contents if f.endswith('.txt')]
            # st.write(f"찾은 .txt 파일: {files}")

        except Exception as e:
            st.error(f"'{self.data_dir}' 디렉토리 목록 읽기 중 오류 발생: {e}")
            st.stop() # 오류 발생 시 앱 중단


        dfs = []
        if not files: # .txt 파일을 하나도 찾지 못했을 때
            st.error(f"오류: '{self.data_dir}'에서 '.txt' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
            raise ValueError(f"데이터 파일을 찾을 수 없습니다.")

        for file in files:
            file_path = os.path.join(self.data_dir, file) # data_dir이 이미 절대 경로
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep='\t')
                df['일자'] = pd.to_datetime(df['일자'])
                dfs.append(df)
            else:
                st.warning(f"경고: 파일 '{file_path}'가 존재하지 않습니다. 건너뜁니다.")

        if dfs:
            self.data = pd.concat(dfs, ignore_index=True)
            self.data = self.data.sort_values('일자') # 날짜 기준으로 정렬
        else:
            st.error(f"오류: '{self.data_dir}'에서 유효한 데이터를 로드할 수 없습니다.")
            raise ValueError("데이터 파일을 로드할 수 없습니다.")

        # 데이터 전처리 및 특징 추가 (기존과 동일)
        self.data['datetime'] = self.data['일자'] + pd.to_timedelta(self.data['시간대'], unit='h')
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
        # holidays_file도 절대 경로를 직접 사용하도록 변경
        holidays_file = os.path.join(self.data_dir, "holidays.json")
        # st.write(f"공휴일 파일 경로: `{holidays_file}`")
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

        weekday_data = training_data[training_data['is_weekend'] == 0]
        weekend_data = training_data[training_data['is_weekend'] == 1]

        prophet_weekday = weekday_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})
        prophet_weekend = weekend_data[['datetime', '건수', 'hour', 'dayofweek', 'month', 'is_holiday', 'is_peak_hour', 'week_of_month', 'sin_hour', 'cos_hour']].rename(columns={'datetime': 'ds', '건수': 'y'})

        if not prophet_weekday.empty:
            self.model_weekday = Prophet(
                yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                holidays=self.holidays, changepoint_prior_scale=0.05, holidays_prior_scale=10
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
                holidays=self.holidays, changepoint_prior_scale=0.05, holidays_prior_scale=10
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

        yesterday = self.current_date - timedelta(days=1)

        current_month_data_until_yesterday = self.data[
            (self.data['datetime'].dt.year == current_year) &
            (self.data['datetime'].dt.month == current_month) &
            (self.data['datetime'].dt.date <= yesterday.date())
        ]

        self.current_month_actual = current_month_data_until_yesterday['건수'].sum() if not current_month_data_until_yesterday.empty else 0
        st.info(f"현재 월({current_month}월) 실제 청약 건수 ({yesterday.strftime('%Y-%m-%d')}까지): {self.current_month_actual}건")

    def get_actual_data_for_date_and_hour(self, target_date, end_hour=23):
        """특정 날짜의 특정 시간까지의 실제 데이터 합계 반환"""
        actual_data = self.data[
            (self.data['datetime'].dt.date == target_date) &
            (self.data['datetime'].dt.hour < end_hour)
        ]
        return actual_data['건수'].sum() if not actual_data.empty else 0

    def predict(self, start_date, end_date, today_full_day_estimated_sales=None):
        """지정된 기간 동안 예측 수행 (실제 데이터가 있으면 우선 사용)"""
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

        daily_predictions = []
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
            target_time = self.current_date

        today = target_time.date()

        actual_sales_so_far_today = self.get_actual_data_for_date_and_hour(today, end_hour=target_time.hour)

        start_hour = target_time.replace(minute=0, second=0, microsecond=0)
        end_hour = target_time.replace(hour=23, minute=0, second=0, microsecond=0)

        future_today = pd.DataFrame({
            'ds': pd.date_range(start=start_hour, end=end_hour, freq='h'),
        })

        if future_today.empty:
            total_actual_today = self.get_actual_data_for_date_and_hour(today, end_hour=24)
            predicted_sales_today = pd.DataFrame({
                'ds': [target_time.replace(hour=23)],
                'yhat': [0],
                '예측값': [0],
                '날짜': [today.strftime("%Y-%m-%d")],
                '시간대': ["23시"],
                '누적_예측값': [0],
                '누적_건수': [self.current_month_actual + total_actual_today],
                '누적_달성율(%)': [( (self.current_month_actual + total_actual_today) / self.target_sales * 100).round(1)]
            })
            return predicted_sales_today, total_actual_today

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

        if future_today['is_weekend'].iloc[0] == 1 and self.model_weekend is not None:
            forecast_today = self.model_weekend.predict(future_today)
        elif future_today['is_weekend'].iloc[0] == 0 and self.model_weekday is not None:
            forecast_today = self.model_weekday.predict(future_today)
        else:
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
        predicted_sales_today_df['누적_달성율(%)'] = (predicted_sales_today_df['누적_건수'] / self.target_sales * 100).round(1)

        return predicted_sales_today_df, today_full_day_estimated_sales

# --- Streamlit 앱 시작 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정
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
# 이전 로그에서 `app.py` 경로가 `/mount/src/ai-project/predict_sales/app.py`였다면,
# `data` 폴더는 같은 `predict_sales` 폴더 안에 있을 것입니다.
# 따라서 아래 경로가 가장 유력합니다.
# GitHub 저장소 구조와 Streamlit Cloud 로그를 기반으로 이 경로를 정확히 입력하세요.
fixed_data_dir = "/mount/src/ai-project/predict_sales/data/"

# 만약 GitHub 저장소의 루트에 바로 `app.py`와 `data/`가 있다면:
# fixed_data_dir = "/mount/src/YOUR_GITHUB_REPOSITORY_NAME/data/"
# (여기서 YOUR_GITHUB_REPOSITORY_NAME은 여러분의 실제 GitHub 저장소 이름입니다.)
# 예: fixed_data_dir = "/mount/src/my-streamlit-app/data/"


# 예측기 인스턴스 생성 (절대 경로 전달)
predictor = SalesPredictor(data_dir=fixed_data_dir, target_sales=target_sales_input)

# 데이터 로드 및 모델 학습
try:
    predictor.load_holidays()
    predictor.load_data()
    predictor.calculate_current_month_actual()
    predictor.train()
    st.success("데이터 로드 및 모델 학습 완료!")
except ValueError as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.warning("경로 및 파일 존재 여부를 다시 확인해 주세요.")
    st.stop() # 오류 발생 시 앱 실행 중단
except Exception as e: # 다른 예상치 못한 오류에 대비
    st.error(f"예측기 초기화 또는 학습 중 알 수 없는 오류가 발생했습니다: {e}")
    st.stop()

now = predictor.current_date
today_str = now.strftime("%Y-%m-%d")

st.write(f"현재 시간: **{now.strftime('%Y-%m-%d %H:%M:%S')}**")
st.markdown("---")

# --- 오늘 시간대별 예측 ---
st.header(f"📅 {today_str} 청약 건수 예측")

predicted_sales_today_df, today_full_day_estimated_sales = predictor.predict_today(now)

if not predicted_sales_today_df.empty:
    st.subheader(f"시간대별 청약 건수 예측 ({now.hour}시~23시):")
    st.dataframe(predicted_sales_today_df[['날짜', '시간대', '예측값', '누적_건수', '누적_달성율(%)']].style.format({
        '누적_건수': "{:,.0f}",
        '달성율(%)': "{:.1f}%"
    }), use_container_width=True, hide_index=True, height=(len(predicted_sales_today_df) + 1) * 35 + 3)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"**{today_str} 마감까지 예상 총 건수**", value=f"{today_full_day_estimated_sales:,.0f}건")
    with col2:
        today_23hr_cumulative_sales = predicted_sales_today_df['누적_건수'].iloc[-1]
        achievement_rate_today_23hr = (today_23hr_cumulative_sales / predictor.target_sales) * 100
        st.metric(label=f"**{now.month}월 목표 달성율 (오늘 마감까지)**", value=f"{achievement_rate_today_23hr:.1f}%", delta=f"{predictor.target_sales - today_23hr_cumulative_sales:,.0f}건 남음")

st.markdown("---")

# --- 이번 달 말일까지 일별 예측 ---
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
            cumulative_count_list.append(cumulative_sales)
        else:
            cumulative_sales += row['예측값']
            cumulative_count_list.append(cumulative_sales)

        achievement_rate_list.append((cumulative_sales / predictor.target_sales * 100).round(1))

    daily_predictions['누적_건수'] = cumulative_count_list
    daily_predictions['누적_달성율(%)'] = achievement_rate_list

    st.dataframe(daily_predictions[['날짜', '예측값', '데이터타입', '누적_건수', '누적_달성율(%)']].style.format({
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
st.caption("Powered by Streamlit and Prophet for sales prediction.")