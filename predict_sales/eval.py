import sys
from datetime import datetime
import pandas as pd
from predict import SalesPredictor

class SalesEvaluator:
    def __init__(self, predictor, year, month):
        self.predictor = predictor
        self.year = year
        self.month = month
        self.data = predictor.data

    def evaluate(self):
        """지정된 연도와 월에 대해 일별 평가 수행"""
        # 해당 월의 데이터 추출
        eval_data = self.data[(self.data['datetime'].dt.year == self.year) & (self.data['datetime'].dt.month == self.month)]

        # 일별 실제값 계산
        daily_actual = eval_data.groupby(eval_data['datetime'].dt.date)['건수'].sum().reset_index()
        daily_actual = daily_actual.rename(columns={'datetime': '날짜'})
        daily_actual['날짜'] = pd.to_datetime(daily_actual['날짜']).dt.strftime('%Y-%m-%d')

        # 예측 수행 (1일부터 말일까지)
        start_date = datetime(self.year, self.month, 1)
        if self.month == 4 or self.month == 6 or self.month == 9 or self.month == 11:
            end_date = datetime(self.year, self.month, 30)
        elif self.month == 2:
            # 윤년 확인
            if (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400 == 0):
                end_date = datetime(self.year, self.month, 29)
            else:
                end_date = datetime(self.year, self.month, 28)
        else:
            end_date = datetime(self.year, self.month, 31)
            
        daily_predictions = self.predictor.predict(start_date, end_date)

        # 실제값과 예측값 병합
        eval_results = pd.merge(daily_actual[['날짜', '건수']], daily_predictions[['날짜', '예측값']], on='날짜')

        # 컬럼 이름 변경 및 정확도 계산
        eval_results = eval_results.rename(columns={'건수': '실제값'})
        eval_results['abs_error'] = (eval_results['실제값'] - eval_results['예측값']).abs()
        eval_results['정확도(%)'] = (100 * (1 - eval_results['abs_error'] / eval_results['실제값'].replace(0, pd.NA))).fillna(0).round(1)

        # 총 정확도 계산
        total_actual = eval_results['실제값'].sum()
        total_predicted = eval_results['예측값'].sum()
        total_accuracy = (100 * (1 - abs(total_actual - total_predicted) / total_actual)).round(1) if total_actual != 0 else 0

        return eval_results, total_accuracy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python eval.py YYYYMM (예: python eval.py 2505)")
        sys.exit(1)

    # 입력 파싱
    try:
        year_month = sys.argv[1]
        year = int(year_month[:2]) + 2000  # 2505 → 2025
        month = int(year_month[2:])        # 2505 → 5
        if month < 1 or month > 12:
            raise ValueError("월은 1~12 사이여야 합니다.")
    except ValueError as e:
        print(f"잘못된 입력 형식입니다: {e}")
        sys.exit(1)

    # 예측 모델 초기화
    predictor = SalesPredictor(data_dir="/Users/linakorea/Project/python/predict_sales/data/")
    predictor.load_holidays()
    predictor.load_data()
    predictor.calculate_actual_may()
    predictor.train()

    # 평가 수행
    evaluator = SalesEvaluator(predictor, year, month)
    eval_results, total_accuracy = evaluator.evaluate()

    # 결과 출력
    print(f"\n{year}년 {month}월 일별 평가 결과:")
    print(eval_results[['날짜', '실제값', '예측값', '정확도(%)']].to_string(index=False))
    print(f"\n{year}년 {month}월 총 정확도: {total_accuracy}%")