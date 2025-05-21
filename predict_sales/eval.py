import sys
from datetime import datetime
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
from predict import SalesPredictor

class SalesEvaluator:
    def __init__(self, predictor: SalesPredictor, year: int, month: int):
        """
        판매 예측 평가기 초기화
        
        Args:
            predictor: 학습된 SalesPredictor 인스턴스
            year: 평가할 연도
            month: 평가할 월
        """
        self.predictor = predictor
        self.year = year
        self.month = month
        self.data = predictor.data

    def _get_month_end_day(self, year: int, month: int) -> int:
        """해당 월의 마지막 날짜를 반환"""
        if month in [4, 6, 9, 11]:
            return 30
        elif month == 2:
            return 29 if self._is_leap_year(year) else 28
        else:
            return 31

    def _is_leap_year(self, year: int) -> bool:
        """윤년 여부를 반환"""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def evaluate(self) -> Tuple[pd.DataFrame, float]:
        """
        지정된 연도와 월에 대해 일별 평가 수행
        
        Returns:
            Tuple[pd.DataFrame, float]: (평가 결과 데이터프레임, 전체 정확도)
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        # 해당 월의 데이터 추출
        eval_data = self.data[
            (self.data['datetime'].dt.year == self.year) & 
            (self.data['datetime'].dt.month == self.month)
        ]

        if eval_data.empty:
            raise ValueError(f"{self.year}년 {self.month}월의 데이터가 없습니다.")

        # 일별 실제값 계산
        daily_actual = eval_data.groupby(eval_data['datetime'].dt.date)['건수'].sum().reset_index()
        daily_actual = daily_actual.rename(columns={'datetime': '날짜'})
        daily_actual['날짜'] = pd.to_datetime(daily_actual['날짜']).dt.strftime('%Y-%m-%d')

        # 예측 수행
        start_date = datetime(self.year, self.month, 1)
        end_date = datetime(self.year, self.month, self._get_month_end_day(self.year, self.month))
        daily_predictions = self.predictor.predict(start_date, end_date)

        # 실제값과 예측값 병합
        eval_results = pd.merge(
            daily_actual[['날짜', '건수']], 
            daily_predictions[['날짜', '예측값']], 
            on='날짜',
            how='outer'
        )

        # 결측치 처리
        eval_results = eval_results.fillna(0)

        # 컬럼 이름 변경 및 정확도 계산
        eval_results = eval_results.rename(columns={'건수': '실제값'})
        eval_results['abs_error'] = (eval_results['실제값'] - eval_results['예측값']).abs()
        eval_results['정확도(%)'] = (
            100 * (1 - eval_results['abs_error'] / eval_results['실제값'].replace(0, pd.NA))
        ).fillna(0).round(1)

        # 총 정확도 계산
        total_actual = eval_results['실제값'].sum()
        total_predicted = eval_results['예측값'].sum()
        total_accuracy = (
            100 * (1 - abs(total_actual - total_predicted) / total_actual)
        ).round(1) if total_actual != 0 else 0.0

        return eval_results, total_accuracy

def parse_year_month(year_month: str) -> Tuple[int, int]:
    """
    YYYYMM 형식의 문자열을 파싱하여 연도와 월을 반환
    
    Args:
        year_month: YYYYMM 형식의 문자열 (예: 2505)
        
    Returns:
        Tuple[int, int]: (연도, 월)
        
    Raises:
        ValueError: 잘못된 입력 형식이나 범위일 경우
    """
    if len(year_month) != 4:
        raise ValueError("입력은 YYMM 형식이어야 합니다 (예: 2505)")
    
    try:
        year = int(year_month[:2]) + 2000  # 2505 → 2025
        month = int(year_month[2:])        # 2505 → 5
        
        if month < 1 or month > 12:
            raise ValueError("월은 1~12 사이여야 합니다.")
            
        return year, month
    except ValueError as e:
        raise ValueError(f"잘못된 입력 형식입니다: {str(e)}")

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("사용법: python eval.py YYYYMM (예: python eval.py 2505)")
            sys.exit(1)

        # 입력 파싱
        year, month = parse_year_month(sys.argv[1])

        # 예측 모델 초기화
        predictor = SalesPredictor(data_dir="data")
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

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        sys.exit(1)