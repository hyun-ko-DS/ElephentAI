"""
결과 정확도 계산기
================

입력 스펙:
---------
1. CSV 파일 경로 (results 폴더 내의 CSV 파일)
   - test_folder: 테스트 이미지가 있는 폴더명
   - top_similar_predict: 가장 유사하다고 예측한 폴더명

출력 스펙:
---------
1. 전체 정확도 (Accuracy)
2. 폴더별 정확도 분석
3. 오분류된 케이스들 상세 분석
4. 혼동 행렬 (Confusion Matrix) 형태의 결과

사용법:
------
python result_calculator.py results/ViT_g_14_result.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

def load_results_csv(csv_path: str) -> pd.DataFrame:
    """
    결과 CSV 파일을 로드합니다.
    
    입력:
        csv_path (str): CSV 파일 경로
    
    출력:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 필수 컬럼 확인
        required_columns = ['test_folder', 'top_similar_predict']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        print(f"✅ CSV 파일 로드 완료: {csv_path}")
        print(f"📊 총 {len(df)}개 행, {len(df.columns)}개 컬럼")
        print(f"📋 컬럼: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ CSV 파일 로드 실패: {e}")
        return None

def calculate_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    """
    전체 정확도를 계산합니다.
    
    입력:
        df (pd.DataFrame): 결과 데이터프레임
    
    출력:
        Dict[str, float]: 정확도 관련 지표들
    """
    # 검색 실패나 오류 케이스 제외
    valid_df = df[
        (df['top_similar_predict'] != '검색 실패') & 
        (~df['top_similar_predict'].str.startswith('오류:', na=False))
    ].copy()
    
    if len(valid_df) == 0:
        print("⚠️  유효한 검색 결과가 없습니다.")
        return {}
    
    # 정확도 계산
    correct_predictions = (valid_df['test_folder'] == valid_df['top_similar_predict']).sum()
    total_predictions = len(valid_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # 검색 실패율 계산
    failed_searches = len(df) - len(valid_df)
    failure_rate = failed_searches / len(df) if len(df) > 0 else 0.0
    
    results = {
        'total_samples': len(df),
        'valid_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'failed_searches': failed_searches,
        'failure_rate': failure_rate
    }
    
    return results

def analyze_folder_accuracy(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    폴더별 정확도를 분석합니다.
    
    입력:
        df (pd.DataFrame): 결과 데이터프레임
    
    출력:
        Dict[str, Dict[str, float]]: 폴더별 정확도 정보
    """
    folder_accuracy = {}
    
    # 검색 실패나 오류 케이스 제외
    valid_df = df[
        (df['top_similar_predict'] != '검색 실패') & 
        (~df['top_similar_predict'].str.startswith('오류:', na=False))
    ].copy()
    
    for folder in valid_df['test_folder'].unique():
        folder_data = valid_df[valid_df['test_folder'] == folder]
        
        if len(folder_data) == 0:
            continue
            
        correct = (folder_data['test_folder'] == folder_data['top_similar_predict']).sum()
        total = len(folder_data)
        accuracy = correct / total if total > 0 else 0.0
        
        folder_accuracy[folder] = {
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'incorrect_predictions': total - correct
        }
    
    return folder_accuracy

def analyze_misclassifications(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    오분류된 케이스들을 분석합니다.
    
    입력:
        df (pd.DataFrame): 결과 데이터프레임
    
    출력:
        Dict[str, List[Dict]]: 오분류 분석 결과
    """
    # 검색 실패나 오류 케이스 제외
    valid_df = df[
        (df['top_similar_predict'] != '검색 실패') & 
        (~df['top_similar_predict'].str.startswith('오류:', na=False))
    ].copy()
    
    # 오분류된 케이스들
    misclassified = valid_df[valid_df['test_folder'] != valid_df['top_similar_predict']].copy()
    
    # 실제 폴더별로 잘못 분류된 케이스들 그룹화
    misclass_by_actual = {}
    for _, row in misclassified.iterrows():
        actual_folder = row['test_folder']
        predicted_folder = row['top_similar_predict']
        
        if actual_folder not in misclass_by_actual:
            misclass_by_actual[actual_folder] = []
        
        misclass_by_actual[actual_folder].append({
            'test_filename': row['test_filename'],
            'predicted_folder': predicted_folder,
            'similarity_score': row.get('similarity_score', 'N/A')
        })
    
    return misclass_by_actual

def create_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    혼동 행렬을 생성합니다.
    
    입력:
        df (pd.DataFrame): 결과 데이터프레임
    
    출력:
        pd.DataFrame: 혼동 행렬
    """
    # 검색 실패나 오류 케이스 제외
    valid_df = df[
        (df['top_similar_predict'] != '검색 실패') & 
        (~df['top_similar_predict'].str.startswith('오류:', na=False))
    ].copy()
    
    if len(valid_df) == 0:
        return pd.DataFrame()
    
    # 혼동 행렬 생성
    confusion_matrix = pd.crosstab(
        valid_df['test_folder'], 
        valid_df['top_similar_predict'], 
        margins=True,
        margins_name='Total'
    )
    
    return confusion_matrix

def print_detailed_analysis(df: pd.DataFrame, csv_path: str) -> None:
    """
    상세한 분석 결과를 출력합니다.
    
    입력:
        df (pd.DataFrame): 결과 데이터프레임
        csv_path (str): CSV 파일 경로
    """
    # print("\n" + "=" * 80)
    # print(f"📊 결과 분석: {os.path.basename(csv_path)}")
    # print("=" * 80)
    
    # 1. 전체 정확도
    accuracy_results = calculate_accuracy(df)
    if not accuracy_results:
        return
    
    print(f"\n🎯 전체 정확도 분석")
    print("-" * 50)
    print(f"📁 총 샘플 수: {accuracy_results['total_samples']:,}")
    print(f"✅ 유효한 예측: {accuracy_results['valid_predictions']:,}")
    print(f"❌ 검색 실패/오류: {accuracy_results['failed_searches']:,}")
    print(f"📊 정확도: {accuracy_results['accuracy']:.4f} ({accuracy_results['accuracy']*100:.2f}%)")
    print(f"⚠️  검색 실패율: {accuracy_results['failure_rate']:.4f} ({accuracy_results['failure_rate']*100:.2f}%)")
    
    # 2. 폴더별 정확도
    folder_accuracy = analyze_folder_accuracy(df)
    if folder_accuracy:
        print(f"\n📂 폴더별 정확도 분석")
        print("-" * 50)
        
        # 정확도 순으로 정렬
        sorted_folders = sorted(folder_accuracy.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for folder, stats in sorted_folders:
            print(f"{folder}: {stats['accuracy']:.4f} ({stats['correct_predictions']}/{stats['total_samples']})")
    
    # 3. 오분류 분석
    misclassifications = analyze_misclassifications(df)
    if misclassifications:
        print(f"\n❌ 오분류 분석")
        print("-" * 50)
        
        for actual_folder, cases in misclassifications.items():
            # print(f"\n🔍 {actual_folder} 폴더에서 잘못 분류된 케이스들:")
            
            # 예측된 폴더별로 그룹화
            pred_counts = Counter([case['predicted_folder'] for case in cases])
            
            for pred_folder, count in pred_counts.most_common():
                # print(f"   → {pred_folder}: {count}개")
                
                # 상세 정보 (처음 3개만)
                detailed_cases = [case for case in cases if case['predicted_folder'] == pred_folder][:3]
                for case in detailed_cases:
                    similarity = case.get('similarity_score', 'N/A')
                    # if isinstance(similarity, (int, float)):
                        # print(f"     - {case['test_filename']} (유사도: {similarity:.4f})")
                    # else:
                        # print(f"     - {case['test_filename']} (유사도: {similarity})")
    
    # 4. 혼동 행렬
    confusion_matrix = create_confusion_matrix(df)
    if not confusion_matrix.empty:
        print(f"\n📋 혼동 행렬 (Confusion Matrix)")
        print("-" * 50)
        print(confusion_matrix.to_string())
    
    # 5. 요약
    print(f"\n📈 성능 요약")
    print("-" * 50)
    print(f"🎯 전체 정확도: {accuracy_results['accuracy']*100:.2f}%")
    print(f"📁 분석된 폴더 수: {len(folder_accuracy)}")
    print(f"❌ 오분류된 케이스: {accuracy_results['valid_predictions'] - accuracy_results['correct_predictions']}")
    
    if folder_accuracy:
        best_folder = max(folder_accuracy.items(), key=lambda x: x[1]['accuracy'])
        worst_folder = min(folder_accuracy.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"🏆 최고 성능 폴더: {best_folder[0]} ({best_folder[1]['accuracy']*100:.2f}%)")
        print(f"📉 최저 성능 폴더: {worst_folder[0]} ({worst_folder[1]['accuracy']*100:.2f}%)")

def main():
    """
    메인 함수
    """
    if len(sys.argv) != 2:
        print("❌ 사용법: python result_calculator.py <CSV_파일_경로>")
        print("예시: python result_calculator.py results/ViT_g_14_result.csv")
        return
    
    csv_path = sys.argv[1]
    
    # CSV 파일 로드
    df = load_results_csv(csv_path)
    if df is None:
        return
    
    # 상세 분석 수행
    print_detailed_analysis(df, csv_path)

if __name__ == "__main__":
    main()
