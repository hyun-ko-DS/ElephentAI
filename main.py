"""
메인 실행 스크립트
================

전체 이미지 유사도 검색 파이프라인을 순차적으로 실행합니다.

실행 순서:
1. train_test_split.py - train/test 데이터 분할
2. embeddings_huge.py - OpenCLIP ViT-g-14 임베딩 생성
3. predict_huge.py - 자동화된 유사도 검색 수행
4. result_calculator.py - 결과 정확도 분석

사용법:
------
python main.py
"""

import os
import sys
import subprocess
import time
from typing import List, Tuple

def run_script(script_name: str, description: str) -> Tuple[bool, str]:
    """
    Python 스크립트를 실행합니다.
    
    입력:
        script_name (str): 실행할 스크립트 파일명
        description (str): 스크립트 설명
    
    출력:
        Tuple[bool, str]: (성공 여부, 오류 메시지)
    """
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"📁 실행 파일: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 스크립트 실행
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} 완료!")
            print(f"⏱️  실행 시간: {execution_time:.2f}초")
            
            # 표준 출력이 있으면 표시
            if result.stdout.strip():
                print(f"\n📋 실행 결과:")
                print("-" * 50)
                print(result.stdout)
            
            return True, ""
        else:
            error_msg = f"❌ {description} 실패 (종료 코드: {result.returncode})"
            if result.stderr.strip():
                error_msg += f"\n오류 메시지:\n{result.stderr}"
            if result.stdout.strip():
                error_msg += f"\n표준 출력:\n{result.stdout}"
            
            print(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"❌ {description} 실행 중 예외 발생: {str(e)}"
        print(error_msg)
        return False, error_msg

def check_prerequisites() -> bool:
    """
    실행 전 필요한 파일들이 존재하는지 확인합니다.
    
    출력:
        bool: 모든 조건이 충족되면 True
    """
    print("🔍 실행 전 조건 확인 중...")
    
    required_files = [
        "train_test_split.py",
        "embeddings_huge.py", 
        "predict_huge.py",
        "result_calculator.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 필요한 파일이 누락되었습니다:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 모든 필요한 파일이 존재합니다.")
    return True

def check_data_directories() -> bool:
    """
    필요한 데이터 디렉토리들이 존재하는지 확인합니다.
    
    출력:
        bool: 모든 디렉토리가 존재하면 True
    """
    print("📁 데이터 디렉토리 확인 중...")
    
    required_dirs = [
        "train",
        "test"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ 필요한 디렉토리가 누락되었습니다:")
        for dir_name in missing_dirs:
            print(f"   - {dir_name}/")
        print("\n💡 train_test_split.py를 먼저 실행하여 디렉토리를 생성하세요.")
        return False
    
    print("✅ 필요한 데이터 디렉토리가 존재합니다.")
    return True

def main():
    """
    메인 실행 함수
    """
    print("🎯 OpenCLIP ViT-g-14 이미지 유사도 검색 파이프라인 시작")
    print("=" * 80)
    
    # 1. 실행 전 조건 확인
    if not check_prerequisites():
        print("\n❌ 실행을 중단합니다. 필요한 파일을 확인하세요.")
        return
    
    # 2. train_test_split.py 실행
    success, error = run_script(
        "train_test_split.py", 
        "1단계: Train/Test 데이터 분할"
    )
    if not success:
        print(f"\n❌ 1단계 실패로 인해 파이프라인을 중단합니다.")
        print(f"오류: {error}")
        return
    
    # 3. 데이터 디렉토리 확인
    if not check_data_directories():
        print(f"\n❌ 데이터 디렉토리 확인 실패로 인해 파이프라인을 중단합니다.")
        return
    
    # 4. embeddings_huge.py 실행
    success, error = run_script(
        "embeddings_huge.py", 
        "2단계: OpenCLIP ViT-g-14 임베딩 생성"
    )
    if not success:
        print(f"\n❌ 2단계 실패로 인해 파이프라인을 중단합니다.")
        print(f"오류: {error}")
        return
    
    # 5. predict_huge.py 실행
    success, error = run_script(
        "predict_huge.py", 
        "3단계: 자동화된 유사도 검색 수행"
    )
    if not success:
        print(f"\n❌ 3단계 실패로 인해 파이프라인을 중단합니다.")
        print(f"오류: {error}")
        return
    
    # 6. 결과 파일 확인
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"\n❌ 결과 디렉토리가 생성되지 않았습니다: {results_dir}")
        return
    
    # CSV 파일 찾기
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"\n❌ 결과 CSV 파일을 찾을 수 없습니다: {results_dir}")
        return
    
    # 가장 최근 CSV 파일 선택 (ViT_g_14_result.csv 우선)
    target_csv = None
    for csv_file in csv_files:
        if "ViT_g_14_result.csv" in csv_file:
            target_csv = csv_file
            break
    
    if not target_csv:
        target_csv = csv_files[0]  # 첫 번째 CSV 파일 사용
    
    csv_path = os.path.join(results_dir, target_csv)
    print(f"\n📊 분석할 결과 파일: {csv_path}")
    
    # 7. result_calculator.py 실행
    success, error = run_script(
        f"result_calculator.py {csv_path}", 
        "4단계: 결과 정확도 분석"
    )
    if not success:
        print(f"\n❌ 4단계 실패: {error}")
        return
    
    # 8. 완료 메시지
    print(f"\n{'='*80}")
    print("🎉 모든 파이프라인 실행 완료!")
    print(f"{'='*80}")
    print("📋 실행된 단계:")
    print("   1. ✅ Train/Test 데이터 분할")
    print("   2. ✅ OpenCLIP ViT-g-14 임베딩 생성")
    print("   3. ✅ 자동화된 유사도 검색 수행")
    print("   4. ✅ 결과 정확도 분석")
    print(f"\n📁 결과 파일 위치: {results_dir}/")
    print(f"📊 분석 결과: {csv_path}")
    print(f"\n🚀 파이프라인 실행이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  사용자에 의해 실행이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
        sys.exit(1)
