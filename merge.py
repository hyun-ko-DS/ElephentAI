"""
CSV 파일 병합 및 전처리 시스템
============================

목적:
    번개장터, 중고나라, 당근마켓의 상품 정보 CSV 파일들을 병합하고,
    가격 데이터를 정규화하여 통합된 products_info.csv 파일을 생성합니다.

입력 스펙:
---------
1. thunder: "../records/bunjang_products.csv" - 번개장터 상품 정보
2. joongna: "../records/joongna_products.csv" - 중고나라 상품 정보  
3. carrot: "../records/carrot_products.csv" - 당근마켓 상품 정보

출력 스펙:
---------
1. "../records/products_info.csv" - 통합된 상품 정보 파일
   - 모든 플랫폼의 상품 정보가 하나의 CSV로 병합됨
   - 가격 데이터가 정수형으로 정규화됨
   - 인덱스가 재설정됨

데이터 전처리:
-------------
- 번개장터: 쉼표(,) 제거 후 정수형 변환
- 중고나라: "원" 제거, 쉼표(,) 제거 후 정수형 변환
- 당근마켓: 원본 데이터 그대로 사용

사용법:
------
python merge.py

주의사항:
---------
- 입력 CSV 파일들이 records 폴더에 존재해야 합니다
- 가격 컬럼명이 'price'여야 합니다
- 출력 파일은 상위 디렉토리의 records 폴더에 저장됩니다
"""

import pandas as pd
from typing import None

def merge_csv(thunder: pd.DataFrame, joongna: pd.DataFrame, carrot: pd.DataFrame) -> None:
    """
    세 개의 플랫폼 CSV 데이터를 병합하고 가격 데이터를 정규화합니다.
    
    입력:
        thunder (pd.DataFrame): 번개장터 상품 데이터프레임
        joongna (pd.DataFrame): 중고나라 상품 데이터프레임
        carrot (pd.DataFrame): 당근마켓 상품 데이터프레임
    
    출력: 없음 (파일로 저장됨)
    
    처리 과정:
    1. 번개장터 가격 데이터 정규화 (쉼표 제거 → 정수형 변환)
    2. 중고나라 가격 데이터 정규화 ("원" 제거, 쉼표 제거 → 정수형 변환)
    3. 세 데이터프레임을 세로로 병합 (concat)
    4. 인덱스 재설정
    5. 통합된 데이터를 CSV 파일로 저장
    6. 병합 결과 미리보기 출력 (상위 3개 행)
    """
    
    # 1. 번개장터 가격 데이터 정규화
    # 쉼표(,) 제거 후 정수형으로 변환
    # 예: "15,000" → 15000
    thunder['price'] = thunder['price'].str.replace(',', '').astype(int)
    
    # 2. 중고나라 가격 데이터 정규화
    # "원" 제거 후 쉼표(,) 제거, 정수형으로 변환
    # 예: "15,000원" → 15000
    joongna['price'] = joongna['price'].str.replace('원', '')
    joongna['price'] = joongna['price'].str.replace(',', '').astype(int)
    
    # 3. 세 데이터프레임을 세로로 병합 (axis=0)
    # concat은 기본적으로 세로 방향으로 데이터를 쌓음
    df: pd.DataFrame = pd.concat([carrot, joongna, thunder], axis=0).reset_index(drop=True)
    
    # 4. 통합된 데이터를 CSV 파일로 저장
    # index=False: 인덱스 번호는 저장하지 않음
    output_path: str = '../records/products_info.csv'
    df.to_csv(output_path, index=False)
    
    # 5. 병합 결과 확인 (상위 3개 행 출력)
    print(f"✅ CSV 병합 완료: {output_path}")
    print(f"📊 총 {len(df)}개의 상품 정보가 병합되었습니다.")
    print(f"🔍 병합 결과 미리보기 (상위 3개 행):")
    print(df.head(3))
    
    # 6. 각 플랫폼별 데이터 개수 확인
    print(f"\n�� 플랫폼별 데이터 개수:")
    print(f"   당근마켓: {len(carrot)}개")
    print(f"   중고나라: {len(joongna)}개")
    print(f"   번개장터: {len(thunder)}개")
    print(f"   총합: {len(df)}개")

def load_csv_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    세 개의 플랫폼 CSV 파일을 로드합니다.
    
    입력: 없음
    
    출력:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        (번개장터, 중고나라, 당근마켓) 데이터프레임 튜플
    
    파일 경로:
    - 번개장터: "../records/bunjang_products.csv"
    - 중고나라: "../records/joongna_products.csv"
    - 당근마켓: "../records/carrot_products.csv"
    
    주의사항:
    - .iloc[:,1:]로 첫 번째 컬럼(인덱스)을 제외하고 로드
    - 상대 경로를 사용하므로 실행 위치가 중요
    """
    
    # 번개장터 CSV 로드 (첫 번째 컬럼 제외)
    thunder_path: str = "../records/bunjang_products.csv"
    thunder: pd.DataFrame = pd.read_csv(thunder_path).iloc[:, 1:]
    print(f"✅ 번개장터 데이터 로드 완료: {thunder_path} ({len(thunder)}개 행)")
    
    # 중고나라 CSV 로드 (첫 번째 컬럼 제외)
    joongna_path: str = "../records/joongna_products.csv"
    joongna: pd.DataFrame = pd.read_csv(joongna_path).iloc[:, 1:]
    print(f"✅ 중고나라 데이터 로드 완료: {joongna_path} ({len(joongna)}개 행)")
    
    # 당근마켓 CSV 로드 (첫 번째 컬럼 제외)
    carrot_path: str = "../records/carrot_products.csv"
    carrot: pd.DataFrame = pd.read_csv(carrot_path).iloc[:, 1:]
    print(f"✅ 당근마켓 데이터 로드 완료: {carrot_path} ({len(carrot)}개 행)")
    
    return thunder, joongna, carrot

def validate_data(df: pd.DataFrame, platform_name: str) -> None:
    """
    데이터프레임의 기본 정보를 검증하고 출력합니다.
    
    입력:
        df (pd.DataFrame): 검증할 데이터프레임
        platform_name (str): 플랫폼 이름 (로깅용)
    
    출력: 없음 (검증 결과만 출력)
    
    검증 항목:
    - 데이터프레임 크기 (행 x 열)
    - 컬럼명 목록
    - 가격 컬럼의 데이터 타입
    - 가격 컬럼의 기본 통계 (최소값, 최대값, 평균)
    """
    
    print(f"\n🔍 {platform_name} 데이터 검증:")
    print(f"   크기: {df.shape[0]}행 x {df.shape[1]}열")
    print(f"   컬럼: {list(df.columns)}")
    
    if 'price' in df.columns:
        price_col = df['price']
        print(f"   가격 타입: {price_col.dtype}")
        print(f"   가격 범위: {price_col.min():,}원 ~ {price_col.max():,}원")
        print(f"   평균 가격: {price_col.mean():,.0f}원")
    else:
        print(f"   ⚠️  'price' 컬럼을 찾을 수 없습니다!")

# ==================== 메인 실행 부분 ====================

if __name__ == "__main__":
    try:
        print("🚀 CSV 병합 시스템 시작")
        print("=" * 50)
        
        # 1. CSV 파일들 로드
        thunder, joongna, carrot = load_csv_files()
        
        # 2. 각 플랫폼 데이터 검증
        validate_data(carrot, "당근마켓")
        validate_data(joongna, "중고나라")
        validate_data(thunder, "번개장터")
        
        print("\n" + "=" * 50)
        
        # 3. CSV 병합 및 저장
        merge_csv(thunder, joongna, carrot)
        
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
        print("�� 다음 사항들을 확인해주세요:")
        print("   1. records 폴더가 존재하는지")
        print("   2. CSV 파일들이 올바른 경로에 있는지")
        print("   3. 파일명이 정확한지")
        print("   4. 실행 위치가 올바른지")
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        print("�� 오류 내용을 확인하고 다시 시도해주세요.")