"""
Train-Test Split 시스템 (Mixed 방식) - 새 상품과 중고 상품을 함께 학습
================================================

목적:
    all_data 폴더의 각 하위 폴더(클래스)에서:
    1. train: 새 상품과 중고 상품 이미지를 모두 포함하여 robust한 학습 데이터 구성
    2. test: 각 폴더에서 1장씩만 선택하여 테스트 데이터 구성
    
    이렇게 하면 train에서 생성되는 mean embedding이 더 robust해지고,
    test와의 유사도 매칭 성능이 향상될 것으로 예상됩니다.

입력 스펙:
---------
1. ALL_DATA_DIR: "all_data" - 원본 이미지들이 있는 폴더
2. TRAIN_DIR: "train" - 학습용 이미지들이 저장될 폴더
3. TEST_DIR: "test" - 테스트용 이미지들이 저장될 폴더
4. ALLOWED_EXTS: {".jpg", ".jpeg", ".png", ".webp", ".bmp"} - 지원 이미지 확장자

출력 스펙:
---------
1. train 폴더: 각 클래스별로 새 상품과 중고 상품 이미지 모두 포함
2. test 폴더: 각 클래스별로 1장씩만 선택된 이미지

사용법:
------
python train_test_split_mixed.py

주의사항:
---------
- all_data 폴더에 하위 폴더들이 있어야 합니다
- 각 하위 폴더는 하나의 상품 클래스를 나타냅니다
- train에는 모든 이미지가, test에는 각 폴더당 1장만 복사됩니다
"""

import os
import shutil
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import time

# ==================== 시스템 설정 ====================
ALL_DATA_DIR: str = "all_data"              # 원본 이미지들이 있는 폴더
TRAIN_DIR: str = "train"                    # 학습용 이미지들이 저장될 폴더
TEST_DIR: str = "test"                      # 테스트용 이미지들이 저장될 폴더
ALLOWED_EXTS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # 지원 이미지 확장자

# 랜덤 시드 설정 (재현 가능한 결과를 위해)
RANDOM_SEED: int = 42

# ==================== 핵심 함수들 ====================

def get_folder_structure(root: str) -> Dict[str, List[str]]:
    """
    all_data 폴더의 하위 폴더 구조를 파악하고, 각 폴더별로 이미지 파일 경로를 수집합니다.
    
    입력:
        root (str): 이미지를 찾을 루트 폴더 경로
    
    출력:
        Dict[str, List[str]]: {폴더경로: [이미지파일경로들]} 형태의 딕셔너리
    
    예시:
        >>> get_folder_structure("all_data")
        {'all_data/cat': ['all_data/cat/img1.jpg', 'all_data/cat/img2.png'], 
         'all_data/dog': ['all_data/dog/img3.jpg', 'all_data/dog/img4.png']}
    """
    folder_images: Dict[str, List[str]] = defaultdict(list)
    
    if not os.path.isdir(root):
        print(f"❌ {root} 폴더를 찾을 수 없습니다.")
        return {}
    
    # all_data 폴더의 직접 하위 폴더들만 스캔
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            # 각 하위 폴더 내의 이미지 파일들 수집
            for filename in os.listdir(item_path):
                if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
                    full_path = os.path.join(item_path, filename)
                    folder_images[item_path].append(full_path)
    
    # 각 폴더 내 이미지들을 정렬하여 일관된 결과 보장
    for folder_path in folder_images:
        folder_images[folder_path].sort()
    
    return folder_images


def create_directories(base_dir: str, folder_names: List[str]) -> None:
    """
    train과 test 폴더에 필요한 하위 폴더들을 생성합니다.
    
    입력:
        base_dir (str): 기본 디렉토리 (train 또는 test)
        folder_names (List[str]): 생성할 폴더명 리스트
    
    출력: 없음
    """
    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 폴더 생성: {folder_path}")


def copy_images_to_train(folder_images: Dict[str, List[str]]) -> None:
    """
    모든 이미지를 train 폴더로 복사합니다.
    
    입력:
        folder_images (Dict[str, List[str]]): {폴더경로: [이미지파일경로들]} 형태의 딕셔너리
    
    출력: 없음
    """
    print("\n📁 Train 폴더로 이미지 복사 중...")
    
    total_copied = 0
    
    for folder_path, image_paths in folder_images.items():
        # 폴더명 추출 (all_data/폴더명 → 폴더명)
        folder_name = os.path.basename(folder_path)
        train_folder_path = os.path.join(TRAIN_DIR, folder_name)
        
        print(f"📂 {folder_name}: {len(image_paths)}개 이미지 복사 중...")
        
        for img_path in image_paths:
            # 파일명 추출
            filename = os.path.basename(img_path)
            dest_path = os.path.join(train_folder_path, filename)
            
            try:
                shutil.copy2(img_path, dest_path)
                total_copied += 1
            except Exception as e:
                print(f"⚠️  복사 실패: {img_path} → {e}")
    
    print(f"✅ Train 폴더 복사 완료: 총 {total_copied}개 이미지")


def copy_images_to_test(folder_images: Dict[str, List[str]]) -> None:
    """
    각 폴더에서 thunder_로 시작하는 파일을 1장씩만 test 폴더로 복사합니다.
    
    입력:
        folder_images (Dict[str, List[str]]): {폴더경로: [이미지파일경로들]} 형태의 딕셔너리
    
    출력: 없음
    """
    print("\n📁 Test 폴더로 thunder_ 이미지 복사 중...")
    
    # 랜덤 시드 설정
    random.seed(RANDOM_SEED)
    
    total_copied = 0
    
    for folder_path, image_paths in folder_images.items():
        # 폴더명 추출 (all_data/폴더명 → 폴더명)
        folder_name = os.path.basename(folder_path)
        test_folder_path = os.path.join(TEST_DIR, folder_name)
        
        if not image_paths:
            print(f"⚠️  {folder_name}: 이미지가 없습니다.")
            continue
        
        # thunder_로 시작하는 파일들만 필터링
        thunder_images = [img for img in image_paths if os.path.basename(img).startswith("thunder_")]
        
        if not thunder_images:
            print(f"⚠️  {folder_name}: thunder_로 시작하는 파일이 없습니다.")
            continue
        
        # thunder_ 파일들 중에서 랜덤하게 1장 선택
        selected_image = random.choice(thunder_images)
        filename = os.path.basename(selected_image)
        dest_path = os.path.join(test_folder_path, filename)
        
        try:
            shutil.copy2(selected_image, dest_path)
            total_copied += 1
            print(f"📂 {folder_name}: {filename} 선택됨 (thunder_ 파일)")
        except Exception as e:
            print(f"⚠️  복사 실패: {selected_image} → {e}")
    
    print(f"✅ Test 폴더 복사 완료: 총 {total_copied}개 thunder_ 이미지")


def analyze_split_results(folder_images: Dict[str, List[str]]) -> None:
    """
    train-test split 결과를 분석하고 출력합니다.
    
    입력:
        folder_images (Dict[str, List[str]]): {폴더경로: [이미지파일경로들]} 형태의 딕셔너리
    
    출력: 없음
    """
    print("\n📊 Train-Test Split 결과 분석")
    print("=" * 60)
    
    total_folders = len(folder_images)
    total_images = sum(len(images) for images in folder_images.values())
    
    print(f"📁 총 폴더 수: {total_folders}")
    print(f"📸 총 이미지 수: {total_images}")
    print(f"🚂 Train 이미지 수: {total_images}")
    print(f"🧪 Test 이미지 수: {total_folders}")
    print(f"📈 Train/Test 비율: {total_images}:{total_folders}")
    
    print(f"\n📋 폴더별 상세 정보:")
    print("-" * 60)
    
    for folder_path, image_paths in folder_images.items():
        folder_name = os.path.basename(folder_path)
        image_count = len(image_paths)
        
        # thunder_ 파일 개수 계산
        thunder_count = sum(1 for img in image_paths if os.path.basename(img).startswith("thunder_"))
        
        print(f"📂 {folder_name}:")
        print(f"   - 총 이미지: {image_count}개")
        print(f"   - thunder_ 파일: {thunder_count}개")
        print(f"   - Test 선택: thunder_ 파일 1개")
        print()


def main() -> None:
    """
    메인 실행 함수: train-test split을 수행합니다.
    
    입력: 없음
    
    출력: 없음 (폴더와 파일이 생성됨)
    
    처리 과정:
    1. all_data 폴더 구조 파악
    2. train과 test 폴더 생성
    3. 모든 이미지를 train으로 복사
    4. 각 폴더에서 1장씩을 test로 복사
    5. 결과 분석 및 출력
    
    예외 처리:
    - FileNotFoundError: all_data 폴더가 존재하지 않을 때
    - RuntimeError: 이미지 파일을 찾을 수 없을 때
    - OSError: 파일 복사 오류
    """
    start_time: float = time.time()
    
    print("🚀 Train-Test Split (Mixed 방식) 시작")
    print("=" * 60)
    print(f"📁 원본 데이터: {ALL_DATA_DIR}")
    print(f"🚂 학습 데이터: {TRAIN_DIR}")
    print(f"🧪 테스트 데이터: {TEST_DIR}")
    print(f"🎲 랜덤 시드: {RANDOM_SEED}")
    print("=" * 60)
    
    # 1. all_data 폴더 존재 확인
    if not os.path.isdir(ALL_DATA_DIR):
        raise FileNotFoundError(f"ALL_DATA_DIR not found: {ALL_DATA_DIR}")
    
    # 2. 폴더 구조 파악 및 이미지 파일 경로 수집
    print("🔍 폴더 구조 및 이미지 파일 스캔 중...")
    folder_images = get_folder_structure(ALL_DATA_DIR)
    
    if not folder_images:
        raise RuntimeError(f"No folders with images found under: {ALL_DATA_DIR}")
    
    folder_names = [os.path.basename(path) for path in folder_images.keys()]
    folder_names.sort()  # 폴더명을 정렬하여 일관된 결과 보장
    
    total_images = sum(len(images) for images in folder_images.values())
    print(f"✅ 스캔 완료: {len(folder_names)}개 폴더, {total_images}개 이미지")
    
    # 3. train과 test 폴더 생성
    print("\n📁 디렉토리 생성 중...")
    create_directories(TRAIN_DIR, folder_names)
    create_directories(TEST_DIR, folder_names)
    
    # 4. 모든 이미지를 train으로 복사
    copy_images_to_train(folder_images)
    
    # 5. 각 폴더에서 1장씩을 test로 복사
    copy_images_to_test(folder_images)
    
    # 6. 결과 분석 및 출력
    analyze_split_results(folder_images)
    
    # 7. 완료 정보 출력
    elapsed_time: float = time.time() - start_time
    print("✅ Train-Test Split 완료!")
    print(f"⏱️  소요 시간: {elapsed_time:.1f}초")
    
    print(f"\n🎯 다음 단계:")
    print(f"1. embeddings_mean.py 또는 embeddings_mean_cleaned.py 실행")
    print(f"2. train 폴더의 모든 이미지로 mean embedding 생성")
    print(f"3. test 폴더의 이미지로 유사도 검색 테스트")
    
    print(f"\n💡 Mixed 방식의 장점:")
    print(f"- Train에 모든 이미지 포함 (새 상품 + 중고 상품)")
    print(f"- Test에는 thunder_ 파일만 각 폴더당 1장씩 선택")
    print(f"- 더 robust한 클래스별 대표 벡터 생성")
    print(f"- Test와의 유사도 매칭 성능 향상 예상")


# ==================== 스크립트 실행 ====================

if __name__ == "__main__":
    try:
        main()
        print("\n🎉 Train-Test Split이 성공적으로 완료되었습니다!")
        print("이제 train 폴더에는 모든 이미지가, test 폴더에는 각 폴더당 1장씩이 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   다음 사항들을 확인해주세요:")
        print("   1. all_data 폴더가 존재하는지")
        print("   2. all_data 폴더에 하위 폴더들이 있는지")
        print("   3. 각 하위 폴더에 이미지 파일이 있는지")
        print("   4. 충분한 디스크 공간이 있는지")
        print("   5. 파일 복사 권한이 있는지")
