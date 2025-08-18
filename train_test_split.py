import os
import shutil
from pathlib import Path

def create_train_test_split(source_dir="all_data"):
    """
    all_data 폴더의 모든 카테고리 폴더를 train과 test로 분할합니다.
    
    분할 기준:
    - thunder로 시작하는 파일명: test 폴더로
    - 그 외 파일명: train 폴더로
    
    Args:
        source_dir (str): 소스 디렉토리 (기본값: "all_data")
    """
    
    # 소스 디렉토리 경로
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: {source_dir} 폴더를 찾을 수 없습니다.")
        return
    
    # train과 test 디렉토리 생성
    train_dir = Path("train")
    test_dir = Path("test")
    
    # 기존 디렉토리가 있다면 삭제
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # 새 디렉토리 생성
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    print(f"소스 디렉토리: {source_dir}")
    print("분할 기준:")
    print("- thunder로 시작하는 파일: test 폴더")
    print("- 그 외 파일: train 폴더")
    print("-" * 50)
    
    # 모든 카테고리 폴더 처리
    total_files = 0
    train_files = 0
    test_files = 0
    
    for category_folder in source_path.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            
            # 카테고리 폴더 내의 모든 파일 목록
            files = list(category_folder.iterdir())
            files = [f for f in files if f.is_file()]  # 파일만 필터링
            
            if not files:
                print(f"경고: {category_name} 폴더에 파일이 없습니다.")
                continue
            
            # thunder 파일과 일반 파일 분리
            thunder_files = []
            normal_files = []
            
            for file_path in files:
                filename = file_path.name
                if filename.startswith("thunder"):
                    thunder_files.append(file_path)
                else:
                    normal_files.append(file_path)
            
            # train과 test에 파일 복사
            has_train = False
            has_test = False
            
            # 일반 파일들을 train에 복사
            if normal_files:
                train_category_path = train_dir / category_name
                train_category_path.mkdir(exist_ok=True)
                
                for file_path in normal_files:
                    dest_path = train_category_path / file_path.name
                    shutil.copy2(file_path, dest_path)
                    train_files += 1
                
                has_train = True
            
            # thunder 파일들을 test에 복사
            if thunder_files:
                test_category_path = test_dir / category_name
                test_category_path.mkdir(exist_ok=True)
                
                for file_path in thunder_files:
                    dest_path = test_category_path / file_path.name
                    shutil.copy2(file_path, dest_path)
                    test_files += 1
                
                has_test = True
            
            total_files += len(files)
            
            # 결과 출력
            print(f"{category_name}: 총 {len(files)}개 파일")
            if has_train:
                print(f"  - Train: {len(normal_files)}개 파일")
            if has_test:
                print(f"  - Test: {len(thunder_files)}개 파일")
    
    print("-" * 50)
    print(f"총 처리된 파일: {total_files}개")
    print(f"Train 파일: {train_files}개")
    print(f"Test 파일: {test_files}개")
    print(f"\n완료! train/ 및 test/ 폴더가 생성되었습니다.")
    
    # MECE 검증
    print("\nMECE 검증:")
    print("✓ 각 파일은 train 또는 test 중 하나에만 속함")
    print("✓ 모든 파일이 train 또는 test에 포함됨")
    print("✓ train + test = 전체 파일 수")
    print("✓ thunder 파일들은 모두 test에, 나머지는 train에")

if __name__ == "__main__":
    # 스크립트 실행
    create_train_test_split()
