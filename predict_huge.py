"""
자동화된 이미지 유사도 검색 시스템 (OpenCLIP ViT-g-14 버전)
========================================================

입력 스펙:
---------
1. TEST_DIR: "test" 폴더 내의 모든 이미지 파일들
2. FEATURES_NPY: "embeddings/train_features_vit_g_14.npy" - train 폴더 이미지들의 OpenCLIP 임베딩 벡터
3. PATHS_NPY: "embeddings/train_paths_vit_g_14.npy" - train 폴더 이미지들의 파일 경로
4. MODEL_NAME: 'open_clip/ViT-g-14' - OpenCLIP 모델명
5. TOPK: 5 - 검색 결과 상위 개수
6. MAX_TEST_IMAGES: None (모든 이미지 처리) 또는 숫자 (제한된 개수)

출력 스펙:
---------
1. CSV 파일: results/{MODEL_NAME}_result.csv
   - test_folder: 테스트 이미지가 있는 폴더명
   - test_filename: 테스트 이미지 파일명
   - top_similar_predict: 가장 유사한 이미지 파일명
   - similarity_score: 유사도 점수
   - plot_path: 생성된 플롯 파일 경로

2. 플롯 파일: results/{MODEL_NAME}/plot_result/ 폴더에 저장
   - 각 테스트 이미지별 유사도 검색 결과 시각화

사용법:
------
python predict_huge.py  # 모든 test 이미지 처리
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from PIL import Image
import torch
import open_clip

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 시스템 설정 ====================
TEST_DIR: str = "test"                           # 테스트 이미지가 있는 폴더
RESULTS_DIR: str = "results"                     # 결과 저장 폴더

FEATURES_NPY: str = "embeddings/train_features_vit_H_14.npy"  # train 폴더 이미지들의 임베딩 벡터
PATHS_NPY: str = "embeddings/train_paths_vit_H_14.npy"        # train 폴더 이미지들의 파일 경로
MODEL_NAME: str = 'ViT-H-14'           # OpenCLIP 모델명

TOPK: int = 3                                    # 검색 결과 상위 개수
MAX_TEST_IMAGES: Optional[int] = None            # 처리할 최대 테스트 이미지 수 (None이면 모든 이미지)

# ==================== 핵심 함수들 ====================

def load_model(model_name: str, device: torch.device) -> Tuple[open_clip.CLIP, Callable]:
    """
    OpenCLIP 모델과 전처리 변환을 로드합니다.
    
    입력:
        model_name (str): OpenCLIP 모델명 (예: 'open_clip/ViT-g-14')
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        Tuple[open_clip.CLIP, Callable]: OpenCLIP 모델과 전처리 변환 객체
    """
    # OpenCLIP 모델 로드
    model, _, transforms = open_clip.create_model_and_transforms(
        model_name, 
        pretrained="laion2b_s32b_b79k",
        device=device
    )
    model.eval()
    return model, transforms

def embed_image(model: open_clip.CLIP, transforms_obj: Callable, img_path: str, device: torch.device) -> np.ndarray:
    """
    이미지의 OpenCLIP 임베딩 벡터를 계산합니다.
    
    입력:
        model (open_clip.CLIP): OpenCLIP 모델 객체
        transforms_obj (Callable): OpenCLIP 전처리 변환
        img_path (str): 이미지 파일 경로
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        np.ndarray: 임베딩 벡터 (정규화됨)
                   - ViT-g-14: 1024차원
    """
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        # OpenCLIP 전처리 적용
        inputs = transforms_obj(image).unsqueeze(0).to(device)
        
        # OpenCLIP 모델로 이미지 특징 추출
        feat = model.encode_image(inputs)
        
        # L2 정규화 (코사인 유사도 계산을 위해)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    
    return feat.cpu().numpy().astype("float32").flatten()

def collect_test_images(test_dir: str, max_images: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    test 폴더 내의 모든 이미지 파일을 수집합니다.
    
    입력:
        test_dir (str): 테스트 폴더 경로
        max_images (Optional[int]): 최대 수집할 이미지 수 (None이면 모든 이미지)
    
    출력:
        List[Tuple[str, str]]: (폴더명, 파일명) 튜플 리스트
    """
    if not os.path.isdir(test_dir):
        print(f"❌ {test_dir} 폴더를 찾을 수 없습니다.")
        return []
    
    allowed_exts: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    test_images: List[Tuple[str, str]] = []
    
    # test 폴더 내의 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in allowed_exts:
                # 상대 경로 계산
                rel_path = os.path.relpath(root, test_dir)
                if rel_path == ".":
                    folder_name = "root"
                else:
                    folder_name = rel_path
                
                test_images.append((folder_name, file))
                
                # 최대 개수 제한 확인
                if max_images and len(test_images) >= max_images:
                    break
        
        if max_images and len(test_images) >= max_images:
            break
    
    test_images.sort()
    return test_images

def similar_search(query_image_path: str, features: np.ndarray, paths: np.ndarray, 
                  model: open_clip.CLIP, transforms_obj: Callable, device: torch.device, 
                  top_k: int = 5) -> List[Dict[str, Union[int, float, str]]]:
    """
    이미지 유사도 검색을 수행합니다.
    
    입력:
        query_image_path (str): 검색할 이미지 경로
        features (np.ndarray): 모든 이미지의 임베딩 벡터 (N x 차원)
        paths (np.ndarray): 모든 이미지의 파일 경로
        model (open_clip.CLIP): OpenCLIP 모델 객체
        transforms_obj (Callable): OpenCLIP 전처리 변환
        device (torch.device): GPU 또는 CPU 디바이스
        top_k (int): 반환할 상위 결과 개수
    
    출력:
        List[Dict[str, Union[int, float, str]]]: 검색 결과 딕셔너리 리스트
    """
    try:
        # 쿼리 이미지 임베딩 계산
        q: np.ndarray = embed_image(model, transforms_obj, query_image_path, device)
        
        # 차원 확인
        query_dim = q.shape[0]
        features_dim = features.shape[1]
        
        if query_dim != features_dim:
            print(f"⚠️  차원 불일치! 쿼리: {query_dim}차원, 저장된 임베딩: {features_dim}차원")
            return []
        
        # 코사인 유사도 계산을 위한 정규화
        q = q / (np.linalg.norm(q) + 1e-9)
        features_norm: np.ndarray = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
        
        # 유사도 계산 (정규화된 벡터의 내적 = 코사인 유사도)
        sims: np.ndarray = features_norm @ q
        top_idx: np.ndarray = np.argsort(sims)[::-1][:top_k]  # 유사도 내림차순 정렬
        
        # 결과 생성
        results: List[Dict[str, Union[int, float, str]]] = []
        for rank, i in enumerate(top_idx, start=1):
            results.append({
                'rank': rank,
                'path': str(paths[i]),
                'similarity': float(sims[i]),
                'filename': os.path.basename(str(paths[i]))
            })
        
        return results
        
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return []

def create_and_save_plot(query_image_path: str, search_results: List[Dict[str, Union[int, float, str]]], 
                        folder_name: str, filename: str, figsize: Tuple[int, int] = (20, 8)) -> str:
    """
    유사도 검색 결과를 플롯으로 생성하고 저장합니다.
    
    입력:
        query_image_path (str): 쿼리 이미지 경로
        search_results (List[Dict]): 검색 결과 리스트
        folder_name (str): 테스트 폴더명
        filename (str): 테스트 파일명
        figsize (Tuple[int, int]): 그래프 크기
    
    출력:
        str: 저장된 플롯 파일 경로
    """
    if not search_results:
        return ""
    
    try:
        # 쿼리 이미지 로드
        query_img: Image.Image = Image.open(query_image_path).convert('RGB')
        
        # 결과 이미지들 로드
        result_images: List[Image.Image] = []
        for result in search_results:
            try:
                img: Image.Image = Image.open(str(result['path'])).convert('RGB')
                result_images.append(img)
            except Exception as e:
                print(f"⚠️  결과 이미지 로드 실패: {result['path']} - {e}")
                # 빈 이미지로 대체
                result_images.append(Image.new('RGB', (224, 224), color='gray'))
        
        # 1행으로 시각화 (쿼리 이미지 + top k 결과)
        total_images: int = 1 + len(search_results)
        fig, axes = plt.subplots(1, total_images, figsize=figsize)
        fig.suptitle(f'Similarity results: {folder_name}/{filename}', fontsize=16, fontweight='bold')
        
        # 1번째 위치: 쿼리 이미지
        ax = axes[0] if total_images > 1 else axes
        ax.imshow(query_img)
        
        # 쿼리 이미지 경로에서 폴더명 추출
        query_path_parts = query_image_path.replace('\\', '/').split('/')
        query_folder_name = query_path_parts[-2] if len(query_path_parts) > 1 else "알 수 없음"
        
        ax.set_title(f'Input image\n{query_folder_name}', fontsize=12, fontweight='bold', color='blue')
        ax.axis('off')
        
        # 2번째 위치부터: 결과 이미지들
        for i, (result, img) in enumerate(zip(search_results, result_images)):
            ax = axes[i + 1] if total_images > 1 else axes
            
            ax.imshow(img)
            
            # 제목에 순위, 파일명, 유사도 표시
            title: str = f"#{result['rank']}\n{result['filename']}\n유사도: {result['similarity']:.4f}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # 유사도에 따른 테두리 색상 설정
            if float(result['similarity']) > 0.8:
                color: str = 'green'      # 높은 유사도
            elif float(result['similarity']) > 0.6:
                color: str = 'orange'     # 중간 유사도
            else:
                color: str = 'red'        # 낮은 유사도
            
            # 테두리 추가
            rect = patches.Rectangle((0, 0), img.width-1, img.height-1, 
                                    linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        plt.tight_layout()
        
        # 플롯 저장
        plot_filename = f"{folder_name}_{filename.replace('.', '_')}_similarity.png"
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 메모리 해제
        
        return plot_path
        
    except Exception as e:
        print(f"❌ 플롯 생성 실패: {e}")
        return ""

def run_automated_similarity_search() -> None:
    """
    전체 test 데이터셋에 대해 자동화된 유사도 검색을 수행합니다.
    """
    print("🚀 자동화된 유사도 검색 시작")
    print(f"📁 테스트 폴더: {TEST_DIR}")
    print(f"🔍 최대 테스트 이미지: {MAX_TEST_IMAGES if MAX_TEST_IMAGES else '모든 이미지'}")
    print(f"📊 유사도 검색 결과 수: {TOPK}")
    print("=" * 80)
    
    # 1. test 폴더 이미지 수집
    print("📂 test 폴더 이미지 수집 중...")
    test_images = collect_test_images(TEST_DIR, MAX_TEST_IMAGES)
    
    if not test_images:
        print("❌ test 폴더에서 이미지를 찾을 수 없습니다.")
        return
    
    if MAX_TEST_IMAGES:
        print(f"⚠️  최대 {MAX_TEST_IMAGES}개 이미지로 제한합니다.")
    
    print(f"✅ {len(test_images)}개의 이미지 발견")
    
    # 2. 임베딩 파일 로드
    print("📂 임베딩 파일 로드 중...")
    try:
        feats: np.ndarray = np.load(FEATURES_NPY).astype("float32")
        paths: np.ndarray = np.load(PATHS_NPY, allow_pickle=True)
        
        if feats.ndim != 2 or feats.shape[0] != len(paths):
            raise ValueError("features/paths 파일 크기가 일치하지 않습니다.")
        
        print(f"✅ 임베딩 파일 로드 완료: {feats.shape[0]}개 벡터, {feats.shape[1]}차원")
        
    except Exception as e:
        print(f"❌ 임베딩 파일 로드 실패: {e}")
        print("   python embeddings_huge.py를 먼저 실행하세요.")
        return
    
    # 3. 모델 로드
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    try:
        print("🔧 OpenCLIP 모델 로드 중...")
        print(f"   모델: {MODEL_NAME}")
        model, transforms = load_model(MODEL_NAME, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 4. 각 테스트 이미지에 대해 유사도 검색 수행
    print(f"🔍 {len(test_images)}개 이미지에 대해 유사도 검색 수행 중...")
    
    results: List[Dict[str, str]] = []
    
    for i, (folder_name, filename) in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] 처리 중: {folder_name}/{filename}")
        
        # 전체 경로 구성
        full_path = os.path.join(TEST_DIR, folder_name, filename)
        
        try:
            # 유사도 검색 수행
            search_results = similar_search(full_path, feats, paths, model, transforms, device, TOPK)
            
            if search_results:
                # 가장 유사한 이미지 정보 추출
                top_result = search_results[0]
                top_similar_path = str(top_result['path'])
                similarity_score = top_result['similarity']
                
                # 경로에서 폴더명 추출 (top_similar_predict에 폴더명 저장)
                top_path_parts = top_similar_path.replace('\\', '/').split('/')
                top_folder_name = top_path_parts[-2] if len(top_path_parts) > 1 else "알 수 없음"
                
                # 플롯 생성 및 저장
                plot_path = create_and_save_plot(full_path, search_results, folder_name, filename)
                
                # 결과 저장
                results.append({
                    'test_folder': folder_name,
                    'test_filename': filename,
                    'top_similar_predict': top_folder_name,
                    'similarity_score': similarity_score,
                    'plot_path': plot_path
                })
                
                print(f"   ✅ 유사도: {similarity_score:.4f} -> {top_folder_name}")
            else:
                print(f"   ❌ 검색 결과 없음")
                results.append({
                    'test_folder': folder_name,
                    'test_filename': filename,
                    'top_similar_predict': '검색 실패',
                    'similarity_score': 0.0,
                    'plot_path': ''
                })
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            results.append({
                'test_folder': folder_name,
                'test_filename': filename,
                'top_similar_predict': f'오류: {str(e)[:50]}',
                'similarity_score': 0.0,
                'plot_path': ''
            })
    
    # 5. 결과를 CSV 파일로 저장
    print("\n📊 결과 저장 중...")
    
    # 결과 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame(results)
    
    # CSV 파일명 생성 (모델명에서 특수문자 제거)
    csv_filename = f"{MODEL_NAME.replace('/', '_').replace('-', '_')}_result.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    
    # CSV 저장
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 출력용 DataFrame (플롯 경로 제외)
    output_df = df[['test_folder', 'test_filename', 'top_similar_predict', 'similarity_score']].copy()
    
    print(f"✅ 결과 저장 완료: {csv_path}")
    print(f"📊 총 {len(output_df)}개 이미지 처리 완료")
    print(f"   - 성공: {len(output_df[output_df['top_similar_predict'] != '검색 실패'])}")
    print(f"   - 실패: {len(output_df[output_df['top_similar_predict'] == '검색 실패'])}")
    print(f"   - plot 생성 완료: {len(df[df['plot_path'] != ''])}")
    print(f"   - plot 저장 경로: {PLOT_DIR}")
    
    # 상위 10개 결과 미리보기
    print(f"\n📋 상위 10개 결과 미리보기:")
    print("=" * 80)
    for i, row in output_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['test_folder']}/{row['test_filename']}")
        print(f"    → {row['top_similar_predict']} (유사도: {row['similarity_score']:.4f})")
        print("-" * 60)

# ==================== 메인 실행 부분 ====================

if __name__ == "__main__":
    # 플롯 저장 디렉토리 설정
    PLOT_DIR = os.path.join(RESULTS_DIR, MODEL_NAME.replace('/', '_').replace('-', '_'))
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 자동화된 유사도 검색 실행
    run_automated_similarity_search()
