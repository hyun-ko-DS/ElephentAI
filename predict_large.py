# ai_pricer_sameitem_price.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 라이브러리 충돌 방지
from typing import Dict, Optional, Tuple, List
import pandas as pd

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from transformers import CLIPProcessor, CLIPModel

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ======================= 전역 변수 (모델 사전 로드용) =======================
global_model = None
global_processor = None
global_device = None

# ======================= 사용자 설정 =======================
# 테스트할 이미지 개수 제한 (None이면 모든 이미지 처리)
MAX_TEST_IMAGES = None  # 예: 10으로 설정하면 처음 10개만 테스트

FEATURES_NPY: str = "embeddings/train_features_large_patch14.npy"  # train 폴더 이미지들의 임베딩 벡터
PATHS_NPY: str = "embeddings/train_paths_large_patch14.npy"        # train 폴더 이미지들의 파일 경로
MODEL_NAME: str = 'openai/clip-vit-large-patch14' 
TOPK         = 3   # 유사도 검색 결과 개수

# test 폴더 경로
TEST_DIR = "test"

# 결과 저장 경로
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# plot 결과 저장 경로
PLOT_DIR = os.path.join(RESULTS_DIR, MODEL_NAME.replace('/', '_').replace('-', '_'), "plot_result")
os.makedirs(PLOT_DIR, exist_ok=True)


# ----------------------- CLIP 임베딩 (모델 사전 로드) -----------------------
def initialize_model_once(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    모델을 한 번만 로드하고 이후에는 재사용합니다.
    전역 변수를 사용하여 메모리 효율성을 높입니다.
    """
    global global_model, global_processor, global_device
    
    if global_model is None:
        print("🔧 모델을 한 번만 로드합니다...")
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Device: {global_device}")
        
        global_model, global_processor = load_model(model_name, global_device)
        print("✅ 모델 로드 완료 (이제 재사용됩니다)")
    else:
        print("♻️  이미 로드된 모델을 재사용합니다")
    
    return global_model, global_processor

def load_model(model_name: str, device: torch.device):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

def embed_image(model, processor, img_path, device) -> np.ndarray:
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32").flatten()


# ----------------------- 유사도 검색 (모델 재사용) -----------------------
def similar_search(query_img: str, feats_npy: str, paths_npy: str,
                   model_name: str, topk: int=6):
    if not os.path.isfile(query_img): raise FileNotFoundError(f"IMAGE_PATH not found: {query_img}")
    if not os.path.isfile(feats_npy) or not os.path.isfile(paths_npy):
        raise FileNotFoundError("features/paths npy 필요")

    feats = np.load(feats_npy).astype("float32")
    paths = np.load(paths_npy, allow_pickle=True)
    if feats.ndim!=2 or feats.shape[0]!=len(paths): raise ValueError("features/paths 크기 불일치")

    # 모델을 한 번만 로드하고 재사용
    model, processor = initialize_model_once(model_name)

    q = embed_image(model, processor, query_img, global_device)
    q = q / (np.linalg.norm(q)+1e-9)
    feats = feats / (np.linalg.norm(feats,axis=1,keepdims=True)+1e-9)

    sims = feats @ q
    top_idx = np.argsort(sims)[::-1][:topk]

    entries=[]
    for rank,i in enumerate(top_idx, start=1):
        p = str(paths[i])
        entries.append({
            "rank": rank,
            "path": p,                                  # 원래 경로 문자열
            "filename": os.path.basename(p),            # basename만 추출
            "sim": float(sims[i])
        })
    return entries

# ----------------------- plot 생성 및 저장 -----------------------
def create_and_save_plot(query_image_path: str, search_results: List[Dict], 
                        folder_name: str, filename: str, figsize: Tuple[int, int] = (20, 8)) -> str:
    """
    검색 결과를 plot으로 생성하고 파일로 저장합니다.
    
    입력:
        query_image_path (str): 쿼리 이미지 경로
        search_results (List[Dict]): 검색 결과 리스트
        folder_name (str): 테스트 폴더명
        filename (str): 테스트 파일명
        figsize (Tuple[int, int]): 그래프 크기
    
    출력:
        str: 저장된 plot 파일 경로
    """
    if not search_results:
        return ""
    
    # 쿼리 이미지 로드
    try:
        query_img: Image.Image = Image.open(query_image_path).convert('RGB')
    except Exception as e:
        print(f"   ⚠️  쿼리 이미지 로드 실패: {e}")
        return ""
    
    # 결과 이미지들 로드
    result_images: List[Image.Image] = []
    for result in search_results:
        try:
            img: Image.Image = Image.open(str(result['path'])).convert('RGB')
            result_images.append(img)
        except Exception as e:
            print(f"   ⚠️  결과 이미지 로드 실패: {result['path']} - {e}")
            # 빈 이미지로 대체
            result_images.append(Image.new('RGB', (224, 224), color='gray'))
    
    # 1행으로 시각화 (쿼리 이미지 + top k 결과)
    total_images: int = 1 + len(search_results)  # 쿼리 이미지 1개 + 결과 이미지들
    fig, axes = plt.subplots(1, total_images, figsize=figsize)
    fig.suptitle(f'Similarity results: {folder_name}/{filename}', fontsize=20, fontweight='bold')
    
    # 1번째 위치: 쿼리 이미지
    ax = axes[0] if total_images > 1 else axes
    ax.imshow(query_img)
    
    # 쿼리 이미지 경로에서 폴더명 추출
    query_path_parts = query_image_path.replace('\\', '/').split('/')
    query_folder_name = query_path_parts[-2] if len(query_path_parts) > 1 else "알 수 없음"
    
    ax.set_title(f'Input image\n{query_folder_name}', fontsize=14, fontweight='bold', color='blue')
    ax.axis('off')
    
    # 2번째 위치부터: 결과 이미지들
    for i, (result, img) in enumerate(zip(search_results, result_images)):
        ax = axes[i + 1] if total_images > 1 else axes
        
        ax.imshow(img)
        
        # 제목에 순위, 파일명, 유사도 표시
        title: str = f"#{result['rank']}\n{result['filename']}\n유사도: {result['sim']:.4f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 유사도에 따른 테두리 색상 설정
        if float(result['sim']) > 0.8:
            color: str = 'green'      # 높은 유사도
        elif float(result['sim']) > 0.6:
            color: str = 'orange'     # 중간 유사도
        else:
            color: str = 'red'        # 낮은 유사도
        
        # 테두리 추가
        rect = patches.Rectangle((0, 0), img.width-1, img.height-1, 
                                linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    plt.tight_layout()
    
    # plot 파일로 저장
    plot_filename = f"{folder_name}_{filename.replace('.', '_')}_similarity.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 메모리 정리
        return plot_path
    except Exception as e:
        print(f"   ⚠️  plot 저장 실패: {e}")
        plt.close(fig)
        return ""

# ----------------------- test 폴더 이미지 수집 -----------------------
def collect_test_images(test_dir: str, max_images: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    test 폴더 내의 모든 이미지 파일을 수집합니다.
    
    입력:
        test_dir (str): test 폴더 경로
        max_images (Optional[int]): 최대 처리할 이미지 개수 (None이면 모든 이미지)
    
    출력:
        List[Tuple[str, str]]: [(폴더명, 파일명), ...] 형태의 리스트
    """
    if not os.path.isdir(test_dir):
        print(f"❌ {test_dir} 폴더를 찾을 수 없습니다.")
        return []
    
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    test_images = []
    
    # test 폴더 내의 모든 하위 폴더와 파일을 스캔
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in allowed_exts:
                # 상대 경로로 폴더명과 파일명 추출
                rel_path = os.path.relpath(root, test_dir)
                if rel_path == ".":
                    folder_name = "root"
                else:
                    folder_name = rel_path.replace("\\", "/")
                
                test_images.append((folder_name, file))
                
                # 최대 개수 제한 확인
                if max_images and len(test_images) >= max_images:
                    print(f"⚠️  최대 {max_images}개 이미지로 제한합니다.")
                    break
        
        if max_images and len(test_images) >= max_images:
            break
    
    # 정렬하여 일관된 결과 보장
    test_images.sort()
    return test_images

# ----------------------- 자동화된 유사도 검색 및 결과 저장 -----------------------
def run_automated_similarity_search():
    """
    test 폴더의 모든 이미지에 대해 자동으로 유사도 검색을 수행하고 결과를 CSV로 저장합니다.
    """
    import time
    
    print("🚀 자동화된 유사도 검색 시작")
    print(f"📁 테스트 폴더: {TEST_DIR}")
    print(f"🔍 최대 테스트 이미지: {MAX_TEST_IMAGES or '모든 이미지'}")
    print(f"📊 유사도 검색 결과 수: {TOPK}")
    
    # 전체 실행 시간 측정 시작
    total_start_time = time.time()
    
    # 1. test 폴더 이미지 수집
    print("\n📂 test 폴더 이미지 수집 중...")
    test_images = collect_test_images(TEST_DIR, MAX_TEST_IMAGES)
    
    if not test_images:
        print("❌ test 폴더에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"✅ {len(test_images)}개의 이미지 발견")
    
    # 2. 임베딩 파일 확인 및 로드
    if not os.path.isfile(FEATURES_NPY) or not os.path.isfile(PATHS_NPY):
        print(f"❌ 임베딩 파일을 찾을 수 없습니다: {FEATURES_NPY}, {PATHS_NPY}")
        print("   먼저 python embeddings_train.py를 실행하여 임베딩을 생성하세요.")
        return
    
    try:
        print("📂 임베딩 파일 로드 중...")
        feats = np.load(FEATURES_NPY).astype("float32")
        paths = np.load(PATHS_NPY, allow_pickle=True)
        
        if feats.ndim != 2 or feats.shape[0] != len(paths):
            raise ValueError("features/paths 파일 크기가 일치하지 않습니다.")
        
        print(f"✅ 임베딩 파일 로드 완료: {feats.shape[0]}개 벡터, {feats.shape[1]}차원")
            
    except Exception as e:
        print(f"❌ 임베딩 파일 로드 실패: {e}")
        return
    
    # 3. CLIP 모델 로드 (한 번만)
    print(f"\n🔧 CLIP 모델 초기화 중...")
    model_load_start = time.time()
    
    try:
        model, processor = initialize_model_once(MODEL_NAME)
        model_load_time = time.time() - model_load_start
        print(f"✅ 모델 초기화 완료 (소요 시간: {model_load_time:.3f}초)")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 4. 각 테스트 이미지에 대해 유사도 검색 수행
    results = []
    search_start_time = time.time()
    print(f"\n🔍 {len(test_images)}개 이미지에 대해 유사도 검색 수행 중...")
    
    for i, (folder_name, filename) in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] 처리 중: {folder_name}/{filename}")
        
        # 전체 경로 구성
        full_path = os.path.join(TEST_DIR, folder_name, filename)
        
        try:
            # 유사도 검색 수행
            search_results = similar_search(full_path, FEATURES_NPY, PATHS_NPY, MODEL_NAME, TOPK)
            
            if search_results:
                # 가장 유사한 이미지 정보 추출
                top_result = search_results[0]
                top_similar_path = str(top_result['path'])
                similarity_score = top_result['sim']
                
                # 경로에서 폴더명 추출 (top_similar_predict에 폴더명 저장)
                top_path_parts = top_similar_path.replace('\\', '/').split('/')
                top_folder_name = top_path_parts[-2] if len(top_path_parts) > 1 else "알 수 없음"
                
                print(f"   ✅ Top1 유사도: {top_folder_name} (유사도: {similarity_score:.4f})")
                
                # plot 생성 및 저장
                plot_path = create_and_save_plot(full_path, search_results, folder_name, filename)
                if plot_path:
                    print(f"   📊 plot 저장 완료: {os.path.basename(plot_path)}")
                
                # 결과 저장
                results.append({
                    'test_folder': folder_name,
                    'test_filename': filename,
                    'top_similar_predict': top_folder_name,
                    'similarity_score': similarity_score,
                    'plot_path': plot_path
                })
            else:
                print(f"   ❌ 검색 결과 없음")
                results.append({
                    'test_folder': folder_name,
                    'test_filename': filename,
                    'top_similar_predict': 'N/A',
                    'top_similarity': 0.0,
                    'full_path': full_path,
                    'plot_path': ''
                })
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            results.append({
                'test_folder': folder_name,
                'test_filename': filename,
                'top_similar_predict': 'ERROR',
                'top_similarity': 0.0,
                'full_path': full_path,
                'plot_path': ''
            })
    
    search_time = time.time() - search_start_time
    
    # 5. 결과를 CSV로 저장
    if results:
        # MODEL_NAME에서 파일명으로 사용할 부분 추출
        model_name_clean = MODEL_NAME.replace('/', '_').replace('-', '_')
        csv_filename = f"{model_name_clean}_result.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_filename)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(results)
        
        # 요청된 형식으로 컬럼 정리 (similarity_score 포함)
        output_df = df[['test_folder', 'test_filename', 'top_similar_predict', 'similarity_score', 'plot_path']].copy()
        
        try:
            output_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ 결과 저장 완료: {csv_path}")
            print(f"📊 총 {len(results)}개 이미지 처리 완료")
            
            # 전체 실행 시간 계산
            total_time = time.time() - total_start_time
            
            # 결과 요약 출력
            print(f"\n📋 결과 요약:")
            print(f"   - 테스트 폴더 수: {output_df['test_folder'].nunique()}")
            print(f"   - 총 테스트 이미지: {len(output_df)}")
            print(f"   - 성공적으로 처리된 이미지: {len(output_df[output_df['top_similar_predict'] != 'ERROR'])}")
            print(f"   - plot 생성 완료: {len(output_df[output_df['plot_path'] != ''])}")
            print(f"   - plot 저장 경로: {PLOT_DIR}")
            print(f"   - CSV 저장 경로: {csv_path}")
            
            # 시간 분석 출력
            print(f"\n⏱️  시간 분석:")
            print(f"   - 모델 초기화 시간: {model_load_time:.3f}초")
            print(f"   - 검색 처리 시간: {search_time:.3f}초")
            print(f"   - 총 소요 시간: {total_time:.3f}초")
            print(f"   - 이미지당 평균 검색 시간: {search_time/len(test_images):.3f}초")
            
        except Exception as e:
            print(f"❌ CSV 저장 실패: {e}")
    else:
        print("❌ 저장할 결과가 없습니다.")


if __name__ == "__main__":
    # 자동화된 유사도 검색 실행
    run_automated_similarity_search()
