"""
자동화된 이미지 유사도 검색 시스템 (DINOv2-large)
========================================================
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Union, Tuple, Optional
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModel

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ======================= 전역 변수 =======================
global_model = None
global_processor = None
global_device = None

# ==================== 시스템 설정 ====================
TEST_DIR: str = "test"
RESULTS_DIR: str = "results"

FEATURES_NPY: str = "embeddings/train_features_dinov2_large_mean.npy"
PATHS_NPY: str = "embeddings/train_paths_dinov2_large_mean.npy"
MODEL_NAME: str = "facebook/dinov2-large"

TOPK: int = 1
MAX_TEST_IMAGES: Optional[int] = None

ALLOWED_EXTS: set = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==================== 모델 초기화 ====================
def initialize_model_once(model_name=MODEL_NAME):
    global global_model, global_processor, global_device
    if global_model is None:
        print("🔧 DINOv2-large 모델 로드 중...")
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_processor = AutoImageProcessor.from_pretrained(model_name)
        global_model = AutoModel.from_pretrained(model_name)
        global_model.to(global_device).eval()
        print("✅ 모델 로드 완료")
    return global_model, global_processor, global_device

# ==================== 이미지 임베딩 ====================
def embed_image(model, processor, img_path, device):
    try:
        image = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        print(f"⚠️ 이미지 로드 실패: {img_path}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS 토큰 기준 임베딩
        feat = outputs.last_hidden_state[:, 0, :]
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32").flatten()

# ==================== 테스트 이미지 수집 ====================
def collect_test_images(test_dir: str, max_images: Optional[int] = None) -> List[Tuple[str, str]]:
    test_images = []
    if not os.path.isdir(test_dir):
        print(f"❌ {test_dir} 폴더를 찾을 수 없습니다.")
        return []

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTS:
                rel_path = os.path.relpath(root, test_dir)
                folder_name = rel_path if rel_path != "." else "root"
                test_images.append((folder_name, file))
                if max_images and len(test_images) >= max_images:
                    break
        if max_images and len(test_images) >= max_images:
            break

    test_images.sort()
    return test_images

# ==================== 유사도 검색 ====================
def similar_search(query_image_path: str, features: np.ndarray, paths: np.ndarray, 
                  model, processor, device, top_k: int = 5) -> List[Dict[str, Union[int, float, str]]]:
    q = embed_image(model, processor, query_image_path, device)
    if q is None:
        return []

    query_dim = q.shape[0]
    features_dim = features.shape[1]
    if query_dim != features_dim:
        print(f"⚠️ 차원 불일치: 쿼리 {query_dim}, 저장 {features_dim}")
        return []

    # L2 정규화 후 코사인 유사도 계산
    q = q / (np.linalg.norm(q) + 1e-9)
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    sims = features_norm @ q
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for rank, i in enumerate(top_idx, start=1):
        folder_path = str(paths[i])
        folder_name = os.path.basename(folder_path)
        results.append({
            'rank': rank,
            'path': folder_path,
            'folder_name': folder_name,
            'similarity': float(sims[i]),
            'filename': folder_name
        })
    return results

# ==================== 플롯 생성 ====================
def find_first_image_in_folder(folder_path: str) -> Optional[str]:
    if not os.path.isdir(folder_path):
        return None
    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
            return os.path.join(folder_path, filename)
    return None

def create_and_save_plot(query_image_path: str, search_results: List[Dict], folder_name: str, filename: str, figsize=(20,8)) -> str:
    if not search_results:
        return ""
    try:
        query_img = Image.open(query_image_path).convert('RGB')
        result_images = []
        for result in search_results:
            first_img = find_first_image_in_folder(result['path'])
            if first_img:
                result_images.append(Image.open(first_img).convert('RGB'))
            else:
                result_images.append(Image.new('RGB', (224,224), color='gray'))

        total_images = 1 + len(search_results)
        fig, axes = plt.subplots(1, total_images, figsize=figsize)
        fig.suptitle(f'Similarity results: {folder_name}/{filename}', fontsize=16, fontweight='bold')

        # 쿼리 이미지
        ax = axes[0] if total_images > 1 else axes
        ax.imshow(query_img)
        ax.set_title(f'Input image\n{folder_name}', fontsize=12, fontweight='bold', color='blue')
        ax.axis('off')

        # 결과 이미지
        for i, (result, img) in enumerate(zip(search_results, result_images)):
            ax = axes[i+1] if total_images > 1 else axes
            ax.imshow(img)
            title = f"#{result['rank']}\n{result['folder_name']}\n유사도: {result['similarity']:.4f}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            # 테두리 색
            color = 'green' if result['similarity']>0.8 else 'orange' if result['similarity']>0.6 else 'red'
            rect = patches.Rectangle((0,0), img.width-1, img.height-1, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        plt.tight_layout()
        plot_filename = f"{folder_name}_{filename.replace('.', '_')}_similarity.png"
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return plot_path
    except Exception as e:
        print(f"❌ 플롯 생성 실패: {e}")
        return ""

# ==================== 메인 실행 ====================
def run_automated_similarity_search():
    print("🚀 DINOv2-large 자동 유사도 검색 시작")
    test_images = collect_test_images(TEST_DIR, MAX_TEST_IMAGES)
    if not test_images:
        print("❌ test 이미지가 없습니다.")
        return

    try:
        feats = np.load(FEATURES_NPY).astype("float32")
        paths = np.load(PATHS_NPY, allow_pickle=True)
        if feats.ndim != 2 or feats.shape[0] != len(paths):
            raise ValueError("features/paths 불일치")
        print(f"✅ 임베딩 로드 완료: {feats.shape}")
    except Exception as e:
        print(f"❌ 임베딩 로드 실패: {e}")
        return

    model, processor, device = initialize_model_once(MODEL_NAME)
    results = []

    for i, (folder_name, filename) in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] 처리 중: {folder_name}/{filename}")
        full_path = os.path.join(TEST_DIR, folder_name, filename)
        search_results = similar_search(full_path, feats, paths, model, processor, device, TOPK)
        if search_results:
            top_result = search_results[0]
            plot_path = create_and_save_plot(full_path, search_results, folder_name, filename)
            results.append({
                'test_folder': folder_name,
                'test_filename': filename,
                'top_similar_predict': top_result['folder_name'],
                'similarity_score': top_result['similarity'],
                'plot_path': plot_path
            })
        else:
            results.append({
                'test_folder': folder_name,
                'test_filename': filename,
                'top_similar_predict': '검색 실패',
                'similarity_score': 0.0,
                'plot_path': ''
            })

    # CSV 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    csv_filename = f"{MODEL_NAME.replace('/', '_')}_mean_fixed_result.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장 완료: {csv_path}")

if __name__ == "__main__":
    PLOT_DIR = os.path.join(RESULTS_DIR, MODEL_NAME.replace('/', '_'))
    os.makedirs(PLOT_DIR, exist_ok=True)
    run_automated_similarity_search()
