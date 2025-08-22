"""
이미지 유사도 검색 시스템 (DINOv2-large 버전)
================================================

입력 스펙:
---------
1. INPUT_DIR: "test" 폴더에 검색할 이미지 파일들 (.jpg, .jpeg, .png, .webp, .bmp)
2. USED_DIR: "train" 폴더에 비교 대상 이미지들
3. FEATURES_NPY: "embeddings/train_features_dinov2_large_mean.npy" - train 폴더 이미지들의 임베딩 벡터
4. PATHS_NPY: "embeddings/train_paths_dinov2_large_mean.npy" - train 폴더 이미지들의 파일 경로
5. PRODUCTS_CSV: "records/carbot_data_final.csv" - 상품 정보 (상품명, 정가, 중고가, 링크)
6. MODEL_NAME: 'facebook/dinov2-large' - DINOv2 모델명
7. TOPK: 5 - 검색 결과 상위 개수

출력 스펙:
---------
1. 시각화: 쿼리 이미지 + top k 유사 이미지들을 1행으로 표시
2. 텍스트: 각 이미지의 순위, 파일명, 폴더명, 유사도, 정가, 중고가, 링크
3. 반환값: 검색 결과 딕셔너리 리스트 (return_results=True일 때)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 충돌 방지

import time
import gc
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 시스템 설정 ====================
INPUT_DIR = "test"
USED_DIR = "train"

FEATURES_NPY = "embeddings/train_features_dinov2_large_mean.npy"
PATHS_NPY = "embeddings/train_paths_dinov2_large_mean.npy"
PRODUCTS_CSV = "records/carbot_data_final.csv"

MODEL_NAME = 'facebook/dinov2-large'
TOPK = 10

# ==================== 모델 초기화 ====================
global_model = None
global_processor = None
global_device = None

def initialize_model_once(model_name=MODEL_NAME):
    global global_model, global_processor, global_device
    if global_model is None:
        print(f"🔍 DINOv2-large 모델 로드 중: {model_name}")
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_processor = AutoImageProcessor.from_pretrained(model_name)
        global_model = AutoModel.from_pretrained(model_name).to(global_device)
        global_model.eval()
        print("✅ 모델 로드 완료")
    return global_model, global_processor, global_device

# ==================== 이미지 임베딩 ====================
def embed_image(model, processor, img_path, device):
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"⚠️ 이미지 로드 실패: {img_path}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32").flatten()

# ==================== input 폴더 이미지 수집 ====================
def list_input_images() -> List[str]:
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ {INPUT_DIR} 폴더를 찾을 수 없습니다.")
        return []
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in allowed_exts]
    images.sort()
    return images

# ==================== 상품 정보 로드 ====================
def load_products_info() -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(PRODUCTS_CSV):
        print(f"⚠️  {PRODUCTS_CSV} 파일을 찾을 수 없습니다.")
        return {}
    df = pd.read_csv(PRODUCTS_CSV)
    products_dict = {}
    for _, row in df.iterrows():
        product_name = str(row['product_name']).strip()
        products_dict[product_name] = {
            'retail_price': row['retail_price'],
            'used_price_avg': row['used_price_avg'],
            'retail_link': str(row['retail_link']).strip()
        }
    print(f"✅ {len(products_dict)}개의 상품 정보를 로드했습니다.")
    return products_dict

def get_product_info_from_path(image_path: str, products_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # 경로에서 폴더명 추출
    path_parts = image_path.replace('\\', '/').split('/')
    folder_name = path_parts[-2] if len(path_parts) > 1 else ""
    
    # 폴더명에서 상품명 추출 (헬로카봇_ 접두사 제거하고 _를 띄어쓰기로 변환)
    product_name = folder_name[6:] if folder_name.startswith("헬로카봇_") else folder_name
    product_name_clean = product_name.replace('_', ' ')
    
    # 1단계: 정확한 매칭 (헬로카봇 + 상품명)
    exact_match_key = f"헬로카봇 {product_name_clean}"
    if exact_match_key in products_dict:
        return products_dict[exact_match_key]
    
    # 2단계: 폴더명 원본으로 시도
    if folder_name in products_dict:
        return products_dict[folder_name]
    
    # 3단계: 부분 매칭
    best_match = None
    best_score = 0
    for key, info in products_dict.items():
        if product_name_clean in key or product_name in key:
            score = len(product_name_clean) / len(key)
            if score > best_score:
                best_score = score
                best_match = (key, info)
    if best_match:
        return best_match[1]
    
    # 4단계: 매칭 실패시 기본값
    return {
        'retail_price': '가격 정보 없음',
        'used_price_avg': '중고가 정보 없음',
        'retail_link': '링크 없음'
    }

# ==================== 유사도 검색 ====================
def search_similar_images(query_image_path, features, paths, model, processor, device, top_k=5):
    q = embed_image(model, processor, query_image_path, device)
    if q is None:
        return [], 0.0, 0.0
    q = q / (np.linalg.norm(q)+1e-9)
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True)+1e-9)
    sims = features_norm @ q
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank,i in enumerate(top_idx,1):
        results.append({'rank':rank,'path':str(paths[i]),'similarity':float(sims[i]),'filename':os.path.basename(str(paths[i]))})
    return results, 0.0, 0.0

# ==================== 시각화 ====================
def visualize_search_results_with_query(query_image_path, results, products_dict, figsize=(20,8)):
    if not results: return
    query_img = Image.open(query_image_path).convert('RGB')
    result_images=[]
    for r in results:
        try: img=Image.open(r['path']).convert('RGB')
        except: img=Image.new('RGB',(224,224),'gray')
        result_images.append(img)
    
    total_images = 1 + len(results)
    fig,axes=plt.subplots(1,total_images,figsize=figsize)
    axes=axes if total_images>1 else [axes]
    axes[0].imshow(query_img)
    query_folder_name = query_image_path.replace('\\','/').split('/')[-2]
    axes[0].set_title(f"Input image\n{query_folder_name}",fontsize=14,color='blue'); axes[0].axis('off')
    
    for i,(r,img) in enumerate(zip(results,result_images)):
        axes[i+1].imshow(img)
        price_info=get_product_info_from_path(r['path'],products_dict)
        retail_price=price_info['retail_price'] if price_info['retail_price']!='가격 정보 없음' else 'N/A'
        used_price=price_info['used_price_avg'] if price_info['used_price_avg']!='중고가 정보 없음' else 'N/A'
        axes[i+1].set_title(f"#{r['rank']}\n{r['filename']}\n유사도:{r['similarity']:.4f}\n정가:{retail_price}\n중고가:{used_price}",fontsize=10)
        axes[i+1].axis('off')
        color='green' if r['similarity']>0.8 else 'orange' if r['similarity']>0.6 else 'red'
        axes[i+1].add_patch(patches.Rectangle((0,0),img.width-1,img.height-1,linewidth=3,edgecolor=color,facecolor='none'))
    plt.tight_layout(); plt.show()

# ==================== 메인 검색 함수 ====================

def get_valid_image_path(top_image_path):
    """
    top1 이미지가 thunder_로 시작하면, 
    같은 폴더 내에서 thunder_가 아닌 다음 이미지를 선택
    """
    folder = os.path.dirname(top_image_path)
    files = sorted(os.listdir(folder))
    base_name = os.path.basename(top_image_path)
    
    # top1이 thunder_로 시작하면 다음 후보 찾기
    if base_name.startswith("thunder_"):
        for f in files:
            if not f.startswith("thunder_"):
                return os.path.join(folder, f)
        # 없으면 원래 이미지 반환
        return top_image_path
    else:
        return top_image_path

def search_by_image_name(image_name, return_results=False):
    query_image_path = os.path.join(INPUT_DIR, image_name)
    if not os.path.isfile(query_image_path):
        print(f"❌ 이미지 없음: {query_image_path}")
        return None

    if not os.path.isfile(FEATURES_NPY) or not os.path.isfile(PATHS_NPY):
        print("❌ 임베딩 파일 먼저 생성 필요")
        return None

    products_dict = load_products_info()
    feats = np.load(FEATURES_NPY).astype('float32')
    paths = np.load(PATHS_NPY, allow_pickle=True)

    model, transforms, device = initialize_model_once()  # processor → transforms

    results, embedding_time, similarity_time = search_similar_images(
        query_image_path, feats, paths, model, transforms, device, TOPK
    )

    if results:
        # top1 이미지 경로 선택 + thunder_ 처리
        # results[0]['path']가 폴더 경로만 저장되어 있으므로, 해당 폴더에서 thunder_로 시작하지 않는 파일을 찾음
        folder_path = results[0]['path']
        if os.path.isdir(folder_path):
            # 폴더 내 파일들을 확인하여 thunder_로 시작하지 않는 파일을 찾음
            files = sorted(os.listdir(folder_path))
            valid_file = None
            for f in files:
                if not f.startswith("thunder_"):
                    valid_file = f
                    break
            if valid_file:
                top1_path = os.path.join(folder_path, valid_file)
            else:
                # thunder_로 시작하지 않는 파일이 없으면 첫 번째 파일 사용
                top1_path = os.path.join(folder_path, files[0]) if files else folder_path
        else:
            top1_path = get_valid_image_path(results[0]['path'])
        
        # 폴더명 추출 (헬로카봇_ 접두사 제거하고 _를 띄어쓰기로 변환)
        # top1_path에서 실제 폴더명을 추출 (train\헬로카봇_라이캅스\ricops_001.png 형태)
        # 폴더명 추출 (헬로카봇_ 접두사 제거하고 _를 띄어쓰기로 변환)
        # top1_path에서 실제 폴더명을 추출 (train\헬로카봇_라이캅스\ricops_001.png 형태)
        # 폴더명 추출 (_를 띄어쓰기로 변환)
        path_parts = top1_path.replace('\\', '/').split('/')
        folder_name = path_parts[1] if len(path_parts) > 1 else ""
        
        # _를 띄어쓰기로 변환
        folder_name = folder_name.replace('_', ' ')
        
        # similar_toy_name은 folder_name 그대로 사용 (이미 "헬로카봇 라이캅스" 형태)
        similar_toy_name = folder_name
        
        # 디버깅용 출력
        print(f"DEBUG: path_parts = {path_parts}")
        print(f"DEBUG: folder_name = {folder_name}")
        print(f"DEBUG: similar_toy_name = {similar_toy_name}")
        print(f"DEBUG: top1_path = {top1_path}")
        print(f"DEBUG: results[0]['path'] = {results[0]['path']}")
        print(f"DEBUG: folder_path = {folder_path}")
        
        # 디버깅용 출력
        print(f"DEBUG: path_parts = {path_parts}")
        print(f"DEBUG: folder_name = {folder_name}")
        print(f"DEBUG: similar_toy_name = {similar_toy_name}")
        print(f"DEBUG: top1_path = {top1_path}")
        print(f"DEBUG: results[0]['path'] = {results[0]['path']}")
        print(f"DEBUG: folder_path = {folder_path}")
        
        # 가격 정보
        price_info = get_product_info_from_path(top1_path, products_dict)

        top1_json = {
            "similar_toy_name": similar_toy_name,
            "similar_image_path": top1_path,
            "similar_retail_price": price_info['retail_price'] if price_info['retail_price'] != '가격 정보 없음' else 0,
            "similar_used_price": int(price_info['used_price_avg']) if price_info['used_price_avg'] != '중고가 정보 없음' else 0
        }

        if return_results:
            return top1_json
        else:
            visualize_search_results_with_query(query_image_path, results, products_dict)
            return None

# ==================== 실행 예시 ====================
if __name__=="__main__":
    t0=time.time()
    print("🚀 DINOv2-large 모델을 사용한 이미지 유사도 검색 시작")
    # result = search_by_image_name("헬로카봇_라이캅스/thunder_1213.webp", return_results=True)
    # result = search_by_image_name("헬로카봇_로드세이버/thunder_1131.webp", return_results=True)
    result = search_by_image_name("헬로카봇_호크X_빅큐브/thunder_0856.webp", return_results=True)
    print("🔍 검색 결과 JSON:")
    print(result)
    print(f"🎯 총 소요 시간: {time.time()-t0:.3f}초")
