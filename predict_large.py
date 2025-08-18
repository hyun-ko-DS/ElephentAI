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

# ======================= 사용자 설정 =======================
# 테스트할 이미지 개수 제한 (None이면 모든 이미지 처리)
MAX_TEST_IMAGES = None  # 예: 10으로 설정하면 처음 10개만 테스트

FEATURES_NPY = "embeddings/train_features_large_patch14-336.npy"
PATHS_NPY    = "embeddings/train_paths_large_patch14-336.npy"
MODEL_NAME   = "openai/clip-vit-large-patch14-336"
TOPK         = 5   # 유사도 검색 결과 개수

# test 폴더 경로
TEST_DIR = "test"

# 결과 저장 경로
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# plot 결과 저장 경로
PLOT_DIR = os.path.join(RESULTS_DIR, MODEL_NAME.replace('/', '_').replace('-', '_'), "plot_result")
os.makedirs(PLOT_DIR, exist_ok=True)

# CSV에서 파일명→가격 매핑 (현재 사용되지 않음)
# JOONGNA_CSV  = "records/carbot_data_final.csv"
# FILENAME_COL = "thumbnail_filename"
# PRICE_COL    = "price"
# PRODUCT_COL  = "title"   # (선택: 없으면 자동 건너뜀)

# NORMALIZE_FILENAMES = True
# ==========================================================

# ----------------------- 유틸 (현재 사용되지 않음) -----------------------
# def _parse_price_to_int(text: str) -> Optional[int]:
#     if text is None: return None
#     t = str(text).strip().replace(",", "").replace("원", "").replace("KRW", "")
#     if not re.search(r"[0-9]", t): return None
#     try:
#         return int(float(t))
#     except ValueError:
#         digits = "".join(ch for ch in t if ch.isdigit())
#         return int(digits) if digits else None
# 
# def _norm_name(path_or_name: str) -> str:
#     name = os.path.basename(str(path_or_name)).strip()
#     return name.lower() if NORMALIZE_FILENAMES else name
# 
# def load_meta_maps(csv_path, filename_col, price_col, product_col=None) -> Tuple[Dict[str,int], Optional[Dict[str,str]]]:
#     if not os.path.isfile(csv_path):
#         return {}, None
#     encodings = ("utf-8-sig", "utf-8", "cp949")
#     print(f"CSV에 '{filename_col}', '{price_col}' 컬럼이 필요. 현재: {fields}")
#         price_map, title_map = {}, {}
#         has_title = product_col in fields if product_col else False
#         for row in reader:
#             raw = row.get(filename_col, "")
#             fname = _norm_name(raw)
#             if not fname: continue
#             price = _parse_price_to_int(row.get(price_col))
#             if price is not None: price_map[fname] = price
#             if has_title: title_map[fname] = str(row.get(product_col, "")).strip()
#         return price_map, (title_map if has_title else None)
#     except UnicodeDecodeError as e:
#         last_err = e
#         continue
#     raise RuntimeError(f"CSV 인코딩 실패(시도: {encodings}) 원인: {last_err}")

# ----------------------- CLIP 임베딩 -----------------------
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

# ----------------------- 이미지 → Base64 (현재 사용되지 않음) -----------------------
# def _resize_to_jpeg_bytes(img: Image.Image, max_side=1024, enhance=False) -> bytes:
#     w,h = img.size
#     scale = min(1.0, max_side / max(w,h))
#     if scale < 1.0: img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
#     if enhance:
#         img = ImageEnhance.Contrast(img).enhance(1.3)
#         img = ImageEnhance.Sharpness(img).enhance(1.6)
#         img = ImageEnhance.Brightness(img).enhance(1.05)
#     buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92); return buf.getvalue()
# 
# def _path_to_b64(path: str, enhance=False) -> str:
#     img = Image.open(path).convert("RGB")
#     return base64.b64encode(_resize_to_jpeg_bytes(img, enhance=enhance)).decode("utf-8")

# ----------------------- 동일품 판정 프롬프트 (현재 사용되지 않음) -----------------------
# SAME_ITEM_PROMPT = """
# You are a product matcher. For each CANDIDATE photo, decide if it is the SAME PRODUCT/MODEL as the QUERY photo.
# Focus on brand/series/character, mold/shape, printed patterns, colorway, scale/size cues, accessories/parts, packaging text or set ID.
# Do NOT be fooled by pose/angle/lighting. If unsure, answer false.
# 
# Return STRICT JSON ONLY, exactly this schema:
# {"same": [true, true, true]}
# 
# Rules:
# - The array order MUST match the order of the CANDIDATE blocks you receive.
# - "same" means same model/edition (not just same category/character).
# - Variant/limited/colorway/set-ID mismatch => false.
# - STRICT JSON only. No extra text.
# """

# ----------------------- Claude 호출 래퍼 (현재 사용되지 않음) -----------------------
# class SameItemJudgeClaude:
#     def __init__(self, client: Anthropic):
#         self.client = client
# 
#     def judge(self, query_path: str, candidates: List[dict]) -> List[bool]:
#         contents = [{"type":"text","text": SAME_ITEM_PROMPT}]
#         contents.append({"type":"text","text": f"QUERY: {os.path.basename(query_path)}"})
#         f"QUERY: {os.path.basename(query_path)}"})
#         contents.append({"type":"image","source":{
#             "type":"base64","media_type":"image/jpeg","data": _path_to_b64(query_path)
#         }})
# 
#         for c in candidates:
#             meta_line = f"CANDIDATE rank={c['rank']} | file={c['filename']} | sim={c['sim']:.4f}"
#             if c.get("title"): metaify(c['filename']} | sim={c['sim']:.4f}"
#             if c.get("title"): meta_line += f" | title={c['title']}"
#             contents.append({"type":"text","text": meta_line})
#             contents.append({"type":"image","source":{
#                 "type":"base64","media_type":"image/jpeg","data": _path_to_b64(c['resolved_path'])
#             }})
# 
#         msg = self.client.messages.create(
#             model=CLAUDE_MODEL,
#             max_tokens=100,  # 출력이 매우 짧음
#             temperature=0.1,
#             system="Return STRICT JSON. No extra text.",
#             messages=[{"role":"user","content": contents}]
#         )
# 
#         raw = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`").replace("json","",1).strip()
# 
#         try:
#             data = json.loads(raws):
#             same = list(data.get("same", []))
#             same = [bool(x) for x in same][:len(candidates)]
#             if len(same) < len(candidates):
#                 same += [False] * (len(candidates) - len(same))
#         except Exception:
#             same = [False] * len(candidates)
#         return same

# ----------------------- 유사도 검색 -----------------------
def similar_search(query_img: str, feats_npy: str, paths_npy: str,
                   model_name: str, topk: int=6):
    if not os.path.isfile(query_img): raise FileNotFoundError(f"IMAGE_PATH not found: {query_img}")
    if not os.path.isfile(feats_npy) or not os.path.isfile(paths_npy):
        raise FileNotFoundError("features/paths npy 필요")

    feats = np.load(feats_npy).astype("float32")
    paths = np.load(paths_npy, allow_pickle=True)
    if feats.ndim!=2 or feats.shape[0]!=len(paths): raise ValueError("features/paths 크기 불일치")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_name, device)

    q = embed_image(model, processor, query_img, device)
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

# ----------------------- 경로 재해결 (현재 사용되지 않음) -----------------------
# def resolve_to_base_dir(filename: str, base_dir: str) -> str:
#     resolved = os.path.join(base_dir, os.path.basename(filename))
#     if not os.path.isfile(resolved):
#         raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {resolved}")
#     return resolved
# 
# ----------------------- 가격 조회 유틸 (현재 사용되지 않음) -----------------------
# def _try_extension_variants(fname: str, price_map: Dict[str,int]) -> Optional[int]:
#     stem,_ = os.path.splitext(fname)
#     if key in price_map: return price_map[key]
#     return None
# 
# def lookup_price_for_filename(filename: str, price_map: Dict[str,int]) -> Optional[int]:
#     key = _norm_name(filename)
#     if key in price_map: return price_map[key]
#     return _try_extension_variants(key, price_map)
# 
# ----------------------- 중앙값/평균 계산 (현재 사용되지 않음) -----------------------
# def median_of(values: List[int]) -> Optional[float]:
#     if not values: return None
#     xs = sorted(values)
#     n = len(xs)
#     if n % 2 == 1:
#         return float(xs[n//2])
#     else:
#         return (xs[n//2 - 1] + xs[n//2]) / 2.0

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
    print("🚀 자동화된 유사도 검색 시작")
    print(f"📁 테스트 폴더: {TEST_DIR}")
    print(f"🔍 최대 테스트 이미지: {MAX_TEST_IMAGES or '모든 이미지'}")
    print(f"📊 유사도 검색 결과 수: {TOPK}")
    
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
    
    # 3. CLIP 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    try:
        print("🔧 CLIP 모델 로드 중...")
        model, processor = load_model(MODEL_NAME, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 4. 각 테스트 이미지에 대해 유사도 검색 수행
    results = []
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
            
            # 결과 요약 출력
            print(f"\n📋 결과 요약:")
            print(f"   - 테스트 폴더 수: {output_df['test_folder'].nunique()}")
            print(f"   - 총 테스트 이미지: {len(output_df)}")
            print(f"   - 성공적으로 처리된 이미지: {len(output_df[output_df['top_similar_predict'] != 'ERROR'])}")
            print(f"   - plot 생성 완료: {len(output_df[output_df['plot_path'] != ''])}")
            print(f"   - plot 저장 경로: {PLOT_DIR}")
            print(f"   - CSV 저장 경로: {csv_path}")
            
        except Exception as e:
            print(f"❌ CSV 저장 실패: {e}")
    else:
        print("❌ 저장할 결과가 없습니다.")

# ----------------------- 메인 파이프라인 (현재 사용되지 않음) -----------------------
# def run_sameitem_price(image_path: str = None,
#                        feats_npy: str = FEATURES_NPY,
#                        paths_npy: str = PATHS_NPY,
#                        csv_path: str = JOONGNA_CSV,
#                        topk: int = TOPK,
#                        base_dir: str = None):
#     """
#     기존 함수는 유지하되, 이미지 경로가 None이면 자동화된 검색을 수행합니다.
#     """
#     if image_path is None:
#         # 자동화된 유사도 검색 수행
#         run_automated_similarity_search()
#         return
#     
#     # 기존 로직 (단일 이미지 처리)
#     # ... 기존 코드 유지 ...

if __name__ == "__main__":
    # 자동화된 유사도 검색 실행
    run_automated_similarity_search()
