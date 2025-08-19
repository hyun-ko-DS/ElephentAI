"""
이미지 임베딩 생성 시스템
=======================

목적:
    train 폴더의 이미지들을 CLIP 모델로 임베딩하여 벡터화하고, 
    numpy 배열 형태로 저장하여 빠른 유사도 검색을 위한 인덱스를 구축합니다.

입력 스펙:
---------
1. IMAGE_DIR: "train" - 임베딩을 생성할 이미지들이 있는 폴더
2. MODEL_NAME: 'openai/clip-vit-large-patch14' - 사용할 CLIP 모델
3. BATCH_SIZE: 16 - 한 번에 처리할 이미지 개수 (GPU 메모리 효율성)
4. RECURSIVE_SCAN: False - 하위 폴더까지 스캔할지 여부
5. ALLOWED_EXTS: {".jpg", ".jpeg", ".png", ".webp", ".bmp"} - 지원 이미지 확장자

출력 스펙:
---------
1. OUT_FEATURES: "embeddings/train_features_base_patch16.npy" - 이미지 임베딩 벡터들 (N x 768)
2. OUT_PATHS: "embeddings/train_paths_base_patch16.npy" - 이미지 파일 경로들 (N개)

사용법:
------
python embeddings_train.py

주의사항:
---------
- GPU 메모리가 충분해야 합니다 (BATCH_SIZE 조정 필요시)
- train 폴더에 이미지 파일들이 있어야 합니다
- 첫 실행 시 CLIP 모델 다운로드가 필요합니다
"""

import os
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# ==================== 시스템 설정 ====================
IMAGE_DIR: str = "train"                    # 임베딩을 생성할 이미지 폴더
OUT_FEATURES: str = "embeddings/train_features_large_patch14.npy"  # 임베딩 벡터 저장 파일
OUT_PATHS: str = "embeddings/train_paths_large_patch14.npy"        # 이미지 경로 저장 파일

# CLIP 모델 선택 (주석 처리된 모델들도 사용 가능)
# MODEL_NAME: str = "openai/clip-vit-base-patch32"      # 기본 모델 (512차원)
# MODEL_NAME: str = 'openai/clip-vit-base-patch16'      # 기본 모델 (512차원)
MODEL_NAME: str = 'openai/clip-vit-large-patch14'      # 대형 모델 (768차원, 더 정확함)
# MODEL_NAME: str = 'openai/clip-vit-large-patch14-336' # 대형 모델 (768차원, 336x336 해상도)
# MODEL_NAME: str = 'openai/clip-vit-huge-patch14-224' 

BATCH_SIZE: int = 32                       # 배치 크기 (GPU 메모리에 따라 조정)
RECURSIVE_SCAN: bool = True                # 하위 폴더까지 스캔할지 여부
ALLOWED_EXTS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # 지원 이미지 확장자

# ==================== 핵심 함수들 ====================

def list_images(root: str, recursive: bool = True) -> list[str]:
    """
    지정된 폴더에서 이미지 파일들의 경로를 수집합니다.
    
    입력:
        root (str): 이미지를 찾을 루트 폴더 경로
        recursive (bool): 하위 폴더까지 스캔할지 여부
    
    출력:
        list[str]: 이미지 파일들의 절대 경로 리스트 (정렬됨)
    
    예시:
        >>> list_images("used", recursive=False)
        ['used/img1.jpg', 'used/img2.png', ...]
    """
    paths: list[str] = []
    
    if recursive:
        # 하위 폴더까지 재귀적으로 스캔
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
                    full_path: str = os.path.join(dirpath, filename)
                    paths.append(full_path)
    else:
        # 현재 폴더만 스캔
        for filename in os.listdir(root):
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
                full_path: str = os.path.join(root, filename)
                paths.append(full_path)
    
    # 파일명 순으로 정렬하여 일관된 결과 보장
    paths.sort()
    return paths

def load_model(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    """
    CLIP 모델과 프로세서를 로드합니다.
    
    입력:
        model_name (str): Hugging Face에서 제공하는 CLIP 모델명
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        tuple[CLIPModel, CLIPProcessor]: CLIP 모델과 프로세서 객체
    
    예시:
        >>> model, processor = load_model("openai/clip-vit-large-patch14", torch.device("cuda"))
        >>> print(type(model))  # <class 'transformers.models.clip.modeling_clip.CLIPModel'>
    """
    # 사전 훈련된 CLIP 모델 로드
    model: CLIPModel = CLIPModel.from_pretrained(model_name).to(device)
    
    # CLIP 프로세서 로드 (이미지 전처리용)
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
    
    # 추론 모드로 설정 (드롭아웃, 배치 정규화 등 비활성화)
    model.eval()
    
    return model, processor

def embed_batch(model: CLIPModel, processor: CLIPProcessor, 
                pil_images: list[Image.Image], device: torch.device) -> np.ndarray:
    """
    이미지 배치를 CLIP 임베딩 벡터로 변환합니다.
    
    입력:
        model (CLIPModel): 로드된 CLIP 모델
        processor (CLIPProcessor): CLIP 프로세서
        pil_images (list[Image.Image]): PIL Image 객체들의 리스트
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        np.ndarray: 정규화된 임베딩 벡터들 (배치크기 x 차원수)
                   - base 모델: 512차원
                   - large 모델: 768차원
    
    예시:
        >>> images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
        >>> embeddings = embed_batch(model, processor, images, device)
        >>> print(embeddings.shape)  # (2, 768) for large model
    """
    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
        # 이미지들을 CLIP 모델 입력 형식으로 변환
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        
        # CLIP 모델로 이미지 특징 추출
        feats: torch.Tensor = model.get_image_features(**inputs)
        
        # L2 정규화 (코사인 유사도 계산을 위해)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    
    # GPU에서 CPU로 이동하고 numpy 배열로 변환
    return feats.cpu().numpy().astype("float32")

def main() -> None:
    """
    메인 실행 함수: train 폴더의 모든 이미지를 임베딩하여 파일로 저장합니다.
    
    입력: 없음
    
    출력: 없음 (파일로 저장됨)
    
    처리 과정:
    1. train 폴더 존재 확인
    2. GPU/CPU 디바이스 선택
    3. CLIP 모델 로드
    4. 이미지 파일 경로 수집
    5. 배치 단위로 임베딩 생성
    6. 결과를 numpy 배열로 저장
    
    예외 처리:
    - FileNotFoundError: IMAGE_DIR이 존재하지 않을 때
    - RuntimeError: 이미지 파일을 찾을 수 없을 때
    - UnidentifiedImageError: 손상된 이미지 파일
    - OSError: 파일 읽기 오류
    """
    start_time: float = time.time()
    
    # 1. train 폴더 존재 확인
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    
    # 2. GPU/CPU 디바이스 선택
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # 3. CLIP 모델 로드
    print("�� CLIP 모델 로드 중...")
    model, processor = load_model(MODEL_NAME, device)
    print("✅ 모델 로드 완료")
    
    # 4. 이미지 파일 경로 수집
    print("🔍 이미지 파일 스캔 중...")
    img_paths: list[str] = list_images(IMAGE_DIR, recursive=RECURSIVE_SCAN)
    
    if not img_paths:
        raise RuntimeError(f"No images found under: {IMAGE_DIR}")
    
    print(f"�� 총 {len(img_paths)}개의 이미지 파일 발견")
    
    # 5. 배치 단위로 임베딩 생성
    features: list[np.ndarray] = []        # 임베딩 벡터들을 저장할 리스트
    kept_paths: list[str] = []             # 성공적으로 처리된 이미지 경로들
    batch: list[Image.Image] = []          # 현재 배치의 이미지들
    batch_paths: list[str] = []            # 현재 배치의 경로들
    
    print(f"⚡ 임베딩 생성 시작 (batch_size={BATCH_SIZE}, device={device})")
    
    # 진행률 표시와 함께 이미지 처리
    for img_path in tqdm(img_paths, desc="이미지 처리"):
        try:
            # 이미지 로드 및 RGB 변환
            img: Image.Image = Image.open(img_path).convert("RGB")
            batch.append(img)
            batch_paths.append(img_path)
            
            # 배치가 가득 찼을 때 임베딩 생성
            if len(batch) == BATCH_SIZE:
                feats: np.ndarray = embed_batch(model, processor, batch, device)
                features.append(feats)
                kept_paths.extend(batch_paths)
                
                # 배치 초기화
                batch, batch_paths = [], []
                
        except (UnidentifiedImageError, OSError) as e:
            # 손상된 이미지나 읽기 오류 시 건너뛰기
            print(f"⚠️  [SKIP] 로드 실패: {img_path} ({e})")
    
    # 마지막 배치 처리 (BATCH_SIZE보다 적은 경우)
    if batch:
        feats: np.ndarray = embed_batch(model, processor, batch, device)
        features.append(feats)
        kept_paths.extend(batch_paths)
    
    # 6. 모든 임베딩을 하나의 배열로 결합
    if features:
        feats_all: np.ndarray = np.vstack(features)
    else:
        # 이미지가 없는 경우 빈 배열 생성
        embedding_dim: int = 512 if "base" in MODEL_NAME else 768
        feats_all: np.ndarray = np.zeros((0, embedding_dim), dtype="float32")
    
    # 7. 결과를 numpy 배열로 저장
    print("💾 임베딩 저장 중...")
    np.save(OUT_FEATURES, feats_all)
    np.save(OUT_PATHS, np.array(kept_paths, dtype=object))
    
    # 8. 완료 정보 출력
    elapsed_time: float = time.time() - start_time
    print("✅ 인덱스 저장 완료")
    print(f" 📊 features: {OUT_FEATURES} {feats_all.shape}")
    print(f" 📁 paths   : {OUT_PATHS} {len(kept_paths)}개")
    print(f" ⏱️  경과시간: {elapsed_time:.1f}초")
    
    # 9. 성공률 계산
    success_rate: float = (len(kept_paths) / len(img_paths)) * 100
    print(f" 🎯 성공률: {success_rate:.1f}% ({len(kept_paths)}/{len(img_paths)})")

# ==================== 스크립트 실행 ====================

if __name__ == "__main__":
    try:
        main()
        print("\n임베딩 생성이 성공적으로 완료되었습니다!")
        print("이제 train 폴더의 이미지들이 CLIP 임베딩으로 벡터화되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   다음 사항들을 확인해주세요:")
        print("   1. train 폴더가 존재하는지")
        print("   2. train 폴더에 이미지 파일이 있는지")
        print("   3. GPU 메모리가 충분한지 (BATCH_SIZE 조정 필요시)")
        print("   4. 인터넷 연결이 안정적인지 (모델 다운로드용)")