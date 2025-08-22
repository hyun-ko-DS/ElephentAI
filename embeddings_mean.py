"""
이미지 임베딩 생성 시스템 (OpenCLIP ViT-g-14 버전) - 폴더별 평균 임베딩
================================================

목적:
    train 폴더의 각 하위 폴더(클래스)별로 OpenCLIP ViT-g-14 모델로 임베딩을 생성하고,
    각 폴더 내 이미지들의 평균 임베딩을 계산하여 클래스별 대표 벡터를 생성합니다.
    
    결과적으로 각 폴더당 하나의 벡터가 생성되어, 폴더 개수만큼의 임베딩 벡터가 저장됩니다.

입력 스펙:
---------
1. IMAGE_DIR: "train" - 임베딩을 생성할 이미지들이 있는 폴더
2. MODEL_NAME: 'ViT-g-14' - 사용할 OpenCLIP 모델
3. BATCH_SIZE: 8 - 한 번에 처리할 이미지 개수 (GPU 메모리 효율성, g-14는 메모리 사용량이 큼)
4. ALLOWED_EXTS: {".jpg", ".jpeg", ".png", ".webp", ".bmp"} - 지원 이미지 확장자

출력 스펙:
---------
1. OUT_FEATURES: "embeddings/train_features_vit_g_14_mean.npy" - 폴더별 평균 임베딩 벡터들 (N x 1024)
2. OUT_PATHS: "embeddings/train_paths_vit_g_14_mean.npy" - 폴더 경로들 (N개)
    - N은 train 폴더 내 하위 폴더의 개수

사용법:
------
python embeddings_mean.py

주의사항:
---------
- GPU 메모리가 충분해야 합니다 (RTX 4090 권장, BATCH_SIZE 조정 필요시)
- train 폴더에 하위 폴더들이 있어야 합니다
- 첫 실행 시 OpenCLIP 모델 다운로드가 필요합니다
- open_clip_torch 라이브러리가 설치되어 있어야 합니다
"""

import os
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import torch
import open_clip
from typing import Callable, Tuple, Dict, List
from collections import defaultdict

# ==================== 시스템 설정 ====================
IMAGE_DIR: str = "train"                    # 임베딩을 생성할 이미지 폴더
OUT_FEATURES: str = "embeddings/train_features_vit_g_14_mean.npy"  # 폴더별 평균 임베딩 벡터 저장 파일
OUT_PATHS: str = "embeddings/train_paths_vit_g_14_mean.npy"        # 폴더 경로 저장 파일

# OpenCLIP 모델 설정
MODEL_NAME: str = 'ViT-g-14'      # 거대 모델 (1024차원, 매우 정확함)     
PRETRAINED = "laion2b_s34b_b88k"  # 사전 훈련된 가중치

BATCH_SIZE: int = 8                         # 배치 크기 (GPU 메모리에 따라 조정, g-14는 메모리 사용량이 큼)
ALLOWED_EXTS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # 지원 이미지 확장자

# ==================== 핵심 함수들 ====================

def get_folder_structure(root: str) -> Dict[str, List[str]]:
    """
    train 폴더의 하위 폴더 구조를 파악하고, 각 폴더별로 이미지 파일 경로를 수집합니다.
    
    입력:
        root (str): 이미지를 찾을 루트 폴더 경로
    
    출력:
        Dict[str, List[str]]: {폴더경로: [이미지파일경로들]} 형태의 딕셔너리
    
    예시:
        >>> get_folder_structure("train")
        {'train/cat': ['train/cat/img1.jpg', 'train/cat/img2.png'], 
         'train/dog': ['train/dog/img3.jpg', 'train/dog/img4.png']}
    """
    folder_images: Dict[str, List[str]] = defaultdict(list)
    
    # train 폴더의 직접 하위 폴더들만 스캔
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


def load_model(model_name: str, pretrained: str, device: torch.device) -> Tuple[torch.nn.Module, Callable]:
    """
    OpenCLIP 모델과 전처리 변환을 로드합니다.
    FP16 (Half Precision) + TorchScript JIT 컴파일을 적용하여 메모리 사용량을 줄이고 속도를 향상시킵니다.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()
    
    # FP16으로 변환하여 메모리 사용량 감소 및 속도 향상
    if device.type == 'cuda':
        model.half()
        print("✅ FP16 (Half Precision) 적용 완료 - 메모리 사용량 50% 감소, 속도 향상")
        
        # TorchScript JIT 컴파일로 추가 속도 향상
        try:
            # 더미 입력으로 JIT 컴파일 (FP16 모델에 맞춤)
            dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
            model = torch.jit.trace(model, dummy_input)
            print("✅ TorchScript JIT 컴파일 완료 - 추가 10-20% 속도 향상")
        except Exception as e:
            print(f"⚠️  TorchScript 컴파일 실패, 원본 모델 사용: {e}")
    
    return model, preprocess


def embed_batch(model: torch.nn.Module, transforms_obj: Callable, 
                pil_images: list[Image.Image], device: torch.device) -> np.ndarray:
    """
    이미지 배치를 OpenCLIP 임베딩 벡터로 변환합니다.
    FP16 최적화가 적용되어 있습니다.
    
    입력:
        model (open_clip.CLIP): 로드된 OpenCLIP 모델
        transforms_obj (open_clip.transform.Transforms): OpenCLIP 전처리 변환
        pil_images (list[Image.Image]): PIL Image 객체들의 리스트
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        np.ndarray: 정규화된 임베딩 벡터들 (배치크기 x 차원수)
                   - ViT-g-14: 1024차원
    
    예시:
        >>> images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
        >>> embeddings = embed_batch(model, transforms_obj, images, device)
        >>> print(embeddings.shape)  # (2, 1024) for ViT-g-14
    """
    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
        # 이미지들을 OpenCLIP 모델 입력 형식으로 변환
        inputs = torch.stack([transforms_obj(img) for img in pil_images])
        
        # FP16으로 변환하여 메모리 사용량 감소
        if device.type == 'cuda':
            inputs = inputs.half()
        
        inputs = inputs.to(device)
        
        # OpenCLIP 모델로 이미지 특징 추출
        feats: torch.Tensor = model.encode_image(inputs)
        
        # L2 정규화 (코사인 유사도 계산을 위해)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    
    # detach()를 추가하여 그래디언트 계산 그래프에서 분리, 메모리 사용량 감소
    # GPU에서 CPU로 이동하고 numpy 배열로 변환
    return feats.detach().cpu().numpy().astype("float32")


def process_folder_images(model: torch.nn.Module, transforms_obj: Callable, 
                         image_paths: List[str], device: torch.device) -> np.ndarray:
    """
    특정 폴더의 모든 이미지들을 처리하여 평균 임베딩을 계산합니다.
    
    입력:
        model: OpenCLIP 모델
        transforms_obj: 전처리 변환
        image_paths: 해당 폴더의 이미지 파일 경로들
        device: GPU 또는 CPU 디바이스
    
    출력:
        np.ndarray: 해당 폴더의 평균 임베딩 벡터 (1 x 1024)
    """
    if not image_paths:
        return np.zeros((1, 1024), dtype="float32")
    
    all_embeddings = []
    batch = []
    
    # 배치 단위로 이미지 처리
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            batch.append(img)
            
            if len(batch) == BATCH_SIZE:
                feats = embed_batch(model, transforms_obj, batch, device)
                all_embeddings.append(feats)
                batch = []
                
        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️  [SKIP] 로드 실패: {img_path} ({e})")
    
    # 마지막 배치 처리
    if batch:
        feats = embed_batch(model, transforms_obj, batch, device)
        all_embeddings.append(feats)
    
    if not all_embeddings:
        return np.zeros((1, 1024), dtype="float32")
    
    # 모든 임베딩을 하나로 합치고 평균 계산
    all_feats = np.vstack(all_embeddings)
    mean_embedding = np.mean(all_feats, axis=0, keepdims=True)
    
    # 평균 벡터도 정규화
    norm = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
    norm[norm == 0] = 1  # 0으로 나누기 방지
    mean_embedding = mean_embedding / norm
    
    return mean_embedding.astype("float32")


def main() -> None:
    """
    메인 실행 함수: train 폴더의 각 하위 폴더별로 평균 임베딩을 계산하여 파일로 저장합니다.
    
    입력: 없음
    
    출력: 없음 (파일로 저장됨)
    
    처리 과정:
    1. train 폴더 존재 확인
    2. GPU/CPU 디바이스 선택
    3. OpenCLIP 모델 로드
    4. 폴더 구조 파악 및 이미지 파일 경로 수집
    5. 각 폴더별로 평균 임베딩 계산
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
    
    # 3. OpenCLIP 모델 로드
    print("🔧 OpenCLIP 모델 로드 중...")
    print(f"   모델: {MODEL_NAME}")
    print(f"   가중치: {PRETRAINED}")
    model, transforms = load_model(MODEL_NAME, PRETRAINED, device)
    print("✅ 모델 로드 완료")
    
    # 4. 폴더 구조 파악 및 이미지 파일 경로 수집
    print("🔍 폴더 구조 및 이미지 파일 스캔 중...")
    folder_images = get_folder_structure(IMAGE_DIR)
    
    if not folder_images:
        raise RuntimeError(f"No folders with images found under: {IMAGE_DIR}")
    
    folder_paths = list(folder_images.keys())
    folder_paths.sort()  # 폴더 경로를 정렬하여 일관된 결과 보장
    
    total_images = sum(len(images) for images in folder_images.values())
    print(f"📊 총 {len(folder_paths)}개의 폴더 발견")
    print(f"📊 총 {total_images}개의 이미지 파일 발견")
    
    # 5. 각 폴더별로 평균 임베딩 계산
    print(f"⚡ 폴더별 평균 임베딩 계산 시작 (device={device})")
    
    all_mean_embeddings = []
    processed_folders = []
    
    for folder_path in tqdm(folder_paths, desc="폴더별 임베딩 계산"):
        image_paths = folder_images[folder_path]
        print(f"📁 처리 중: {folder_path} ({len(image_paths)}개 이미지)")
        
        # 해당 폴더의 평균 임베딩 계산
        mean_embedding = process_folder_images(model, transforms, image_paths, device)
        all_mean_embeddings.append(mean_embedding)
        processed_folders.append(folder_path)
    
    # 6. 모든 평균 임베딩을 하나의 배열로 결합
    if all_mean_embeddings:
        mean_embeddings_all = np.vstack(all_mean_embeddings)
    else:
        # 폴더가 없는 경우 빈 배열 생성 (ViT-g-14는 1024차원)
        mean_embeddings_all = np.zeros((0, 1024), dtype="float32")
    
    # 7. 결과를 numpy 배열로 저장
    print("💾 폴더별 평균 임베딩 저장 중...")
    np.save(OUT_FEATURES, mean_embeddings_all)
    np.save(OUT_PATHS, np.array(processed_folders, dtype=object))
    
    # 8. 완료 정보 출력
    elapsed_time: float = time.time() - start_time
    print("✅ 폴더별 평균 임베딩 저장 완료")
    print(f" 📊 features: {OUT_FEATURES} {mean_embeddings_all.shape}")
    print(f" 📁 paths   : {OUT_PATHS} {len(processed_folders)}개")
    print(f" ⏱️  경과시간: {elapsed_time:.1f}초")
    
    # 9. 폴더별 이미지 개수 정보 출력
    print(f" 🎯 폴더별 이미지 개수:")
    for folder_path in processed_folders:
        img_count = len(folder_images[folder_path])
        print(f"    - {folder_path}: {img_count}개")
    
    # 10. 모델 정보 출력
    print(f" 🚀 모델 정보:")
    print(f"    - 모델명: {MODEL_NAME}")
    print(f"    - 가중치: {PRETRAINED}")
    print(f"    - 임베딩 차원: {mean_embeddings_all.shape[1] if mean_embeddings_all.shape[0] > 0 else 1024}")
    print(f"    - 배치 크기: {BATCH_SIZE}")
    print(f"    - 총 폴더 수: {len(processed_folders)}")
    print(f"    - 총 이미지 수: {total_images}")

# ==================== 스크립트 실행 ====================

if __name__ == "__main__":
    try:
        main()
        print("\nOpenCLIP ViT-g-14 폴더별 평균 임베딩 생성이 성공적으로 완료되었습니다!")
        print("이제 train 폴더의 각 하위 폴더(클래스)별로 하나의 대표 벡터가 생성되었습니다.")
        print("\n사용법:")
        print("1. predict.py에서 FEATURES_NPY와 PATHS_NPY를 이 파일들의 경로로 변경")
        print("2. MODEL_NAME을 'open_clip/ViT-g-14'로 설정")
        print("3. python predict.py 실행")
        print("\n참고:")
        print("- 각 벡터는 해당 폴더 내 모든 이미지의 평균 임베딩입니다")
        print("- 벡터 개수는 train 폴더 내 하위 폴더의 개수와 동일합니다")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   다음 사항들을 확인해주세요:")
        print("   1. train 폴더가 존재하는지")
        print("   2. train 폴더에 하위 폴더들이 있는지")
        print("   3. 각 하위 폴더에 이미지 파일이 있는지")
        print("   4. GPU 메모리가 충분한지 (RTX 4090 권장, BATCH_SIZE 조정 필요시)")
        print("   5. 인터넷 연결이 안정적인지 (모델 다운로드용)")
        print("   6. open_clip_torch 라이브러리가 설치되어 있는지")
        print("      설치 명령어: pip install open_clip_torch")
