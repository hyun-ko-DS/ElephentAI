import os
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, AutoModel
from typing import Dict, List
from collections import defaultdict

# ==================== 설정 ====================
IMAGE_DIR = "train"
OUT_FEATURES = "embeddings/train_features_dinov2_large_mean.npy"
OUT_PATHS = "embeddings/train_paths_dinov2_large_mean.npy"
MODEL_NAME = "facebook/dinov2-large"
BATCH_SIZE = 8
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==================== 이미지 스캔 함수 ====================
def get_folder_structure(root: str) -> Dict[str, List[str]]:
    folder_images = defaultdict(list)
    for d in os.listdir(root):
        dpath = os.path.join(root, d)
        if os.path.isdir(dpath):
            for fname in os.listdir(dpath):
                if os.path.splitext(fname)[1].lower() in ALLOWED_EXTS:
                    folder_images[dpath].append(os.path.join(dpath, fname))
            folder_images[dpath].sort()
    return folder_images

# ==================== 모델 로딩 함수 ====================
def load_transformers_model(model_name: str, device: torch.device):
    print(f"🔧 모델 로드 (Transformers): {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)
    print("✅ 로드 완료")
    return model, processor

# ==================== 배치 임베딩 함수 ====================
def embed_batch(model, processor, pil_images, device: torch.device) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS 토큰 기준
        feats = outputs.last_hidden_state[:, 0, :]
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

# ==================== 폴더별 평균 계산 ====================
def process_folder_images(model, processor, image_paths: List[str], device: torch.device) -> np.ndarray:
    if not image_paths:
        return np.zeros((1, model.config.hidden_size), dtype="float32")
    all_feats, batch_imgs = [], []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(img)
            if len(batch_imgs) == BATCH_SIZE:
                all_feats.append(embed_batch(model, processor, batch_imgs, device))
                batch_imgs = []
        except (UnidentifiedImageError, OSError):
            print(f"⚠️ SKIP: {p}")
    if batch_imgs:
        all_feats.append(embed_batch(model, processor, batch_imgs, device))
    feats = np.vstack(all_feats)
    mean = np.mean(feats, axis=0, keepdims=True)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    return mean.astype("float32")

# ==================== 메인 함수 ====================
def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Device: {device}")

    model, processor = load_transformers_model(MODEL_NAME, device)

    folders = get_folder_structure(IMAGE_DIR)
    if not folders:
        raise RuntimeError(f"No image folders under {IMAGE_DIR}")

    paths = sorted(folders.keys())
    print(f" 총 폴더 {len(paths)}개, 이미지 약 {sum(len(v) for v in folders.values())}개")

    all_means, processed = [], []
    for fp in tqdm(paths, desc="폴더 임베딩 처리"):
        mean = process_folder_images(model, processor, folders[fp], device)
        all_means.append(mean)
        processed.append(fp)

    features = np.vstack(all_means) if all_means else np.zeros((0, model.config.hidden_size), dtype="float32")
    np.save(OUT_FEATURES, features)
    np.save(OUT_PATHS, np.array(processed, dtype=object))

    print("✅ 완료")
    print(f" features shape: {features.shape}")
    print(f" 저장 위치: {OUT_FEATURES}, {OUT_PATHS}")
    print(f" 소요 시간: {time.time() - start:.1f}초")

if __name__ == "__main__":
    main()
