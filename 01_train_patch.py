"""
Cross-Architecture Adversarial Patch 학습

모드:
  --mode yolo   : YOLOv8 단독 baseline
  --mode rtdetr : RT-DETR 단독 baseline
  --mode ensemble : 앙상블 (제안 방법)

사용 예:
  python 01_train_patch.py --mode ensemble --epochs 10
  python 01_train_patch.py --mode yolo     --epochs 5
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────

IMG_SIZE = 640           # YOLO 표준 입력 크기
PATCH_SIZE = 300         # 패치 픽셀 크기 (정사각형)
PATCH_FRACTION = 0.35    # 패치가 bounding box에서 차지하는 비율
PGD_STEPS = 10           # 이미지당 PGD 업데이트 횟수
PGD_STEP_SIZE = 2 / 255  # 각 스텝의 크기
ALPHA = 0.5              # YOLO loss 가중치
BETA = 0.5               # RT-DETR loss 가중치
BATCH_SIZE = 16          # GPU 메모리에 따라 조절 (24GB: 16, CPU: 1)
MAX_IMAGES = None        # None = 전체 사용. 빠른 테스트: 200

DATA_FILE = Path("data/person_annotations.json")
OUT_DIR = Path("output")

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_and_resize(file_name: str, size: int) -> torch.Tensor:
    """이미지를 로드하여 [3, size, size] float32 tensor 반환 (0~1 범위)."""
    img = Image.open(Path("data/val2017") / file_name).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def scale_bboxes(bboxes: list, orig_w: int, orig_h: int, new_size: int) -> list:
    """bbox 좌표를 새 이미지 크기에 맞게 스케일."""
    sx = new_size / orig_w
    sy = new_size / orig_h
    scaled = []
    for x1, y1, x2, y2 in bboxes:
        scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return scaled


def apply_patch(img: torch.Tensor, patch: torch.Tensor, bboxes: list) -> torch.Tensor:
    """
    패치를 이미지의 모든 person bbox에 붙임.
    img   : [3, H, W]  (leaf tensor, no grad)
    patch : [3, PATCH_SIZE, PATCH_SIZE]  (requires_grad=True)
    반환  : [3, H, W]  (패치가 붙은 텐서, 그래디언트 연결 유지)
    """
    H, W = img.shape[1], img.shape[2]
    # mask와 patch_canvas를 통해 differentiable하게 합성
    mask = torch.zeros(1, H, W, device=img.device)
    patch_canvas = torch.zeros(3, H, W, device=img.device)

    for x1, y1, x2, y2 in bboxes:
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)

        # 패치를 bbox 크기 × PATCH_FRACTION으로 리사이즈
        target = max(int(min(bw, bh) * PATCH_FRACTION), 16)
        p_resized = F.interpolate(
            patch.unsqueeze(0), size=(target, target),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # bbox 중심에 배치
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        px1 = max(0, cx - target // 2)
        py1 = max(0, cy - target // 2)
        px2 = min(W, px1 + target)
        py2 = min(H, py1 + target)
        ph, pw = py2 - py1, px2 - px1
        if ph <= 0 or pw <= 0:
            continue

        mask[:, py1:py2, px1:px2] = 1.0
        patch_canvas[:, py1:py2, px1:px2] = p_resized[:, :ph, :pw]

    # 합성: 패치 영역은 patch, 나머지는 원본
    patched = img * (1 - mask) + patch_canvas * mask
    return patched


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(name: str, device: torch.device):
    """YOLO 모델 로드 + 가중치 동결."""
    print(f"  모델 로드: {name}")
    m = YOLO(name)
    nn = m.model.to(device).eval()
    for p in nn.parameters():
        p.requires_grad_(False)
    return nn


def debug_output_shape(nn_model, device: torch.device):
    """모델 출력 shape을 확인 (처음 한 번만)."""
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        out = nn_model(dummy)
    if isinstance(out, (list, tuple)):
        shapes = [o.shape if hasattr(o, "shape") else type(o) for o in out]
        print(f"    출력 shape: {shapes}")
        return out
    print(f"    출력 shape: {out.shape}")
    return out


# ── Loss 함수 ──────────────────────────────────────────────────────────────────

def get_conf_yolo(nn_model, images: torch.Tensor) -> torch.Tensor:
    """
    YOLOv8 evasion loss.
    출력 [B, 84, 8400] → class scores [B, 80, 8400] → max conf per anchor
    이 값을 최소화 → 검출 억제
    """
    out = nn_model(images)
    # ultralytics eval 모드: tuple (decoded_preds, raw) 또는 tensor
    preds = out[0] if isinstance(out, (list, tuple)) else out
    # preds: [B, 4+nc, na]
    if preds.dim() == 3 and preds.shape[1] > 4:
        class_scores = preds[:, 4:, :]          # [B, nc, na]
        conf = class_scores.sigmoid().max(dim=1).values  # [B, na]
        return conf.max(dim=1).values.mean()     # scalar
    # fallback: 마지막 차원이 nc+4인 경우 [B, na, nc+4]
    if preds.dim() == 3 and preds.shape[2] > 4:
        class_scores = preds[:, :, 4:]           # [B, na, nc]
        conf = class_scores.sigmoid().max(dim=2).values  # [B, na]
        return conf.max(dim=1).values.mean()
    raise ValueError(f"예상치 못한 YOLOv8 출력 shape: {preds.shape}")


def get_conf_rtdetr(nn_model, images: torch.Tensor) -> torch.Tensor:
    """
    RT-DETR evasion loss.
    ultralytics RT-DETR eval 모드 출력은 이미 sigmoid 적용된 값 → sigmoid 재적용 금지
    [B, queries, 4+nc] 또는 [B, 4+nc, queries] 두 형태 모두 처리
    queries=300, 4+nc=84(COCO 기준)
    """
    out = nn_model(images)
    preds = out[0] if isinstance(out, (list, tuple)) else out
    if preds.dim() == 3:
        if preds.shape[1] > preds.shape[2]:
            # [B, queries, 4+nc] e.g. [B, 300, 84]
            class_scores = preds[:, :, 4:]              # [B, queries, nc]
            conf = class_scores.max(dim=2).values       # [B, queries]
        else:
            # [B, 4+nc, queries] e.g. [B, 84, 300]
            class_scores = preds[:, 4:, :]              # [B, nc, queries]
            conf = class_scores.max(dim=1).values       # [B, queries]
        return conf.max(dim=1).values.mean()
    raise ValueError(f"예상치 못한 RT-DETR 출력 shape: {preds.shape}")


# ── 학습 루프 ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n장치: {device}")

    # 데이터 로드
    with open(DATA_FILE) as f:
        records = json.load(f)
    random.shuffle(records)
    if MAX_IMAGES:
        records = records[:MAX_IMAGES]
    print(f"학습 이미지: {len(records)}장")

    # 모델 로드
    yolo_nn = rtdetr_nn = None
    if args.mode in ("yolo", "ensemble"):
        yolo_nn = load_model("yolov8n.pt", device)
        print("  YOLOv8 출력 확인:", end=" ")
        debug_output_shape(yolo_nn, device)
    if args.mode in ("rtdetr", "ensemble"):
        rtdetr_nn = load_model("rtdetr-l.pt", device)
        print("  RT-DETR 출력 확인:", end=" ")
        debug_output_shape(rtdetr_nn, device)

    # 패치 초기화 (random noise)
    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)

    OUT_DIR.mkdir(exist_ok=True)
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []

        # 배치 구성
        batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]

        for batch in tqdm(batches, desc=f"Epoch {epoch}/{args.epochs}"):
            # 이미지 + bbox 로드
            imgs, all_bboxes = [], []
            for rec in batch:
                img = load_and_resize(rec["file_name"], IMG_SIZE).to(device)
                bboxes = scale_bboxes(rec["bboxes"], rec["width"], rec["height"], IMG_SIZE)
                imgs.append(img)
                all_bboxes.append(bboxes)

            # PGD inner loop
            for _ in range(PGD_STEPS):
                if patch.grad is not None:
                    patch.grad.zero_()

                # 각 이미지에 패치 적용 후 배치로 합치기
                patched_list = [
                    apply_patch(img, patch, bboxes)
                    for img, bboxes in zip(imgs, all_bboxes)
                ]
                batch_tensor = torch.stack(patched_list)  # [B, 3, H, W]

                # Loss 계산
                loss = torch.tensor(0.0, device=device)
                if yolo_nn is not None:
                    loss_yolo = get_conf_yolo(yolo_nn, batch_tensor)
                    loss = loss + ALPHA * loss_yolo
                if rtdetr_nn is not None:
                    loss_rtdetr = get_conf_rtdetr(rtdetr_nn, batch_tensor)
                    loss = loss + BETA * loss_rtdetr

                # 역전파 (loss를 최소화 = 검출 confidence 최소화)
                loss.backward()

                with torch.no_grad():
                    patch.data -= PGD_STEP_SIZE * patch.grad.sign()
                    patch.data.clamp_(0, 1)

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

        # 중간 저장
        save_patch(patch, args.mode, epoch)

    # 최종 저장
    save_patch(patch, args.mode, "final")
    plot_loss(loss_history, args.mode)
    print(f"\n완료! output/{args.mode}_patch_final.pt 에 저장됨")


def save_patch(patch: torch.Tensor, mode: str, tag):
    """패치를 .pt (tensor)와 .png (이미지)로 저장."""
    p = patch.detach().cpu()
    torch.save(p, OUT_DIR / f"{mode}_patch_{tag}.pt")
    # PNG로도 저장
    arr = (p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT_DIR / f"{mode}_patch_{tag}.png")


def plot_loss(history: list, mode: str):
    plt.figure()
    plt.plot(history, marker="o")
    plt.title(f"{mode} loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Avg confidence loss")
    plt.savefig(OUT_DIR / f"{mode}_loss.png")
    plt.close()


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["yolo", "rtdetr", "ensemble"],
        default="ensemble",
        help="학습 모드",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--alpha", type=float, default=ALPHA, help="YOLO loss 가중치"
    )
    parser.add_argument(
        "--beta", type=float, default=BETA, help="RT-DETR loss 가중치"
    )
    args = parser.parse_args()
    ALPHA = args.alpha
    BETA = args.beta
    train(args)
