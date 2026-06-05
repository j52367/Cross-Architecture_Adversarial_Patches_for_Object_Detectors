"""
이종(異種) 아키텍처 Adversarial Patch - 균형 최적화 (Balanced / Min-Max)

기존 01_train_patch.py 의 앙상블(--mode ensemble)은 두 모델의 loss를
단순 가중합(α·L_YOLO + β·L_RTDETR)으로 결합한다. 그러나 CNN(YOLOv8)과
Transformer(RT-DETR)의 gradient가 서로 충돌하면, 단순 가중합은 한쪽으로
치우치거나 양쪽 모두 어중간한 타협점에 수렴하는 한계가 있다.

본 스크립트는 두 가지를 추가한다:
  1) Min-Max 균형 최적화:  L = max(L_YOLO, L_RTDETR)
     → 매 스텝 "덜 공격된(=confidence 높은) 모델"을 우선 공격 → 균형 유도
  2) Gradient 충돌 측정:  두 모델 gradient의 cosine 유사도를 로깅
     → 단순 가중합이 왜 어중간해지는지 정량적 근거 제공

기존 01/02 파일은 그대로 두고, 결과도 balanced_* 로 따로 저장한다.

사용 예:
  python 03_train_balanced.py --method minmax   --epochs 50   # 제안 (균형)
  python 03_train_balanced.py --method weighted --epochs 50   # 비교용 (단순 가중합)
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

IMG_SIZE = 640
PATCH_SIZE = 400
PATCH_FRACTION = 0.50
PGD_STEPS = 20
PGD_STEP_SIZE = 2 / 255
ALPHA = 0.5              # weighted 모드에서만 사용
BETA = 0.5              # weighted 모드에서만 사용
BATCH_SIZE = 16
MAX_IMAGES = None

DATA_FILE = Path("data/person_annotations.json")
OUT_DIR = Path("output")

# ── 유틸 (01_train_patch.py와 동일) ───────────────────────────────────────────

def load_and_resize(file_name: str, size: int) -> torch.Tensor:
    img = Image.open(Path("data/val2017") / file_name).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def scale_bboxes(bboxes: list, orig_w: int, orig_h: int, new_size: int) -> list:
    sx = new_size / orig_w
    sy = new_size / orig_h
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in bboxes]


def apply_patch(img: torch.Tensor, patch: torch.Tensor, bboxes: list) -> torch.Tensor:
    H, W = img.shape[1], img.shape[2]
    mask = torch.zeros(1, H, W, device=img.device)
    patch_canvas = torch.zeros(3, H, W, device=img.device)

    for x1, y1, x2, y2 in bboxes:
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)
        target = max(int(min(bw, bh) * PATCH_FRACTION), 16)
        p_resized = F.interpolate(
            patch.unsqueeze(0), size=(target, target),
            mode="bilinear", align_corners=False
        ).squeeze(0)
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

    return img * (1 - mask) + patch_canvas * mask


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(name: str, device: torch.device, train_mode: bool = False):
    print(f"  모델 로드: {name}")
    m = YOLO(name)
    nn = m.model.to(device)
    if train_mode:
        nn.train()
    else:
        nn.eval()
    for p in nn.parameters():
        p.requires_grad_(False)
    return nn


# ── Loss 함수 (01_train_patch.py와 동일하게 유지 → 공정 비교) ─────────────────

def get_conf_yolo(nn_model, images: torch.Tensor) -> torch.Tensor:
    out = nn_model(images)
    preds = out[0] if isinstance(out, (list, tuple)) else out
    if preds.dim() == 3 and preds.shape[1] > 4:
        class_scores = preds[:, 4:, :]
        conf = class_scores.sigmoid().max(dim=1).values
        return conf.max(dim=1).values.mean()
    if preds.dim() == 3 and preds.shape[2] > 4:
        class_scores = preds[:, :, 4:]
        conf = class_scores.sigmoid().max(dim=2).values
        return conf.max(dim=1).values.mean()
    raise ValueError(f"예상치 못한 YOLOv8 출력 shape: {preds.shape}")


def get_conf_rtdetr(nn_model, images: torch.Tensor) -> torch.Tensor:
    out = nn_model(images)
    pred_scores = out[3]
    conf = pred_scores.sigmoid().max(dim=2).values
    return conf.max(dim=1).values.mean()


# ── 학습 루프 ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n장치: {device}")
    print(f"결합 방식: {args.method}")

    with open(DATA_FILE) as f:
        records = json.load(f)
    random.shuffle(records)
    if MAX_IMAGES:
        records = records[:MAX_IMAGES]
    print(f"학습 이미지: {len(records)}장")

    # 두 모델 모두 로드 (균형 최적화는 항상 앙상블)
    yolo_nn = load_model("yolov8n.pt", device)
    rtdetr_nn = load_model("rtdetr-l.pt", device, train_mode=True)

    run_name = f"balanced_{args.method}"
    print(f"실행 이름: {run_name}")

    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
    OUT_DIR.mkdir(exist_ok=True)

    loss_history = []
    cos_history = []        # gradient 충돌(cosine 유사도) 추이
    yolo_loss_history = []  # 모델별 loss 추이 (균형 확인용)
    detr_loss_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses, epoch_cos = [], []
        epoch_ly, epoch_ld = [], []

        batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]

        for batch in tqdm(batches, desc=f"[{args.method}] Epoch {epoch}/{args.epochs}"):
            imgs, all_bboxes = [], []
            for rec in batch:
                img = load_and_resize(rec["file_name"], IMG_SIZE).to(device)
                bboxes = scale_bboxes(rec["bboxes"], rec["width"], rec["height"], IMG_SIZE)
                imgs.append(img)
                all_bboxes.append(bboxes)

            for _ in range(PGD_STEPS):
                patched_list = [
                    apply_patch(img, patch, bboxes)
                    for img, bboxes in zip(imgs, all_bboxes)
                ]
                batch_tensor = torch.stack(patched_list)

                # 두 모델 loss를 각각 계산
                loss_yolo = get_conf_yolo(yolo_nn, batch_tensor)
                loss_detr = get_conf_rtdetr(rtdetr_nn, batch_tensor)

                # 모델별 gradient 분리 계산 (충돌 측정 + min-max 업데이트용)
                g_yolo = torch.autograd.grad(loss_yolo, patch, retain_graph=True)[0]
                g_detr = torch.autograd.grad(loss_detr, patch)[0]

                # gradient 충돌 측정: cosine 유사도 (-1=정반대, +1=일치)
                cos = F.cosine_similarity(
                    g_yolo.flatten().unsqueeze(0),
                    g_detr.flatten().unsqueeze(0),
                ).item()
                epoch_cos.append(cos)
                epoch_ly.append(loss_yolo.item())
                epoch_ld.append(loss_detr.item())

                # ── 결합 방식에 따라 사용할 gradient 결정 ──
                if args.method == "minmax":
                    # 더 안 막힌(confidence 높은) 모델을 우선 공격 → 균형
                    if loss_yolo.item() >= loss_detr.item():
                        grad = g_yolo
                        step_loss = loss_yolo.item()
                    else:
                        grad = g_detr
                        step_loss = loss_detr.item()
                else:  # weighted (단순 가중합, 비교용)
                    grad = ALPHA * g_yolo + BETA * g_detr
                    step_loss = (ALPHA * loss_yolo + BETA * loss_detr).item()

                # PGD 업데이트
                with torch.no_grad():
                    patch.data -= PGD_STEP_SIZE * grad.sign()
                    patch.data.clamp_(0, 1)

            epoch_losses.append(step_loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_cos = sum(epoch_cos) / len(epoch_cos)
        avg_ly = sum(epoch_ly) / len(epoch_ly)
        avg_ld = sum(epoch_ld) / len(epoch_ld)
        loss_history.append(avg_loss)
        cos_history.append(avg_cos)
        yolo_loss_history.append(avg_ly)
        detr_loss_history.append(avg_ld)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} | "
              f"YOLO={avg_ly:.4f} RT-DETR={avg_ld:.4f} | "
              f"grad_cos={avg_cos:+.4f}")

        save_patch(patch, run_name, epoch)

    # 최종 저장
    save_patch(patch, run_name, "final")
    plot_curves(loss_history, cos_history, yolo_loss_history, detr_loss_history, run_name)
    dump_metrics(run_name, loss_history, cos_history, yolo_loss_history, detr_loss_history)
    print(f"\n완료! output/{run_name}_patch_final.pt 에 저장됨")
    print(f"  평균 gradient cosine 유사도: {sum(cos_history)/len(cos_history):+.4f}")
    print(f"  (값이 낮거나 음수일수록 두 아키텍처의 gradient 충돌이 크다는 의미)")


def save_patch(patch: torch.Tensor, run_name: str, tag):
    p = patch.detach().cpu()
    torch.save(p, OUT_DIR / f"{run_name}_patch_{tag}.pt")
    arr = (p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT_DIR / f"{run_name}_patch_{tag}.png")


def plot_curves(loss_h, cos_h, ly_h, ld_h, run_name):
    epochs = range(1, len(loss_h) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (1) 모델별 loss 추이 → 균형 확인
    axes[0].plot(epochs, ly_h, marker="o", label="YOLOv8 (CNN)")
    axes[0].plot(epochs, ld_h, marker="s", label="RT-DETR (Transformer)")
    axes[0].set_title(f"{run_name}: per-model confidence")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg confidence")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # (2) gradient 충돌(cosine) 추이
    axes[1].plot(epochs, cos_h, marker="d", color="crimson")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{run_name}: gradient cosine similarity")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("cos(g_YOLO, g_RT-DETR)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{run_name}_curves.png", dpi=120)
    plt.close()


def dump_metrics(run_name, loss_h, cos_h, ly_h, ld_h):
    """발표/분석용으로 수치를 json으로도 저장."""
    data = {
        "method": run_name,
        "epochs": len(loss_h),
        "final_loss": loss_h[-1],
        "avg_grad_cosine": sum(cos_h) / len(cos_h),
        "final_yolo_conf": ly_h[-1],
        "final_detr_conf": ld_h[-1],
        "loss_history": loss_h,
        "cos_history": cos_h,
        "yolo_conf_history": ly_h,
        "detr_conf_history": ld_h,
    }
    with open(OUT_DIR / f"{run_name}_metrics.json", "w") as f:
        json.dump(data, f, indent=2)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["minmax", "weighted"],
        default="minmax",
        help="minmax=균형 최적화(제안), weighted=단순 가중합(비교용)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()
    ALPHA = args.alpha
    BETA = args.beta
    train(args)
