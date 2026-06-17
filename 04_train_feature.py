"""
Feature-Level Cross-Architecture Adversarial Patch (제안 — 직교성 해결 시도)

배경:
  01~03에서 두 검출기의 최종 confidence 손실을 결합(가중합·Min-Max)했으나,
  CNN(YOLOv8)과 Transformer(RT-DETR)의 gradient가 거의 직교(cos≈0)하여
  결합 방식과 무관하게 공격 성능이 비슷·미미했다.

가설:
  최종 출력 대신 두 모델이 공통으로 의존하는 "중간 feature"를 직접 교란하면,
  두 아키텍처의 공격 방향이 덜 직교(더 정렬)해져 전이성이 개선될 수 있다.

방법:
  각 모델 백본의 중간 layer에 forward hook을 걸어 feature map을 추출하고,
   - disrupt: 패치가 적용된 feature를 원본 feature에서 멀어지게 (deviation 최대화)
   - suppress: feature activation 자체를 0으로 (activation 최소화)
  두 모델의 feature 손실을 결합해 PGD로 패치를 최적화한다.
  gradient cosine 유사도도 함께 로깅 → confidence 방식(03) 대비 직교성 변화 측정.

사용 예:
  python 04_train_feature.py --objective disrupt --epochs 50
  python 04_train_feature.py --objective suppress --epochs 50
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
ALPHA = 0.5              # YOLO feature loss 가중치
BETA = 0.5              # RT-DETR feature loss 가중치
BATCH_SIZE = 8           # feature hook 메모리 부담 → 03보다 작게
N_HOOK_LAYERS = 3        # 모델당 hook 거는 중간 layer 수
MAX_IMAGES = None

DATA_FILE = Path("data/person_annotations.json")
OUT_DIR = Path("output")

# ── 유틸 (01~03과 동일) ───────────────────────────────────────────────────────

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


# ── Feature hook ───────────────────────────────────────────────────────────────

def register_feature_hooks(nn_model, n_layers: int):
    """백본의 중간~후반 Conv2d 중 n_layers개를 균등 선택해 forward hook 등록.
    반환: (feats_buffer 리스트, handle 리스트)
    feats_buffer: 매 forward 때 hook이 feature map을 append. 사용 전 clear() 필요.
    """
    convs = [m for m in nn_model.modules() if isinstance(m, torch.nn.Conv2d)]
    if len(convs) < n_layers:
        raise RuntimeError(f"Conv2d layer가 너무 적음: {len(convs)}")
    # 중간~후반(40%~100% 깊이)에서 균등 선택 (의미적 feature 영역)
    start = int(len(convs) * 0.4)
    pool = convs[start:]
    idxs = [int(i) for i in np.linspace(0, len(pool) - 1, n_layers)]
    chosen = [pool[i] for i in idxs]

    feats_buffer = []

    def hook(module, inp, out):
        feats_buffer.append(out)

    handles = [c.register_forward_hook(hook) for c in chosen]
    print(f"    hook 등록: 전체 Conv {len(convs)}개 중 {n_layers}개 선택")
    return feats_buffer, handles


def get_clean_feats(nn_model, feats_buffer, images):
    """패치 없는 원본 이미지의 중간 feature (detach, no grad)."""
    feats_buffer.clear()
    with torch.no_grad():
        nn_model(images)
    return [f.detach() for f in feats_buffer]


def feature_loss(nn_model, feats_buffer, images, clean_feats, objective):
    """
    objective="disrupt": 원본 feature에서 멀어지게 (deviation 최대화 → 음수 최소화)
    objective="suppress": feature activation 자체를 최소화
    값을 '최소화'하면 공격이 강해지도록 정의 (PGD descent와 일치).
    """
    feats_buffer.clear()
    nn_model(images)
    cur = list(feats_buffer)
    if not cur:
        raise RuntimeError("feature hook이 비어있음 (forward 실패?)")
    if objective == "suppress":
        return sum(f.pow(2).mean() for f in cur) / len(cur)
    # disrupt
    dev = sum((a - c).pow(2).mean() for a, c in zip(cur, clean_feats)) / len(cur)
    return -dev


# ── 학습 루프 ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n장치: {device}")
    print(f"Feature objective: {args.objective}")

    with open(DATA_FILE) as f:
        records = json.load(f)
    random.shuffle(records)
    if MAX_IMAGES:
        records = records[:MAX_IMAGES]
    print(f"학습 이미지: {len(records)}장")

    yolo_nn = load_model("yolov8n.pt", device)
    rtdetr_nn = load_model("rtdetr-l.pt", device, train_mode=True)

    # feature hook 등록
    print("  YOLOv8 feature hook:", end=" ")
    yolo_feats, yolo_handles = register_feature_hooks(yolo_nn, N_HOOK_LAYERS)
    print("  RT-DETR feature hook:", end=" ")
    detr_feats, detr_handles = register_feature_hooks(rtdetr_nn, N_HOOK_LAYERS)

    run_name = f"feature_{args.objective}"
    print(f"실행 이름: {run_name}")

    patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
    OUT_DIR.mkdir(exist_ok=True)

    loss_history, cos_history = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_losses, epoch_cos = [], []
        batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]

        for batch in tqdm(batches, desc=f"[feat-{args.objective}] Epoch {epoch}/{args.epochs}"):
            imgs, all_bboxes = [], []
            for rec in batch:
                img = load_and_resize(rec["file_name"], IMG_SIZE).to(device)
                bboxes = scale_bboxes(rec["bboxes"], rec["width"], rec["height"], IMG_SIZE)
                imgs.append(img)
                all_bboxes.append(bboxes)

            # disrupt: 원본(패치 없음) feature를 배치당 1회 미리 계산
            clean_batch = torch.stack(imgs)
            clean_y = clean_d = None
            if args.objective == "disrupt":
                clean_y = get_clean_feats(yolo_nn, yolo_feats, clean_batch)
                clean_d = get_clean_feats(rtdetr_nn, detr_feats, clean_batch)

            for _ in range(PGD_STEPS):
                patched_list = [
                    apply_patch(img, patch, bb)
                    for img, bb in zip(imgs, all_bboxes)
                ]
                batch_tensor = torch.stack(patched_list)

                loss_yolo = feature_loss(yolo_nn, yolo_feats, batch_tensor, clean_y, args.objective)
                loss_detr = feature_loss(rtdetr_nn, detr_feats, batch_tensor, clean_d, args.objective)

                # 모델별 gradient 분리 → 직교성 측정 (03과 동일 지표)
                g_yolo = torch.autograd.grad(loss_yolo, patch, retain_graph=True)[0]
                g_detr = torch.autograd.grad(loss_detr, patch)[0]
                cos = F.cosine_similarity(
                    g_yolo.flatten().unsqueeze(0), g_detr.flatten().unsqueeze(0)
                ).item()
                epoch_cos.append(cos)

                grad = ALPHA * g_yolo + BETA * g_detr
                step_loss = (ALPHA * loss_yolo + BETA * loss_detr).item()

                with torch.no_grad():
                    patch.data -= PGD_STEP_SIZE * grad.sign()
                    patch.data.clamp_(0, 1)

            epoch_losses.append(step_loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_cos = sum(epoch_cos) / len(epoch_cos)
        loss_history.append(avg_loss)
        cos_history.append(avg_cos)
        print(f"  Epoch {epoch}: feat_loss={avg_loss:.4f} | grad_cos={avg_cos:+.4f}")

        save_patch(patch, run_name, epoch)

    for h in yolo_handles + detr_handles:
        h.remove()

    save_patch(patch, run_name, "final")
    plot_curves(loss_history, cos_history, run_name)
    dump_metrics(run_name, loss_history, cos_history)
    print(f"\n완료! output/{run_name}_patch_final.pt 에 저장됨")
    print(f"  평균 gradient cosine 유사도: {sum(cos_history)/len(cos_history):+.4f}")
    print(f"  (confidence 방식(03)의 ≈0 대비 이 값이 커지면 feature-level이 "
          f"직교성을 완화한다는 증거)")


def save_patch(patch: torch.Tensor, run_name: str, tag):
    p = patch.detach().cpu()
    torch.save(p, OUT_DIR / f"{run_name}_patch_{tag}.pt")
    arr = (p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(OUT_DIR / f"{run_name}_patch_{tag}.png")


def plot_curves(loss_h, cos_h, run_name):
    epochs = range(1, len(loss_h) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss_h, marker="o")
    axes[0].set_title(f"{run_name}: feature loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg feature loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, cos_h, marker="d", color="crimson")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{run_name}: gradient cosine similarity")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("cos(g_YOLO, g_RT-DETR)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{run_name}_curves.png", dpi=120)
    plt.close()


def dump_metrics(run_name, loss_h, cos_h):
    data = {
        "method": run_name,
        "epochs": len(loss_h),
        "final_loss": loss_h[-1],
        "avg_grad_cosine": sum(cos_h) / len(cos_h),
        "loss_history": loss_h,
        "cos_history": cos_h,
    }
    with open(OUT_DIR / f"{run_name}_metrics.json", "w") as f:
        json.dump(data, f, indent=2)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective", choices=["disrupt", "suppress"], default="disrupt",
        help="disrupt=원본 feature에서 멀어지게(권장), suppress=activation 최소화",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()
    ALPHA = args.alpha
    BETA = args.beta
    train(args)
