"""
Adversarial Patch 전이성 평가

저장된 패치를 로드하여 각 모델별 mAP drop을 측정.
출력 예:
  모델          패치없음 mAP  패치있음 mAP  mAP drop
  YOLOv8n       0.512        0.203         0.309  ← 높을수록 공격 성공
  RT-DETR-l     0.498        0.271         0.227

사용 예:
  # 단일 패치로 모든 모델 평가
  python 02_evaluate.py --patch output/ensemble_patch_final.pt

  # 여러 패치 비교 (baseline vs 제안)
  python 02_evaluate.py --compare
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────

IMG_SIZE = 640
PATCH_FRACTION = 0.50
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
EVAL_IMAGES = 300        # 평가에 사용할 이미지 수 (전체 사용시 None)

DATA_FILE = Path("data/person_annotations.json")
OUT_DIR = Path("output")

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_and_resize(file_name: str) -> torch.Tensor:
    img = Image.open(Path("data/val2017") / file_name).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def scale_bboxes(bboxes, orig_w, orig_h, new_size):
    sx, sy = new_size / orig_w, new_size / orig_h
    return [[x1*sx, y1*sy, x2*sx, y2*sy] for x1, y1, x2, y2 in bboxes]


def apply_patch(img: torch.Tensor, patch: torch.Tensor, bboxes: list) -> torch.Tensor:
    H, W = img.shape[1], img.shape[2]
    mask = torch.zeros(1, H, W, device=img.device)
    patch_canvas = torch.zeros(3, H, W, device=img.device)

    for x1, y1, x2, y2 in bboxes:
        bw, bh = max(int(x2-x1), 1), max(int(y2-y1), 1)
        target = max(int(min(bw, bh) * PATCH_FRACTION), 16)
        p_resized = F.interpolate(
            patch.unsqueeze(0), size=(target, target),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        px1 = max(0, cx - target//2)
        py1 = max(0, cy - target//2)
        px2 = min(W, px1 + target)
        py2 = min(H, py1 + target)
        ph, pw = py2-py1, px2-px1
        if ph <= 0 or pw <= 0:
            continue
        mask[:, py1:py2, px1:px2] = 1.0
        patch_canvas[:, py1:py2, px1:px2] = p_resized[:, :ph, :pw]

    return img * (1-mask) + patch_canvas * mask


# ── mAP 계산 ──────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """11-point interpolation AP."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = precision[recall >= t]
        ap += (p.max() if len(p) > 0 else 0.0)
    return ap / 11.0


def iou(box1, box2) -> float:
    box1 = [float(v) for v in box1]
    box2 = [float(v) for v in box2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return float(inter) / float(union) if union > 0.0 else 0.0


def evaluate_map(model: YOLO, records: list, patch, device) -> float:
    """
    model: YOLO wrapper (ultralytics)
    patch: None이면 clean 이미지 평가
    반환: person 클래스 AP
    """
    all_tp, all_fp = [], []
    total_gt = 0

    for rec in tqdm(records, desc="  평가 중", leave=False):
        img_tensor = load_and_resize(rec["file_name"]).to(device)
        bboxes = scale_bboxes(rec["bboxes"], rec["width"], rec["height"], IMG_SIZE)
        total_gt += len(bboxes)

        if patch is not None:
            with torch.no_grad():
                img_tensor = apply_patch(img_tensor, patch, bboxes)

        # tensor → numpy 변환 (detach로 grad graph 완전 분리)
        img_np = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        results = model.predict(img_np, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                 classes=[0], verbose=False)  # class 0 = person

        preds = results[0].boxes
        if preds is None or preds.xyxy is None:
            n_preds = 0
        else:
            n_preds = int(preds.xyxy.shape[0])

        if n_preds > 0:
            pred_boxes = np.array(preds.xyxy.cpu().detach().tolist())   # (n, 4)
            pred_confs = np.array(preds.conf.cpu().detach().tolist()).flatten()  # (n,)
        else:
            pred_boxes = np.zeros((0, 4))
            pred_confs = np.zeros(0)

        gt_matched = [False] * len(bboxes)

        # confidence 내림차순 정렬
        order = np.argsort(-pred_confs)
        for idx in order:
            pb = [float(v) for v in pred_boxes[idx]]
            matched = False
            for gi, gb in enumerate(bboxes):
                if not gt_matched[gi] and iou(pb, gb) >= 0.5:
                    gt_matched[gi] = True
                    matched = True
                    break
            all_tp.append(1 if matched else 0)
            all_fp.append(0 if matched else 1)

    if not all_tp:
        return 0.0

    tp_cum = np.cumsum(all_tp)
    fp_cum = np.cumsum(all_fp)
    recall = tp_cum / (total_gt + 1e-8)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    return compute_ap(recall, precision)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run_evaluation(patch_path: str | None, label: str, records: list, device):
    patch = None
    if patch_path:
        patch = torch.load(patch_path, map_location=device, weights_only=True)
        patch = patch.detach().float().to(device)
        patch.requires_grad_(False)

    print(f"\n{'='*55}")
    print(f"패치: {label}")
    print(f"{'='*55}")
    print(f"{'모델':<15} {'Clean mAP':>10} {'Patch mAP':>10} {'mAP Drop':>10}")
    print("-" * 55)

    results = {}
    for model_name in ["yolov8n.pt", "rtdetr-l.pt", "yolov10n.pt"]:
        try:
            model = YOLO(model_name)
            clean_ap = evaluate_map(model, records, None, device)
            patch_ap = evaluate_map(model, records, patch, device) if patch is not None else clean_ap
            drop = clean_ap - patch_ap
            results[model_name] = (clean_ap, patch_ap, drop)
            print(f"  {model_name:<13} {clean_ap:>10.3f} {patch_ap:>10.3f} {drop:>10.3f}")
        except Exception as e:
            print(f"  {model_name:<13} 스킵 ({e})")

    return results


def compare_all(records: list, device):
    """baseline A, baseline B, 앙상블 세 가지 패치 비교."""
    configs = [
        (None,                                             "Clean (no patch)"),
        ("output/yolo_patch_final.pt",                    "Baseline A (YOLOv8 단독)"),
        ("output/rtdetr_patch_final.pt",                  "Baseline B (RT-DETR 단독)"),
        ("output/ensemble_patch_final.pt",                "Ensemble α=0.5 β=0.5"),
        ("output/ensemble_a03_b07_patch_final.pt",        "Ensemble α=0.3 β=0.7"),
        ("output/ensemble_a07_b03_patch_final.pt",        "Ensemble α=0.7 β=0.3"),
    ]
    all_results = {}
    for path, label in configs:
        if path and not Path(path).exists():
            print(f"\n  {label}: 파일 없음, 스킵 ({path})")
            continue
        all_results[label] = run_evaluation(path, label, records, device)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=str, default=None, help="평가할 패치 .pt 파일")
    parser.add_argument("--compare", action="store_true", help="모든 패치 비교")
    parser.add_argument("--n", type=int, default=EVAL_IMAGES, help="평가 이미지 수")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    with open(DATA_FILE) as f:
        records = json.load(f)
    random.shuffle(records)
    if args.n:
        records = records[:args.n]
    print(f"평가 이미지: {len(records)}장")

    if args.compare:
        compare_all(records, device)
    elif args.patch:
        run_evaluation(args.patch, args.patch, records, device)
    else:
        # 기본: 앙상블 패치 평가
        default_patch = "output/ensemble_patch_final.pt"
        if Path(default_patch).exists():
            run_evaluation(default_patch, "Ensemble patch", records, device)
        else:
            print("패치 파일이 없습니다. --patch 옵션으로 경로를 지정하거나 01_train_patch.py를 먼저 실행하세요.")
