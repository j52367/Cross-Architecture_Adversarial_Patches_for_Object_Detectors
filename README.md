# Cross-Architecture Adversarial Patch

## 실행 순서

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. COCO 데이터 준비 (약 1.2GB, 처음 한 번만)
python 00_prepare_data.py

# 3-A. Baseline A — YOLOv8 단독
python 01_train_patch.py --mode yolo --epochs 10

# 3-B. Baseline B — RT-DETR 단독
python 01_train_patch.py --mode rtdetr --epochs 10

# 3-C. 제안 방법 — 앙상블
python 01_train_patch.py --mode ensemble --epochs 10

# 4. 전이성 평가 (세 패치 비교표 출력)
python 02_evaluate.py --compare
```

## 빠른 테스트 (GPU 없을 때)

`01_train_patch.py` 상단의 `MAX_IMAGES = None` 을 `200` 으로 바꾸고,  
`BATCH_SIZE = 4` 를 `1` 로 바꿔서 실행.

## 출력 파일

```
output/
  yolo_patch_final.pt / .png
  rtdetr_patch_final.pt / .png
  ensemble_patch_final.pt / .png
  *_loss.png          ← epoch별 loss 그래프
```
