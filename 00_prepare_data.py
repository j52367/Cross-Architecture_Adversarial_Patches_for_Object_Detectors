"""
COCO val2017 person 서브셋 준비
- COCO val2017 이미지 + 어노테이션 다운로드
- person 카테고리 이미지만 필터링
- data/person_annotations.json 저장
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "val2017"
ANNO_FILE = DATA_DIR / "annotations" / "instances_val2017.json"
OUT_FILE = DATA_DIR / "person_annotations.json"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_with_progress(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  이미 존재: {dest}")
        return

    print(f"  다운로드: {url}")

    class ProgressBar(tqdm):
        def update_hook(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with ProgressBar(unit="B", unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_hook)


def extract(zip_path: Path, dest: Path):
    print(f"  압축 해제: {zip_path.name} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="  추출 중"):
            zf.extract(member, dest)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 1. 어노테이션 다운로드 (약 240MB)
    anno_zip = DATA_DIR / "annotations_trainval2017.zip"
    if not ANNO_FILE.exists():
        download_with_progress(COCO_ANNO_URL, anno_zip)
        extract(anno_zip, DATA_DIR)
        anno_zip.unlink(missing_ok=True)

    # 2. 이미지 다운로드 (약 1GB) — 오래 걸림
    images_zip = DATA_DIR / "val2017.zip"
    if not IMAGES_DIR.exists() or len(list(IMAGES_DIR.glob("*.jpg"))) < 1000:
        download_with_progress(COCO_IMAGES_URL, images_zip)
        extract(images_zip, DATA_DIR)
        images_zip.unlink(missing_ok=True)

    # 3. person 이미지 필터링
    print("\nCOCO 어노테이션 로드 중...")
    with open(ANNO_FILE) as f:
        coco = json.load(f)

    # person 카테고리 ID 찾기
    person_cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "person")
    print(f"  person category ID: {person_cat_id}")

    # person bbox가 있는 이미지 수집
    image_id_to_bboxes: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != person_cat_id:
            continue
        x, y, w, h = ann["bbox"]
        if w < 30 or h < 60:  # 너무 작은 사람 제외
            continue
        iid = ann["image_id"]
        if iid not in image_id_to_bboxes:
            image_id_to_bboxes[iid] = []
        image_id_to_bboxes[iid].append([x, y, x + w, y + h])  # [x1,y1,x2,y2]

    # 이미지 정보 매핑
    id_to_info = {img["id"]: img for img in coco["images"]}

    # 유효한 이미지만 (파일 실제 존재 확인)
    records = []
    for iid, bboxes in image_id_to_bboxes.items():
        info = id_to_info[iid]
        img_path = IMAGES_DIR / info["file_name"]
        if not img_path.exists():
            continue
        records.append(
            {
                "image_id": iid,
                "file_name": info["file_name"],
                "width": info["width"],
                "height": info["height"],
                "bboxes": bboxes,
            }
        )

    print(f"  총 person 이미지: {len(records)}장")

    # 저장
    with open(OUT_FILE, "w") as f:
        json.dump(records, f)
    print(f"  저장 완료: {OUT_FILE}")


if __name__ == "__main__":
    main()
