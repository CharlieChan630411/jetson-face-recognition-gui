#!/usr/bin/env python3
# scripts/batch_update_faces.py
"""
批次檢查 dataset 照片：
1. 掃描來源資料夾 (預設 face-capture/dataset)
2. 有臉 → good/、側臉 → sideface/、無臉 → bad/
3. 可 --user 限制單一成員；--src 自訂來源路徑
"""

import argparse
import shutil
from pathlib import Path

import cv2                               # ✅ 加入 cv2
import face_recognition as fr            # ✅ 使用簡短別名 fr


# ---------- CLI 參數 ----------
def parse_args():
    p = argparse.ArgumentParser(description="分類 dataset 照片（正臉 / 側臉 / 無臉）")
    p.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "face-capture" / "dataset",
        help="來源資料夾（預設：face-capture/dataset）",
    )
    p.add_argument(
        "--user",
        type=str,
        default=None,
        help="僅處理指定使用者（檔名前綴，例如 charlie）",
    )
    return p.parse_args()


# ---------- 主流程 ----------
def main():
    args = parse_args()
    src: Path = args.src
    if not src.exists():
        raise FileNotFoundError(f"{src} 不存在")

    good_dir = src.parent / "good"
    bad_dir = src.parent / "bad"
    sideface_dir = src.parent / "sideface"
    good_dir.mkdir(exist_ok=True)
    bad_dir.mkdir(exist_ok=True)
    sideface_dir.mkdir(exist_ok=True)

    # 側臉 Haar Cascade
    profile_model_path = "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml"
    profile_cascade = cv2.CascadeClassifier(profile_model_path)
    if profile_cascade.empty():
        raise RuntimeError("❌ 無法載入側臉模型 (haarcascade_profileface.xml)")

    total_good = total_bad = total_side = 0
    pattern = "*.[jp][pn]g"

    for img_path in sorted(src.rglob(pattern)):
        if args.user and not img_path.name.startswith(args.user):
            continue

        img = fr.load_image_file(img_path)
        if fr.face_locations(img, model="hog"):
            shutil.copy2(img_path, good_dir / img_path.name)
            total_good += 1
            continue  # 已歸類為正臉，跳到下一張

        # 若正面偵測不到 → 嘗試側臉
        bgr = cv2.imread(str(img_path))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        profiles = profile_cascade.detectMultiScale(gray, 1.1, 5)
        if len(profiles):
            shutil.copy2(img_path, sideface_dir / img_path.name)
            print(f"👤 側臉 → {img_path.name}")
            total_side += 1
        else:
            shutil.copy2(img_path, bad_dir / img_path.name)
            print(f"⚠️  無臉 → {img_path.name}")
            total_bad += 1

    print("\n===== 統計結果 =====")
    print(f"✅ 正面臉照片：{total_good}")
    print(f"👤 側臉照片：{total_side}")
    print(f"❌ 無法辨識臉：{total_bad}")


if __name__ == "__main__":
    main()

