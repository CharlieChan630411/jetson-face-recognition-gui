#!/usr/bin/env python3
# scripts/regenerate_faces.py

"""
從 face-capture/dataset 下所有人臉照片中萃取特徵並建立 faces.pkl
"""

import face_recognition as fr
from pathlib import Path
import pickle
import sys

DATASET_DIR = Path(__file__).resolve().parent.parent / "face-capture" / "dataset"
PKL_PATH     = DATASET_DIR / "faces.pkl"

def build_database(dataset_dir: Path, output_path: Path):
    encodings = []
    names = []

    print(f"📁 掃描資料夾：{dataset_dir}")
    for img_path in sorted(dataset_dir.rglob("*.[jp][pn]g")):
        name = img_path.stem.split("_")[0]
        img = fr.load_image_file(img_path)
        face_encs = fr.face_encodings(img)

        if face_encs:
            encodings.append(face_encs[0])
            names.append(name)
            print(f"  ✅ {img_path.name} → {name}")
        else:
            print(f"  ⚠️ 無法辨識臉：{img_path.name}")

    if not encodings:
        print("❌ 沒有成功擷取任何人臉特徵，請檢查資料集！")
        sys.exit(1)

    with open(output_path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print(f"\n✅ 已儲存特徵庫：{output_path}（共 {len(encodings)} 筆）")

if __name__ == "__main__":
    build_database(DATASET_DIR, PKL_PATH)

