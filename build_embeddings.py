#!/usr/bin/env python3
"""
重新掃描 dataset/ 內所有子資料夾，抽取每張臉特徵向量 (128 維)，
並覆寫 faces.pkl 供 GUI／即時辨識使用。
"""

import os
import pickle
from pathlib import Path

import face_recognition as fr

DATA_DIR = Path("/home/user/test/face-capture/dataset")   # ★依需要修改
OUT_PATH = DATA_DIR / "faces.pkl"

encodings, names = [], []

print("⏳ 重新建立人臉特徵…")
for img_path in DATA_DIR.rglob("*.[jp][pn]g"):
    if not img_path.is_file():
        continue
    label = img_path.parent.name          # 資料夾名稱 = 人名
    img   = fr.load_image_file(img_path)
    boxes = fr.face_locations(img, model="cnn")  # HOG→輕量 / CNN→較準
    if len(boxes) != 1:
        print(f"⚠️  {img_path.name} 跳過（找到 {len(boxes)} 張臉）")
        continue
    enc = fr.face_encodings(img, boxes)[0]
    encodings.append(enc)
    names.append(label)
    print(f"  ✔  {img_path.relative_to(DATA_DIR)} → {label}")

# ★★ 以「wb」覆寫寫入新特徵檔
with open(OUT_PATH, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"\n✅ faces.pkl 已更新：{OUT_PATH}")
print(f"   ➜ 共寫入 {len(encodings)} 張臉特徵，人物清單：{set(names)}")
