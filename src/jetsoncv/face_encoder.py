#!/usr/bin/env python3
# file: face_recognizer.py
# 功能：即時辨識多張臉，標示姓名

import glob
import os
from pathlib import Path

import cv2
import face_recognition as fr

# === 1. 載入已知人臉特徵 ===
DATA_DIR = Path("/home/user/test/face-capture/dataset")
known_encodings = []
known_names     = []

print(f"⏳ 讀取資料夾: {DATA_DIR}")
# 遞迴抓所有 jpg / png
for img_path in DATA_DIR.rglob("*.[jp][pn]g"):           # 支援 jpg / png
    name = img_path.stem.split("_")[0]      # 取檔名前綴
    img  = fr.load_image_file(img_path)
    encodings = fr.face_encodings(img)
    if len(encodings):                                    # 確保有偵測到臉
        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"  ✔ 讀取 {img_path} → {name}")
    else:
        print(f"  ⚠ 跳過 {img_path}（偵測不到臉）")

if not known_encodings:
    raise SystemExit("❌ 沒有任何有效的臉部特徵可供比對，請檢查資料夾")

# === 2. 開啟攝影機 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ 無法開啟 /dev/video0")

print("🟢 辨識開始：按 q 離開")
PROCESS_EVERY_N_FRAMES = 2   # 每 2 幀做一次比對，可提高 FPS
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # face_recognition 用 RGB
    names  = []
    boxes  = []

    # === 3. 每 N 幀做一次比對 ===
    if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
        boxes  = fr.face_locations(rgb, model="hog")      # hog 節能，cnn 更準但慢
        encods = fr.face_encodings(rgb, boxes)

        names = []
        for enc in encods:
            # 與所有已知臉比對，取最小距離
            matches  = fr.compare_faces(known_encodings, enc, tolerance=0.48)
            # tolerance 越低越嚴格 (0.4~0.6 常用)
            name = "Unknown"

            # 取距離最近者
            distances = fr.face_distance(known_encodings, enc)
            best_match_idx = distances.argmin() if len(distances) else None
            if best_match_idx is not None and matches[best_match_idx]:
                name = known_names[best_match_idx]
            names.append(name)

    # === 4. 畫框 + 姓名 ===
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 22), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Face Recognition", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
