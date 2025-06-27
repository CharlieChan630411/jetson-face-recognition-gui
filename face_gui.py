#!/usr/bin/env python3
"""
OpenCV + face_recognition  即時人臉辨識 GUI
‒ 顯示 FPS、辨識信心度，門檻可透過 TOLERANCE 調整
"""

import pickle
import time
from pathlib import Path

import cv2
import face_recognition as fr
import numpy as np

TOLERANCE = 0.45                                     # 越小越嚴格
PICKLE_PATH = Path("/home/user/test/face-capture/dataset/faces.pkl")  # ★

# === 讀取 faces.pkl ===
if not PICKLE_PATH.exists():
    raise FileNotFoundError(f"{PICKLE_PATH} 不存在，請先執行 build_embeddings.py")

data = pickle.load(open(PICKLE_PATH, "rb"))
known_encodings = data["encodings"]
known_names     = data["names"]
print(f"✅ 載入 {len(known_encodings)} 張人臉特徵：{set(known_names)}")

# === 開啟攝影機 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ 無法開啟攝影機")

print("🟢 按 q 離開")
while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locs  = fr.face_locations(rgb)
    encs  = fr.face_encodings(rgb, locs)

    for (top, right, bottom, left), enc in zip(locs, encs):
        dists = fr.face_distance(known_encodings, enc)
        best  = np.argmin(dists) if len(dists) else None
        name, conf = "Unknown", 0.0
        if best is not None and dists[best] < TOLERANCE:
            name  = known_names[best]
            conf  = 1 - dists[best]

        # 還原座標
        top, right, bottom, left = [v*4 for v in (top, right, bottom, left)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        label = name if name=="Unknown" else f"{name} ({conf:.2f})"
        cv2.putText(frame, label, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    fps = 1.0 / (time.time() - tic)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
