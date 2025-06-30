#!/usr/bin/env python3
# src/gui.py
"""
OpenCV + face_recognition 即時人臉辨識 GUI 主控制模組
由 main.py 呼叫 run_gui() 啟動
"""

import cv2
import face_recognition as fr
from src.face_database import load_db

TOLERANCE = 0.45  # 比對閾值（越小越嚴格）
PROCESS_EVERY = 2  # 每 N 幀處理一次

def run_gui():
    # 載入人臉特徵庫
    DB = load_db()
    known_encodings = DB["encodings"]
    known_names = DB["names"]
    print(f"✅ 載入 {len(known_encodings)} 張人臉特徵：{set(known_names)}")

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ 無法開啟攝影機")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取影像")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_idx % PROCESS_EVERY == 0:
            boxes = fr.face_locations(rgb, model="hog")
            faces = fr.face_encodings(rgb, boxes)
            names = []
            for enc in faces:
                matches = fr.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
                name = "Unknown"
                if True in matches:
                    best = matches.index(True)
                    name = known_names[best]
                names.append(name)

        # 畫框與標示
        for (top, right, bottom, left), name in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 離開辨識畫面")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

