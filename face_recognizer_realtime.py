#!/usr/bin/env python3
import pickle
import time

import cv2
import face_recognition as fr

DATA_DIR = "/home/user/test/face-capture/dataset/faces.pkl"
with open(DATA_DIR, "rb") as f:
    DB = pickle.load(f)

print(f"載入 {len(DB['encodings'])} 張臉特徵…")

cap = cv2.VideoCapture(0)
PROCESS_EVERY = 2      # 每 2 幀辨識一次
thresh = 0.48          # 辨識閾值
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_idx % PROCESS_EVERY == 0:
        boxes = fr.face_locations(rgb, model="hog")
        faces = fr.face_encodings(rgb, boxes)
        names = []
        for enc in faces:
            matches = fr.compare_faces(DB["encodings"], enc, tolerance=thresh)
            name = "Unknown"
            if True in matches:
                best = matches.index(True)
                name = DB["names"][best]
            names.append(name)

    for (t,r,b,l), name in zip(boxes, names):
        cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(frame, name, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
