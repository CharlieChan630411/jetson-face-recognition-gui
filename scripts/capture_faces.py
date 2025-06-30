import argparse
import os
import time
from pathlib import Path

import cv2

# === 參數設定 ===
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='檔名前綴 (人名)')
parser.add_argument('--max',  type=int, default=15, help='自動存圖上限')
args = parser.parse_args()

name      = args.name
max_auto  = args.max
save_root = Path('./face-capture/dataset/')
person_dir = save_root / name  # ➤ 改為每人一個資料夾
person_dir.mkdir(parents=True, exist_ok=True)

# === 模型設定 ===
model_path = '/home/user/models/face-dnn/'
prototxt = model_path + 'deploy.prototxt'
weights  = model_path + 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === 拍照條件 ===
face_size = (300, 300)
min_face_w = 100
stable_frames = 5
cap = cv2.VideoCapture(0)

counter = 1
stable_count = 0
prev_box = None

print(f"🟢 偵測中：人員 {name}，最多自動拍 {max_auto} 張")
print("📸 空白鍵手動拍照，q 離開")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104, 177, 123), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    face_detected = False
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1

            if face_w < min_face_w or face_h < min_face_w:
                continue

            # ➤ 判斷是否穩定
            if prev_box and abs(x1 - prev_box[0]) < 15 and abs(y1 - prev_box[1]) < 15:
                stable_count += 1
            else:
                stable_count = 0
            prev_box = (x1, y1)

            face_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ➤ 加入正面判斷條件（左右差距不大）
            face_center_diff = abs((x1 + x2)//2 - w//2)
            if face_center_diff > face_w * 0.3:
                continue  # 臉偏太側，不存

            # ➤ 自動儲存
            if stable_count >= stable_frames and counter <= max_auto:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_resize = cv2.resize(face, face_size)

                    # ➤ 自動避開重複檔名
                    while True:
                        filename = person_dir / f"{name}_{counter}.jpg"
                        if not filename.exists():
                            break
                        counter += 1

                    cv2.imwrite(str(filename), face_resize)
                    print(f"📸 自動儲存 {filename.name}")
                    counter += 1
                    stable_count = 0
            break

    cv2.imshow("自動抓臉拍照", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # 空白鍵手動拍照
        if face_detected:
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face_resize = cv2.resize(face, face_size)

                while True:
                    filename = person_dir / f"{name}_{counter}.jpg"
                    if not filename.exists():
                        break
                    counter += 1

                cv2.imwrite(str(filename), face_resize)
                print(f"📸 手動儲存 {filename.name}")
                counter += 1

cap.release()
cv2.destroyAllWindows()
