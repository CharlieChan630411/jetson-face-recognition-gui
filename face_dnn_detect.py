import sys

import cv2

# === 1. 載入模型檔案 ===
model_path = '/home/user/models/face-dnn/'
prototxt = model_path + 'deploy.prototxt'
weights  = model_path + 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# === 2. 嘗試使用 GPU 加速（Jetson 可選）===
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# === 3. 開啟攝影機 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("❌ 無法開啟攝影機")

print("🟢 DNN 臉部偵測中（按 q 離開）")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 無法讀取畫面")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 閾值越低，會抓越多臉（包含誤判）
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("DNN 臉部偵測", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
