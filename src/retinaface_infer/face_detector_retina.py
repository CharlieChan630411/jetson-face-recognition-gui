"""
face_detector_retina.py  ── 單張測試版
步驟：
1. 透過 insightface.model_zoo.RetinaFace 下載官方 mnet0.25 模型
2. 讀取攝影機單張影像
3. 偵測人臉 → 畫框 → 存檔 output_retina.jpg
"""

import cv2
import os, tempfile

from insightface.model_zoo import get_model

# 1. 下載 / 載入官方 RetinaFace 模型
print("🚀 重新下载官方 RetinaFace mnet025 模型并测试")

# 临时目录，保证肯定是空的，程序会自动下载正确版本
temp_root = tempfile.mkdtemp()

def get_detector():
    # 把名稱改成有效的型號
    return get_model("retinaface_mnet025", root="~/.insightface")


# ↓↓↓ 只在「直接執行」時才跑 demo ↓↓↓
if __name__ == "__main__":
    print("🚀 重新下載官方 RetinaFace mnet025 模型並測試")    
    detector = get_detector()
    detector.prepare(ctx_id=0)   # ctx_id=0 → CPU；若之後用 GPU 改 -1

    # 2. 讀取攝影機影像
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("❌ 摄影机画面为空")

    # ✅ 加入這一行：確認影像尺寸
    print("frame.shape =", frame.shape)

    # 3. 偵測人臉
    faces, _ = detector.detect(frame, threshold=0.5, scale=1.0)
    print(f"✅ 偵測到 {len(faces)} 張人臉")

    # 4. 畫框並存檔
    for x1, y1, x2, y2, *_ in faces:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imwrite("output_retina.jpg", frame)
    print("📸 已輸出 output_retina.jpg")

