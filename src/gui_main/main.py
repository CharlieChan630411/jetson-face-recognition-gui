# src/gui_main/camera.py

import cv2

CAM_INDEX = 0  # 默認攝影機 ID，可依需要修改
cap = None

def open_camera():
    global cap
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("❌ 無法開啟攝影機")

def read_frame():
    global cap
    if cap is None:
        raise RuntimeError("❌ 攝影機尚未開啟")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ 讀取畫面失敗")
    return frame

def close_camera():
    global cap
    if cap:
        cap.release()
        cap = None

