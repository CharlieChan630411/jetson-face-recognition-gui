import cv2

def open_camera(cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("❌ 無法開啟攝影機")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ 無法讀取攝影機畫面")
    return frame

def close_camera(cap):
    cap.release()
    print("📷 攝影機已關閉")

# ✅ 測試區：可以直接執行此檔測試
if __name__ == "__main__":
    print("🚀 測試 camera.py 啟動")
    cap = open_camera()
    frame = read_frame(cap)
    print("✅ 成功讀取畫面，frame.shape =", frame.shape)
    close_camera(cap)

