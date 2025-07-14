"""
face_detector.py
----------------
以物件導向方式封裝「攝影機開啟 → 臉部偵測 → 顯示結果」流程。
後續若要切換到 RetinaFace + TensorRT，只要改 init() / detect() 兩處即可。
"""

import cv2
import sys
from datetime import datetime

def get_face_detector():
    from gui_main.face_detector import FaceDetector
    return FaceDetector



class FaceDetector:
    """臉部偵測主類別"""

    def __init__(self, cam_id: int = 0):
        """
        初始化：
        1. 載入 (暫時) Haar Cascade 模型
        2. 預留未來 TensorRT engine 的載入位置
        """
        # ======== 預留 TensorRT engine 入口 ========
        engine_path = "models/retinaface.engine"
        print("🚧 目前使用 OpenCV Haar Cascade；日後將整合 TensorRT")
        print(f"🧪 預留 engine 路徑：{engine_path}")
        # =========================================

        # 1. 先載入 Haar Cascade
        front_model = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        profile_model = "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml"

        self.face_cascade = cv2.CascadeClassifier(front_model)
        self.profile_cascade = cv2.CascadeClassifier(profile_model)

        if self.face_cascade.empty() or self.profile_cascade.empty():
            sys.exit("❌ 無法載入 Haar Cascade 模型")

        # 2. 打開攝影機
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            sys.exit("❌ 無法開啟攝影機")

        # 3. 設定視窗名稱
        self.window = "Jetson Face Detection (Haar Cascade)"

    # --------------------------------------------------
    def detect(self, frame):
        """
        臉部偵測核心函式：
        1. 轉灰階提高效率
        2. 同時偵測正臉與側臉
        3. 回傳所有框框 (list)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測正臉
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # 偵測側臉
        profiles = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # 將兩種結果合併
        return list(faces) + list(profiles)

    # --------------------------------------------------
    def run(self):
        """主迴圈：讀取影像 → 偵測 → 畫框 → 顯示"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取畫面")
                break

            boxes = self.detect(frame)

            # 畫框
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 在左上角顯示 FPS 時間戳
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time {ts}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(self.window, frame)

            # 按 q 離開
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 收尾
        self.cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------
# 當直接執行此檔時，啟動偵測；被 import 時不會跑
if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()

