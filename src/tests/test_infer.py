#!/usr/bin/env python3
"""
test_infer.py – 單圖驗證 RetinaFace TensorRT engine
"""
import os
import cv2
import numpy as np
from retinaface_infer.retinaface_trt import RetinaFaceTRT
from retinaface_infer.landmark_drawer import draw_landmarks

# 設定相對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(BASE_DIR, "../retinaface_infer/retinaface.engine")
IMAGE_PATH  = os.path.join(BASE_DIR, "./output_retina.jpg")      # 你手邊的測試圖，640×608/640 都可

def main():
    model = RetinaFaceTRT(ENGINE_PATH)
    img = cv2.imread(IMAGE_PATH)
    # ---- 推論 ----
    boxes, scores, landms = model.infer(img)  # 假設 infer 回傳三 tensor
    
    # 取信心值最高的一張臉示範
    top_idx = int(np.argmax(scores[0][:, 1]))
    best_box   = boxes[0][top_idx]
    best_landm = landms[0][top_idx]

    # 畫框
    x1, y1, x2, y2 = map(int, best_box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 畫 landmark
    img = draw_landmarks(img, [best_landm.tolist()])

    # 存檔
    # 輸出結果到「與 test_infer.py 同層」目錄
    out_path = os.path.join(BASE_DIR, "output_retina_landmark.jpg")

	# 確保資料夾存在（保險手段）
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ok = cv2.imwrite(out_path, img)
    print("✅ 完成！結果已輸出：", out_path)
    print("[DEBUG] imwrite 成功？", ok)

    
if __name__ == "__main__":
    main()

