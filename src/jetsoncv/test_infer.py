#!/usr/bin/env python3
"""
test_infer.py – 單圖驗證 RetinaFace TensorRT engine
"""
import cv2
from retinaface_trt import RetinaFaceTRT

ENGINE_PATH = "retinaface.engine"
IMAGE_PATH  = "output_retina.jpg"      # 你手邊的測試圖，640×608/640 都可

def main():
    model = RetinaFaceTRT(ENGINE_PATH)
    img = cv2.imread(IMAGE_PATH)
    outputs = model.infer(img)
    print("=== 推論完成 ===")
    for i, out in enumerate(outputs):
        print(f"output[{i}] shape:", out.shape, "範例值:", out.flat[:5])

if __name__ == "__main__":
    main()

