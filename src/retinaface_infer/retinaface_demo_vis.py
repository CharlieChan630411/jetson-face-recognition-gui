import argparse
import os
import sys

import cv2, numpy as np
from src.retinaface_infer.retinaface_trt import RetinaFaceTRT
from src.retinaface_infer.retinaface_post import decode, decode_landm, nms, _PRIORS

parser = argparse.ArgumentParser()
parser.add_argument('--engine', required=True, help='Path to TensorRT engine file')
parser.add_argument('--image', required=True, help='Path to input image')
args = parser.parse_args()

engine_path = args.engine
if not os.path.exists(engine_path):
    print(f"[❌] 找不到指定 engine：{engine_path}")
    sys.exit(1)

image_path  = args.image       # 測試圖

model = RetinaFaceTRT(engine_path)
if not os.path.exists(image_path):
    print(f"[❌] 找不到圖片檔案：{image_path}")
    sys.exit(1)

img = cv2.imread(image_path)

boxes_raw, scores_raw, landm_raw = model.infer(img)
# fix: 對齊 anchor 數量（TRT 可能多輸出 padding，這裡只取對齊 priors 數）
N = _PRIORS.shape[0] #N=7980

boxes  = decode(boxes_raw[0][:N], _PRIORS)
landms = decode_landm(landm_raw[0][:N], _PRIORS)
# 原本的錯誤版本
# scores = scores_raw[0][:,1]   ← 這是 (15960, 2)

# 改成這樣：只取前 7980 行，這樣才跟 decode 出來的 priors 對齊
scores = scores_raw[0][:N, 1]

mask = scores > 0.5
boxes, landms, scores = boxes[mask], landms[mask], scores[mask]

keep  = nms(boxes, scores)
boxes, landms, scores = boxes[keep], landms[keep], scores[keep]

for b,l,s in zip(boxes, landms, scores):
    x1,y1,x2,y2 = (b * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).astype(int)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    for i in range(5):
        x,y = int(l[2*i]*img.shape[1]), int(l[2*i+1]*img.shape[0])
        cv2.circle(img,(x,y),2,(0,0,255),-1)
    cv2.putText(img,f"{s:.2f}",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)

cv2.imwrite("output_vis.jpg", img)
print("✅ 已輸出 output_vis.jpg")
