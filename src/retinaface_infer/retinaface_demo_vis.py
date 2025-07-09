import cv2, numpy as np
from retinaface_trt import RetinaFaceTRT
from retinaface_infer.retinaface_post import decode, decode_landm, nms, _PRIORS

engine_path = "retinaface.engine"       # 與 .py 同層
image_path  = "output_retina.jpg"       # 測試圖

model = RetinaFaceTRT(engine_path)
img   = cv2.imread(image_path)
boxes_raw, scores_raw, landm_raw = model.infer(img)

boxes  = decode(boxes_raw[0], _PRIORS)
landms = decode_landm(landm_raw[0], _PRIORS)
scores = scores_raw[0][:,1]

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
