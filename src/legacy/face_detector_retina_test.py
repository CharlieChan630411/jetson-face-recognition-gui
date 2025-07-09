import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)
ret, frame = cap.read();  cap.release()
if not ret: raise RuntimeError("cam fail")

faces = app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print("detect =", len(faces))

for f in faces:
    x1,y1,x2,y2 = map(int, f.bbox)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite("output_retina.jpg", frame)

