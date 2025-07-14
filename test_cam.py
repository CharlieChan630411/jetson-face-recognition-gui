# test_cam.py
import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("❌ Camera open failed")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

