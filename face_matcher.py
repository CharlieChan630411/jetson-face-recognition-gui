import os
import sys

import face_recognition

if len(sys.argv) != 3:
    print("❌ 使用方式: python3 face_matcher.py 圖片1 圖片2")
    sys.exit(1)

img1_path = sys.argv[1]
img2_path = sys.argv[2]

# 確認檔案存在
if not os.path.exists(img1_path) or not os.path.exists(img2_path):
    print("❌ 找不到圖片檔案！")
    sys.exit(1)

# 載入兩張圖片
img1 = face_recognition.load_image_file(img1_path)
img2 = face_recognition.load_image_file(img2_path)

# 擷取人臉特徵
encodings1 = face_recognition.face_encodings(img1)
encodings2 = face_recognition.face_encodings(img2)

if len(encodings1) == 0 or len(encodings2) == 0:
    print("⚠️ 圖片中找不到人臉，請使用正面清晰照片")
    sys.exit(1)

# 比對
result = face_recognition.compare_faces([encodings1[0]], encodings2[0])

if result[0]:
    print("✅ 是同一個人！")
else:
    print("❌ 不是同一個人。")
