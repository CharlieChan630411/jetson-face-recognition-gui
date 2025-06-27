import shutil
from pathlib import Path

import face_recognition as fr

SRC = Path("/home/user/test/face-capture/dataset")
GOOD = SRC.parent / "good"
BAD  = SRC.parent / "bad"
GOOD.mkdir(exist_ok=True); BAD.mkdir(exist_ok=True)

for img_path in SRC.rglob("*.[jp][pn]g"):
    img = fr.load_image_file(img_path)
    if fr.face_locations(img, model="hog"):
        shutil.copy(img_path, GOOD / img_path.name)
    else:
        shutil.copy(img_path, BAD / img_path.name)
        print("⚠ 無臉 →", img_path.name)

print("✅ 可用照片數：", len(list(GOOD.iterdir())))
print("❌ 無臉照片數：", len(list(BAD.iterdir())))
