import cv2

# === 1. 載入正面與側臉模型 ===
front_model = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml' #載入正面模型
profile_model = '/usr/share/opencv4/haarcascades/haarcascade_profileface.xml' #載入側面模型

# 載入 OpenCV 內建的臉部偵測模型（Haar Cascade）
face_cascade = cv2.CascadeClassifier(front_model)
profile_cascade = cv2.CascadeClassifier(profile_model) 

if face_cascade.empty() or profile_cascade.empty():
    sys.exit("❌ 無法載入模型")
    

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取畫面")
        break

    # 將畫面轉為灰階（臉部偵測效率更高）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 3. 偵測正臉 ===
    # 偵測臉部，返回的 faces 是座標陣列
    faces = face_cascade.detectMultiScale(
        gray,              # 灰階影像
        scaleFactor=1.3,   # 每次縮小影像比例（數字越小越精細）
        minNeighbors=5,    # 偵測到多少相鄰區塊才算真的臉（越大越嚴格）
        minSize=(30, 30)   # 最小臉部尺寸
    )

    # === 4. 偵測側臉（原圖 + 水平鏡像）===
    profiles = profile_cascade.detectMultiScale(
        gray,              # 灰階影像
        scaleFactor=1.3,   # 每次縮小影像比例（數字越小越精細）
        minNeighbors=5,    # 偵測到多少相鄰區塊才算真的臉（越大越嚴格）
        minSize=(30, 30)   # 最小臉部尺寸)
    )
    flipped_gray = cv2.flip(gray, 1) # 左右翻轉（偵測另一側）
    flipped_profiles = profile_cascade.detectMultiScale(
        flipped_gray,              # 灰階影像
        scaleFactor=1.3,   # 每次縮小影像比例（數字越小越精細）
        minNeighbors=5,    # 偵測到多少相鄰區塊才算真的臉（越大越嚴格）
        minSize=(30, 30)   # 最小臉部尺寸)
    )


    # === 5. 畫框（正臉是綠色） ===
    #OpenCV 是使用BGR 不是RGB
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # === 6. 畫框（原圖側臉是藍色） ===
    for (x, y, w, h) in profiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 

    # === 7. 畫框（翻轉側臉是紅色） ===  
    frame_width = frame.shape[1]
    for (x, y, w, h) in flipped_profiles:
        x_flip = frame_width - x - w
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)     

    # === 8. 顯示影像 ===
    cv2.imshow('正臉 + 側臉偵測', frame)

    # 按下 q 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清除資源
cap.release()
cv2.destroyAllWindows()
