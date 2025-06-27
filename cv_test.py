# 匯入 OpenCV 模組
import cv2

# 建立一個 VideoCapture 物件，用來存取攝影機（0 表示 /dev/video0）
cap = cv2.VideoCapture(0)

# 如果攝影機開啟失敗，就跳出程式
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

# === ❷ 建立視窗並加滑桿（只做一次） ===
cv2.namedWindow('原始畫面')         # 之後所有影像都以這個視窗為主
# 低閾值滑桿：初始 100，上限 500
cv2.createTrackbar('low',  '原始畫面', 100, 500, lambda x: None)
# 高閾值滑桿：初始 200，上限 500
cv2.createTrackbar('high', '原始畫面', 200, 500, lambda x: None)    

# 開始無限迴圈，持續讀取每一張影像
while True:
    # 讀取一張畫面（ret 是成功與否，frame 是影像內容）
    ret, frame = cap.read()
    
    if not ret:
        print("無法讀取畫面")
        break



    # 讀滑桿數值
    low  = cv2.getTrackbarPos('low',  '原始畫面')
    high = cv2.getTrackbarPos('high', '原始畫面')    

    # 將影像轉為灰階（從彩色 BGR → 單通道 GRAY）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 用 Canny 方法進行邊緣偵測
    # 參數是兩個閾值（越高，偵測邊緣越「乾淨」）
    # 使用滑桿設定的閾值做 Canny 邊緣偵測
    edges = cv2.Canny(gray, low, high)
    #print("邊緣像素數量：", cv2.countNonZero(edges))


    # 顯示原始畫面
    cv2.imshow('原始畫面', frame)
    
    # 顯示灰階影像
    cv2.imshow('灰階影像', gray)

    # 顯示邊緣偵測結果
    cv2.imshow('邊緣偵測', edges)

    # 等待鍵盤輸入（1 毫秒），若按下 q 鍵則跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機資源
cap.release()

# 關閉所有 OpenCV 開啟的視窗
cv2.destroyAllWindows()

