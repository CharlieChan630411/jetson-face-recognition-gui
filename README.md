# jetson-face-recognition-gui
> Jetson AGX Orin 即時人臉辨識 GUI 系統 (RetinaFace + GUI + Dataset)

A real-time face recognition GUI system on Jetson AGX Orin, integrating RetinaFace inference with TensorRT, 5-point landmark decoding, face registration, webcam input and GUI interface.

---

## 📌 專案介紹 | Project Overview

本專案整合 NVIDIA Jetson 平台與 TensorRT，提供一個完整的實時人臉偵測與 GUI 管理介面，可延伸應用至人臉比對、註冊與辨識。

This project integrates NVIDIA Jetson platform with TensorRT-based RetinaFace for real-time face detection and provides a GUI system for managing face datasets and visualization.

---

## 💡 功能特色 | Features

- ✅ RetinaFace TensorRT 模型推論 (with pycuda)
- ✅ 解碼 bounding boxes 與五點人臉關鍵點
- ✅ 單張圖片畫框 demo（輸出 `output_vis.jpg`）
- ✅ 支援 webcam 輸入與即時人臉截圖
- ✅ Dataset 分類（good / bad / sideface）
- ✅ GUI 管理與人臉資料建立

---

## 🚀 快速開始 | Quick Start

### 安裝依賴
```bash
sudo apt install python3-pycuda
pip install -r requirements.txt
```

### 單圖推論 + 畫框輸出
```bash
python3 src/jetsoncv/retinaface_demo_vis.py
```

結果圖將輸出為 `output_vis.jpg`，可視覺化人臉偵測結果。

---

## 🧱 專案結構 | Project Structure

```
test/
├── face-capture/            # 人臉擷取分類圖像資料夾（good / bad / sideface）
│
├── scripts/                 # 管理與重建 dataset 的腳本
│   ├── capture_faces.py
│   ├── batch_update_faces.py
│   └── regenerate_faces.py
│
├── src/jetsoncv/
│   ├── retinaface_trt.py         # TensorRT 推論核心
│   ├── retinaface_post.py        # Anchor 解碼與 Landmark 邏輯
│   ├── retinaface_demo_vis.py    # 單圖 demo，輸出畫框圖
│   ├── retinaface.engine         # 編譯後 TRT 引擎
│   ├── output_retina.jpg         # 原始圖片
│   ├── output_vis.jpg            # 畫框結果圖
│   └── (其他 face_encoder, gui, main 等模組)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 可延伸方向 | Future Extensions

- 🔍 臉部辨識模型整合（FaceNet / ArcFace embedding）
- 🧬 臉部註冊與身份標註、比對
- 🎛️ GUI 增加身份管理、清單與操作按鈕
- 🎥 多鏡頭輸入與錄影串接
- 🧠 加入姿態估計與側臉偵測處理

---

## 📜 授權 License

本專案採用 MIT 授權條款  
Licensed under the MIT License

---

Maintained by [Charlie Chan](https://github.com/CharlieChan630411)
