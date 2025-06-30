# Jetson-FaceRecognizer
> Real-time Face Recognition GUI and Tools on Jetson AGX Orin  
> 在 Jetson AGX Orin 上執行的即時人臉辨識系統與工具集

---

## 💡 Features 功能特色

- 🎥 Real-time GUI face recognition using `face_recognition` + OpenCV
- 🧠 Automatic encoding and `.pkl` feature database generation
- 🧹 Batch cleaner: classify face / non-face / sideface images
- 📦 Modular codebase with CLI tools for dataset and training

---

## 🚀 Quick Start 快速開始

```bash
# Install dependencies 安裝套件
pip install -r requirements.txt

# Build features 建立臉部特徵庫
python3 scripts/regenerate_faces.py

# (Optional) Classify images into good/bad/sideface 分類照片
python3 scripts/batch_update_faces.py --user charlie

# Launch GUI 執行 GUI 辨識主程式
PYTHONPATH=. python3 src/main.py

---

Directory Structure 資料夾結構

face-capture/         # Original dataset 圖片資料庫（每人一資料夾）
├── dataset/
│   ├── charlie/*.jpg
│   ├── Joy/*.jpg
│   ├── Lamar/*.jpg
│   └── faces.pkl     # Generated features 特徵庫

scripts/              # CLI 工具腳本
├── regenerate_faces.py       # 建立 .pkl 特徵庫
├── batch_update_faces.py     # good/bad/sideface 自動分類

src/                  # 主程式模組
├── main.py           # 程式入口，呼叫 GUI
├── gui.py            # OpenCV GUI 視窗邏輯
├── face_database.py  # 載入 faces.pkl 的工具
├── face_encoder.py   # 從圖片建立 encodings
├── face_detector.py  # 側臉/正臉模型輔助
└── camera.py         # 攝影機模組（可擴充）

---

⚙️ Requirements 套件需求
Jetson AGX Orin / CUDA GPU

Python ≥ 3.8

face_recognition

opencv-python

numpy

---

📄 License 授權
MIT License.
歡迎使用、修改與延伸本專案。














