#!/usr/bin/env python3
# src/main.py
"""
專案主入口：負責初始化並啟動 GUI。
辨識邏輯與畫面由 main.py 控制。
"""

from .face_detector import FaceDetector

def main():
    detector = FaceDetector() 
    detector.run()

if __name__ == "__main__":
    main()  # 一行啟動 GUI

