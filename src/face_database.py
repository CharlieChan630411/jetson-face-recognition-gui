#!/usr/bin/env python3
# src/face_database.py
"""
統一管理人臉特徵庫 (faces.pkl) 的讀取功能。
之後如果要換路徑、換檔名，只改這裡就好。
"""

from pathlib import Path
import pickle

# 預設的 faces.pkl 路徑（相對於專案根目錄）
DEFAULT_PKL = (
    Path(__file__).resolve().parent.parent
    / "face-capture" / "dataset" / "faces.pkl"
)

def load_db(pkl_path: Path = DEFAULT_PKL):
    """
    讀取 faces.pkl 並回傳一個 dict：
    {'encodings': [...], 'names': [...]}
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} 不存在，請先建立特徵庫")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


