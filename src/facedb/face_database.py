#!/usr/bin/env python3
# jetsoncv.face_database

from pathlib import Path
import pickle

# 專案根目錄： .../test/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# faces.pkl 預設路徑
DEFAULT_PKL = PROJECT_ROOT / "face-capture" / "dataset" / "faces.pkl"


def load_db(pkl_path: Path = DEFAULT_PKL):
    """載入 faces.pkl，回傳 dict：{'encodings': …, 'names': …}"""
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} 不存在，請先建立特徵庫")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

