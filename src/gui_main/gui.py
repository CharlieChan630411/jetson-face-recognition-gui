#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/gui.py  –  Jetson‑FaceRecognizer 即時 GUI（CNN + 效能資訊）
────────────────────────────────────────────────────────
• 讀取 faces.pkl 後以 CNN 模型 (face_recognition) 進行偵測/辨識。
• 於畫面左上角即時顯示：
    ‑ FPS (整體流暢度)
    ‑ 單次推理時間 (ms) ＝ face_locations + face_encodings 全流程
• 人臉框旁顯示「姓名 + 信心值」。
• 按下「q」離開。

可調參數：
    TOLERANCE      ─ 比對容忍度 (越小越嚴格)
    PROCESS_EVERY  ─ 每 N 幀才做一次推理 (降低算力)
    CAM_INDEX      ─ 攝影機索引 (0=內建, 1=USB…)

"""

from pathlib import Path
import time

import cv2                            # OpenCV 影像處理
import face_recognition as fr         # face_recognition 函式庫

from jetsoncv.face_database import load_db # 讀取 faces.pkl 自家模組

# 置於 import 區域
import subprocess, threading, re

# ── GPU 使用率監聽 ───────────────────────────────
_GPU_UTIL = "GPU: ..."
_gpu_re   = re.compile(r"GR3D_FREQ\s+(\d+)%")    # 只抓 GR3D_FREQ 99%

def _update_gpu_util(interval=1.0):
    """背景執行 tegrastats，將 GPU% 寫入 _GPU_UTIL."""
    global _GPU_UTIL
    cmd = ["tegrastats", "--interval", f"{int(interval*1000)}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        for line in proc.stdout:
            m = _gpu_re.search(line)
            if m:
                _GPU_UTIL = f"GPU: {m.group(1)} %"
    except Exception:
        _GPU_UTIL = "GPU: ???"
    finally:
        proc.kill()



# ────────────────────────────────────────────────
# 參數設定
# ────────────────────────────────────────────────
TOLERANCE: float   = 0.45   # 0~1，越小匹配越嚴格
PROCESS_EVERY: int = 2      # 每 2 幀執行一次推理
CAM_INDEX: int     = 0      # 攝影機 ID
FONT                = cv2.FONT_HERSHEY_SIMPLEX

# ────────────────────────────────────────────────
# 辅助函式
# ────────────────────────────────────────────────

def _confidence_from_distance(dist: float) -> float:
    """將 dlib 距離 (0~1) 線性轉換為 0~100% 信心值。"""
    return max(0.0, min(1.0, 1.0 - dist)) * 100.0

# ────────────────────────────────────────────────
# 主執行函式
# ────────────────────────────────────────────────

def run_gui() -> None:
    """由 src/main.py 呼叫的入口點；獨立執行亦可。"""

    # 1️⃣ 讀取特徵庫 -----------------------------------------------------
    db = load_db()
    encodings = db["encodings"]
    names_db  = db["names"]
    print(f"✅ faces.pkl 載入完成：共 {len(encodings)} 筆特徵，人物 {set(names_db)}")
    threading.Thread(target=_update_gpu_util, daemon=True).start()


    # 2️⃣ 開啟攝影機 -----------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("❌ 無法開啟攝影機，請確認連線或權限")

    frame_idx = 0             # 幀計數器 (決定何時推理)
    boxes, labels = [], []    # 暫存結果
    prev_t = time.time()      # 上一幀時間，用來計算 FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 讀取畫面失敗，程式結束")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 只有在指定間隔才執行人臉推理 (省算力)
        inference_ms = 0.0
        if frame_idx % PROCESS_EVERY == 0:
            t0 = time.perf_counter()                 # ➜ 推理計時開始
            boxes = fr.face_locations(rgb, model="cnn")  # CNN 模型偵測
            face_vecs = fr.face_encodings(rgb, boxes)
            inference_ms = (time.perf_counter() - t0) * 1000  # 轉 ms
            labels = []
            for vec in face_vecs:
                matches = fr.compare_faces(encodings, vec, tolerance=TOLERANCE)
                if True in matches:
                    idxs = [i for i, m in enumerate(matches) if m]
                    dists = fr.face_distance([encodings[i] for i in idxs], vec)
                    best_idx  = idxs[int(dists.argmin())]
                    best_dist = dists.min()
                    conf = _confidence_from_distance(best_dist)
                    label = f"{names_db[best_idx]} {conf:.1f}%"
                else:
                    label = "Unknown"
                labels.append(label)

        # 3️⃣ 繪製框線與文字 ---------------------------------------------
        for (top, right, bottom, left), label in zip(boxes, labels):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), FONT, 0.6, (0, 255, 0), 2)

        # 4️⃣ 顯示效能資訊 ---------------------------------------------
        curr_t = time.time()
        fps = 1.0 / (curr_t - prev_t) if curr_t != prev_t else 0.0
        prev_t = curr_t
        info1 = f"FPS: {fps:.1f}"
        info2 = f"Infer: {inference_ms:.1f} ms" if inference_ms else "Infer: ..."
        cv2.putText(frame, info1, (10, 20), FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, info2, (10, 45), FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, _GPU_UTIL, (10, 70), FONT, 0.6, (0, 255, 0), 2)

        # 5️⃣ 顯示視窗 & 退出判定 ---------------------------------------
        cv2.imshow("Jetson Face Recognition (CNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 使用者結束程式")
            break

        frame_idx += 1

    # 6️⃣ 清理資源 -----------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

# ────────────────────────────────────────────────
# 偵錯執行（直接 python src/gui.py）
# ────────────────────────────────────────────────
if __name__ == "__main__":
    run_gui()

