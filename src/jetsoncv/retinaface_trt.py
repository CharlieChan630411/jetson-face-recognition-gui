#!/usr/bin/env python3
"""retinaface_trt.py – TensorRT 推論封裝 (NHWC 1x608x640x3)

★ 2025‑07‑08 修正版 1.1
    • 將所有 `engine.num_bindings` 改為 `engine.get_nb_bindings()` (TensorRT 10+ API)

此版本專為 Jetson AGX Orin + TensorRT 10.x 設計，
搭配已建構完成的 `retinaface.engine`（input shape: 1×608×640×3，NHWC）。

功能：
    1. 載入 ICudaEngine 並建立 IExecutionContext
    2. 自動配置 host / device buffer
    3. 提供 `infer(image_bgr)`，回傳 raw outputs（[boxes, scores, landms]）

依賴：
    - numpy
    - opencv‑python （cv2）
    - tensorrt (JetPack 6 預裝)
    - pycuda (sudo apt install python3-pycuda)
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自動管理 context
except ModuleNotFoundError as e:
    raise SystemExit("❌ pycuda 未安裝，請先： sudo apt install python3-pycuda") from e


class RetinaFaceTRT:
    """RetinaFace TensorRT 推論封裝（自動相容各版本 API）"""

    # --- 新增：統一取 shape 函式 ------------------------------------------
    def _shape(self, idx: int):
        """跨版本取得 tensor shape"""
        if hasattr(self.engine, "get_binding_shape"):       # TRT ≤ 8
            return self.engine.get_binding_shape(idx)
        # TRT 9/10：先拿 tensor name，再拿 shape
        name = self.engine.get_tensor_name(idx)
        return self.engine.get_tensor_shape(name)

    @staticmethod
    def _nb_bindings(engine: trt.ICudaEngine) -> int:
        """跨版本取得 binding 數量"""
        if hasattr(engine, "num_io_tensors"):   # TensorRT 9/10 新 API
            return engine.num_io_tensors

        if hasattr(engine, "get_nb_bindings"):
            return engine.get_nb_bindings()
        if hasattr(engine, "num_bindings"):
            return engine.num_bindings  # type: ignore[attr-defined]
        raise AttributeError("ICudaEngine 无法取得 bindings 數量 (未知 API 版本)")
        
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"找不到 TensorRT engine：{engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        nb = self._nb_bindings(self.engine)  # 跨版本取得 binding 數

        # 取得輸入 / 輸出 binding index
        self.input_binding = 0  # 第一個 binding 為 input
        self.output_bindings = [i for i in range(1, nb)]

        # 解析 input shape (NHWC)
        self.input_shape = tuple(self._shape(self.input_binding))
        self.batch, self.in_h, self.in_w, self.in_c = self.input_shape

        # 分配 buffer
        self._allocate_buffers(nb)

    
    
    # ------------------------- Buffer ------------------------- #
    def _allocate_buffers(self, nb_bindings: int):
        self.bindings: list[int] = [None] * nb_bindings
        # Input
        self.host_in = cuda.pagelocked_empty(shape=self.input_shape, dtype=np.float32)
        self.dev_in = cuda.mem_alloc(self.host_in.nbytes)
        self.bindings[self.input_binding] = int(self.dev_in)
        # Outputs
        self.host_outs = []
        self.dev_outs = []
        for idx in self.output_bindings:
            shape = tuple(self._shape(idx))
            host_buf = cuda.pagelocked_empty(shape=shape, dtype=np.float32)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            self.bindings[idx] = int(dev_buf)
            self.host_outs.append(host_buf)
            self.dev_outs.append(dev_buf)
        # Stream
        self.stream = cuda.Stream()

    # ------------------------- 前處理 ------------------------- #
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img_bgr, (self.in_w, self.in_h))
        norm = resized.astype(np.float32) / 255.0  # (H,W,C)
        norm = np.expand_dims(norm, axis=0)        # (1,H,W,C)
        return norm

    # ------------------------- 推論 ------------------------- #
    def infer(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        input_np = self._preprocess(img_bgr)
        np.copyto(self.host_in, input_np)
        # ------------------- 將 host → device ------------------- #
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        
        # ------------------- 執行推論 --------------------------- #
        # TensorRT 9/10
        if hasattr(self.context, "execute_async_v3"):                 
            # 1. 先把每個 tensor 名稱對應到 device ptr
            for idx, dev_ptr in enumerate(self.bindings):
                name = self.engine.get_tensor_name(idx)
                self.context.set_tensor_address(name, dev_ptr)
            # 2. 執行 v3（只要給 stream_handle）
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:                                                 # 舊 API (≤ TRT 8)
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
                
        # ------------------- device → host --------------------- #
        for host, dev in zip(self.host_outs, self.dev_outs):
            cuda.memcpy_dtoh_async(host, dev, self.stream)
        self.stream.synchronize()
        return [host.copy() for host in self.host_outs]


# ------------------------- CLI 測試 ------------------------- #
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(description="RetinaFace TensorRT 單圖推論測試")
    parser.add_argument("--engine", default="retinaface.engine", help="TensorRT engine 路徑")
    parser.add_argument("--image", default="output_retina.jpg", help="測試圖片 (BGR)")
    args = parser.parse_args()

    model = RetinaFaceTRT(args.engine)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"❌ 無法讀取圖片 {args.image}")
    outs = model.infer(img)

    print("=== 推論完成 ===")
    for i, o in enumerate(outs):
        print(textwrap.dedent(f"""
        output[{i}]:
            shape  = {o.shape}
            sample = {o.flatten()[:5]} ...
        """))
