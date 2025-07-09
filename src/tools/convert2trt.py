#!/usr/bin/env python3
"""convert2trt.py – RetinaFace ONNX ➜ TensorRT 轉換工具（支援 TensorRT 10+）

★ 2025-07-07 更新：
  - ✅ 支援 TensorRT v10 API (`build_serialized_network + deserialize_cuda_engine`)
  - ✅ 相容舊版 API (`build_engine`)：會自動偵測
  - ✅ 可跳過 pycuda，純用 TensorRT 編譯引擎

使用範例：
    python3 convert2trt.py \
        --onnx retinaface.onnx \
        --engine retinaface.engine \
        --input-shape 1x3x640x640 \
        --fp16
"""
from __future__ import annotations
import argparse
import os
import sys
import tensorrt as trt

# ------------------ PyCUDA (Optional) ------------------
try:
    import pycuda.autoinit  # noqa: F401 – 初始化 CUDA context
    import pycuda.driver as cuda  # noqa: F401
except ModuleNotFoundError:
    print("⚠️ 偵測到系統未安裝 pycuda；將以 TensorRT 內建 Driver 建構引擎，這在 *純建構* 階段完全沒問題。\n"
          "   如需日後在 Python 內直接推論並管理 GPU buffer，仍建議 `sudo pip3 install pycuda` 或使用 apt 安裝 `python3-pycuda`。")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# -------------------------- Helper 函式 -------------------------- #

def _parse_shape(shape_str: str) -> tuple[int, int, int, int]:
    try:
        n, c, h, w = map(int, shape_str.lower().split("x"))
        return n, c, h, w
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Shape 必須為 NxCxHxW，例如 1x3x640x640；收到: {shape_str}") from e


def _add_builder_config(builder: trt.Builder, fp16: bool, max_ws: int) -> trt.BuilderConfig:
    config = builder.create_builder_config()
    ws_bytes = max_ws * (1 << 20)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
    else:
        config.max_workspace_size = ws_bytes
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return config

# -------------------------- 主要流程 -------------------------- #

def build_engine(
    onnx_path: str,
    engine_path: str,
    min_shape: tuple[int, int, int, int],
    opt_shape: tuple[int, int, int, int],
    max_shape: tuple[int, int, int, int],
    fp16: bool,
    max_ws_mb: int = 2048,
) -> None:
    if not os.path.exists(onnx_path):
        sys.exit(f"❌ 找不到 ONNX 檔案：{onnx_path}")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        print(f"📖 讀取 ONNX：{onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ 解析 ONNX 失敗：")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit(1)

        if network.num_outputs == 0:
            sys.exit("❌ 模型沒有輸出節點，請檢查 ONNX 匯出流程！")

        print("🔧 建立 Optimization Profile …")
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

        config = _add_builder_config(builder, fp16, max_ws_mb)
        config.add_optimization_profile(profile)

        print("🚧 開始建構 TensorRT 引擎 …")

        if hasattr(builder, "build_serialized_network"):
            # TensorRT 8.5+ 用法
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                sys.exit("❌ 引擎序列化失敗！")
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            # 舊版 API
            engine = builder.build_engine(network, config)

        if engine is None:
            sys.exit("❌ 引擎建構失敗！")

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✅ 成功儲存引擎：{engine_path}")


# -------------------------- CLI -------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX ➜ TensorRT 轉換腳本 (RetinaFace 專用)")
    p.add_argument("--onnx", required=True, help="ONNX 模型路徑")
    p.add_argument("--engine", required=True, help="輸出 TensorRT engine 檔名")
    p.add_argument("--input-shape", default="1x3x640x640", type=_parse_shape,
                   help="最小輸入尺寸 (min_shape)，預設 1x3x640x640")
    p.add_argument("--opt-shape", type=_parse_shape,
                   help="最佳輸入尺寸 (opt_shape)，預設同 --input-shape")
    p.add_argument("--max-shape", type=_parse_shape,
                   help="最大輸入尺寸 (max_shape)，預設同 --opt-shape")
    p.add_argument("--fp16", action="store_true", help="啟用 FP16")
    p.add_argument("--workspace", type=int, default=2048,
                   help="最大工作空間 (MB)，預設 2048MB")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    opt_shape = args.opt_shape or args.input_shape
    max_shape = args.max_shape or opt_shape
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        min_shape=args.input_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        fp16=args.fp16,
        max_ws_mb=args.workspace,
    )
