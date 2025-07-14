#!/usr/bin/env python3
"""
smoke_imports.py – 遞迴 import src 內所有 .py
列出「失敗的模組」與例外訊息，最後彙總。
"""
import importlib.util, pathlib, traceback, sys, os

# 設定專案根目錄，並確保將 src 加入 sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]  # 專案根目錄
os.chdir(ROOT)  # 設置當前工作目錄為專案根目錄
sys.path.append(str(ROOT / "src"))  # 添加 src 目錄到 sys.path

print(f"已將 {ROOT / 'src'} 添加到 sys.path")

TARGET_DIRS = ["retinaface_infer", "facedb", "gui_main"]

errors = []

for sub in TARGET_DIRS:
    for py in (ROOT / "src" / sub).rglob("*.py"):
        if py.name == "__init__.py":
            continue
        module_name = ".".join(py.relative_to(ROOT / "src").with_suffix("").parts)

        # 跳過會開相機的 gui_main.camera
        if module_name == "gui_main.camera":
            continue

        try:
            spec = importlib.util.spec_from_file_location(module_name, py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            errors.append((module_name, e, traceback.format_exc(limit=3)))


if errors:
    print(f"\n❌  共 {len(errors)} 個模組 import 失敗：")
    for name, err, tb in errors:
        print(f"\n—— {name} ———————————————————————————\n{err}\n{tb}")
    sys.exit(1)
else:
    print("✅  所有模組 import 成功！")

