#!/usr/bin/env python3
# scripts/menu.py

import os
import subprocess

def run_command(cmd):
    print(f"\n▶ 執行：{cmd}\n")
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n⏹️ 已中斷")
    input("\n✅ 任務完成，按 Enter 回到選單...")

def submenu(title, options):
    while True:
        print(f"\n📂 {title}")
        for i, (label, _) in enumerate(options, 1):
            print(f"  {i}. {label}")
        print(f"  0. 🔙 返回上一層")

        choice = input("請選擇功能：").strip()
        if choice == "0":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
            print("❌ 無效選項")
            continue
        _, cmd = options[int(choice)-1]
        run_command(cmd)

def main_menu():
    while True:
        print("\n📋 Jetson 人臉辨識工具選單")
        print("  1. 👁️  即時辨識功能")
        print("  2. 🧰  資料管理工具")
        print("  3. 🔧  攝影機 / 系統測試")
        print("  0. ❎ 離開")

        choice = input("請選擇功能：").strip()
        if choice == "0":
            print("👋 Bye!")
            break
        elif choice == "1":
            submenu("即時辨識功能", [
                ("啟動 GUI 人臉辨識", "PYTHONPATH=. python3 src/main.py"),
            ])
        elif choice == "2":
            submenu("資料管理工具", [
                ("建立 faces.pkl 特徵庫", "python3 scripts/regenerate_faces.py"),
                ("照片分類（有臉 / 無臉）", "python3 scripts/batch_update_faces.py"),
            ])
        elif choice == "3":
            submenu("攝影機 / 系統測試", [
                ("測試攝影機畫面", "python3 cv_test.py"),
                #("即時 GPU 使用率 (Ctrl+C 離開)", "tegrastats --interval 1000"),
            ])
        else:
            print("❌ 無效選項")

if __name__ == "__main__":
    main_menu()

