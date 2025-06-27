#!/bin/bash
set -e

echo "📥 安裝必要套件中..."
sudo apt update
sudo apt install -y git cmake build-essential \
  libgtk-3-dev libavcodec-dev libavformat-dev \
  libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
  libpng-dev libtiff-dev 

echo "📦 下載 OpenCV 原始碼（版本 4.11.0）"
cd ~
git clone -b 4.11.0 https://github.com/opencv/opencv.git
git clone -b 4.11.0 https://github.com/opencv/opencv_contrib.git

echo "🔧 建立編譯目錄"
mkdir -p ~/opencv_build && cd ~/opencv_build

cmake ../opencv \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D ENABLE_NEON=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D BUILD_opencv_python3=ON \
  -D PYTHON_EXECUTABLE=$(which python3) \
  -D BUILD_EXAMPLES=OFF

echo "⚙️ 開始編譯︰請耐心等待"
make -j$(nproc)

echo "📦 安裝..."
sudo make install
sudo ldconfig

echo "✅ 完成！請稍候確認版本資訊"

