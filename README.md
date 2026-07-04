# Optical Flow Visualization (C++ / LibTorch)

Real-time optical flow estimation using the RAFT-small deep learning model, loaded in C++ via LibTorch (PyTorch's C++ API). Visualizes motion as hot-pink directional arrows and an HSV color map — hue encodes direction, brightness encodes speed.

## What it does

- Loads a TorchScript RAFT-small model via LibTorch
- Captures a live webcam feed and computes optical flow between consecutive frames
- Runs the model every 4 frames for smooth real-time performance on CPU
- Displays the webcam with motion arrows alongside a full HSV color flow map

## Color map guide

| Color | Meaning |
|-------|---------|
| Red | Moving right |
| Cyan | Moving left |
| Green | Moving down |
| Purple/Magenta | Moving up |
| Bright | Fast motion |
| Dark | Slow/no motion |

## Stack

- C++17
- LibTorch (PyTorch C++ API)
- OpenCV 4 (webcam, drawing, display)
- RAFT-small (torchvision pretrained weights, exported to TorchScript)
- CMake + vcpkg

## Setup

### 1. Install Python dependencies

```bash
pip install torch torchvision
```

### 2. Export the RAFT model (run once)

```bash
python export_raft.py
```

This downloads the pretrained RAFT-small weights and saves `raft_small.pt`.

### 3. Download LibTorch (C++ runtime)

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select:
- **Stable | Windows | LibTorch | C++ | CPU | Release**

Download and extract the zip to `C:\libtorch`.

### 4. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### 5. Copy model and OpenCV DLLs into build folder

```bash
copy ..\raft_small.pt build\
```

Also copy all non-debug DLLs from `C:\vcpkg\installed\x64-windows\bin\` into the build folder (same step as the yolo-midas-cpp project).

## Usage

```bash
cd build
.\optical_flow_cpp.exe
```

Press `q` or `Esc` to quit.

## Sample output

```
Loading RAFT model...
Model loaded. Running on CPU.
Running live — press 'q' or close window to quit.
```

Left panel shows live webcam with hot-pink motion arrows. Right panel shows the HSV color flow map.

## Robotics connection

Optical flow is a fundamental building block in robot perception — used for ego-motion estimation, obstacle detection from a moving platform, action recognition, and visual odometry. RAFT is a state-of-the-art deep flow model that outperforms classical methods (like Farneback) especially in low-texture regions and under large displacements.
