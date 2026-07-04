cmake_minimum_required(VERSION 3.18)
project(optical_flow_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ─────────────────────────────────────────────────
# Download the CPU release from https://pytorch.org/get-started/locally/
# (select: Stable | Windows | LibTorch | C++ | CPU | Release)
# Extract the zip to C:\libtorch
list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)

# ── Stub out OpenCV's external dependencies ───────────────────
# OpenCV's cmake config calls find_dependency() for several packages.
# CMake 4.x has stricter version checking that breaks these lookups.
# Pre-defining the targets and FOUND variables skips the searches entirely.
if(NOT TARGET libprotobuf)
    add_library(libprotobuf INTERFACE IMPORTED)
endif()
if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf INTERFACE IMPORTED)
endif()
if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers INTERFACE IMPORTED)
    set(flatbuffers_FOUND TRUE)
endif()
if(NOT TARGET quirc)
    add_library(quirc INTERFACE IMPORTED)
    set(quirc_FOUND TRUE)
endif()
if(NOT TARGET TIFF::TIFF)
    add_library(TIFF::TIFF INTERFACE IMPORTED)
    set(TIFF_FOUND TRUE)
endif()

# ── OpenCV (via vcpkg — already installed from previous project) ──
set(OpenCV_ROOT "C:/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

# ── Executable ───────────────────────────────────────────────
add_executable(optical_flow_cpp main.cpp)

target_link_libraries(optical_flow_cpp
    PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

target_include_directories(optical_flow_cpp
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# ── Copy LibTorch DLLs to build folder (Windows only) ────────
# Without this the exe won't find the torch DLLs at runtime.
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET optical_flow_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:optical_flow_cpp>
    )
endif()
