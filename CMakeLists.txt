cmake_minimum_required(VERSION 3.18)

# Project Name
project(cuda_toys LANGUAGES CXX CUDA)

# Enable OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
# find_package(thrust) might not be needed
#find_package(Thrust REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

# CUDA architectures (GTX 1080, RTX 2000-4000, A100, V100, H100)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89 70 80 90)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS} -O3 --use_fast_math -lineinfo")


# Add subdirectories for programs
add_subdirectory(examples/thrust_median)

