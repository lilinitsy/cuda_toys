cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME thrust_with_kernel)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

file(GLOB SRC_FILES "*.cpp" "*.cu")

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(thrust_with_kernel CUDA::cudart)
