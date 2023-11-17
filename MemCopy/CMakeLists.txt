cmake_minimum_required(VERSION 3.26)
project(MemCopy CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(MemCopy main.cu)

set_target_properties(MemCopy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
