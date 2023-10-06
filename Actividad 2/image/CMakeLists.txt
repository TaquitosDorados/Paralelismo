cmake_minimum_required(VERSION 3.26)
project(image)

set(CMAKE_CXX_STANDARD 17)

add_executable(image main.cpp)

find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(image ${OpenCV_LIBS})