cmake_minimum_required(VERSION 2.8)
project(challenge)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(challenge main.cpp)
target_link_libraries(challenge ${OpenCV_LIBS})

