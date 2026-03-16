cmake_minimum_required(VERSION 3.10)
project(CalibrateGPT CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(fisheye_stereo_calib stereo_fisheye_calibrator.cpp)
target_link_libraries(fisheye_stereo_calib PRIVATE ${OpenCV_LIBS})
