cmake_minimum_required(VERSION 3.5)
project(bytetrack-opencv-cpp)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH PATHS /home/punzeonlung/cvDemo/opencv480-linux/lib/cmake/opencv4)

find_package (Eigen3  REQUIRED NO_MODULE)
include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include
   
    )
add_executable(2view_reconstruct src/extract_features.cpp
src/main.cpp
src/processor.cpp
src/structure.cpp
src/image_stich.cpp)
target_link_libraries(2view_reconstruct ${OpenCV_LIBS} Eigen3::Eigen)
