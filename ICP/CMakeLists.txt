cmake_minimum_required(VERSION 3.5)
project(ICP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV 4 REQUIRED)


find_package(Eigen3 REQUIRED NO_MODULE)



include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)


add_executable(ICP
    main.cpp
    src/commonview.cpp
    src/camera.cpp
    src/rebuild.cpp
   
)


target_link_libraries(ICP
    ${OpenCV_LIBS} 
    Eigen3::Eigen 
 
)