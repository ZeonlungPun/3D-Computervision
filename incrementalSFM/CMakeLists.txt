cmake_minimum_required(VERSION 3.5)
project(incrementalSFM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 尋找 OpenCV
find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH PATHS /home/punzeonlung/cvDemo/opencv480-linux/lib/cmake/opencv4)

# 2. 尋找 Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# 3. 尋找 Ceres
find_package(Ceres REQUIRED)

# 4. 設定標頭檔路徑
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# 5. 加入執行檔
add_executable(incrementalSFM
    main.cpp
    src/commonview.cpp
    src/camera.cpp
    src/rebuild.cpp
)

# 6. 連結庫 (使用 Ceres 變數而非 Target 名稱以避免版本不相容)
target_link_libraries(incrementalSFM 
    ${OpenCV_LIBS} 
    Eigen3::Eigen 
    ${CERES_LIBRARIES}
)