#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Camera() = default;
    Camera(const cv::Mat &K, const cv::Mat &dist);

public:
    Eigen::Matrix3d _K;
    Eigen::Matrix<double, 5, 1> _dist;
    Eigen::Matrix3d _R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d _t = Eigen::Vector3d::Zero();
};
