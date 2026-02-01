#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class Camera {
public:
    Camera() = default;
    Camera(const cv::Mat &K, const cv::Mat &dist);

public:
    Eigen::Matrix3d _K;
    Eigen::VectorXd _dist = Eigen::VectorXd::Zero(5);
    Eigen::Matrix3d _R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d _t = Eigen::Vector3d::Zero();
};
