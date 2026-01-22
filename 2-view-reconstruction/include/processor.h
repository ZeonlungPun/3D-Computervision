#ifndef PROCESSOR_H
#define PROCESSOR_H
#include<iostream>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
std::vector<Eigen::Vector3d>cart2hom(std::vector<cv::Point2f>pt_group);

Eigen::Vector2d ReconstructionError(Eigen::Vector4d P_3d,Eigen::Matrix<double, 3, 4> project_matrix,Eigen::Vector2d p_2d_obs);

float TotalReconstructionError(std::vector<Eigen::Vector4d>& tripoints3d,
    std::vector<Eigen::Matrix<double, 3, 4>>& project_matrix_list,
    std::vector<Eigen::Vector2d>& p_2d_obs_list);

Eigen::VectorXd BA_CostFunction(std::vector<Eigen::Vector4d>tripoints3d,std::vector<Eigen::Matrix<double, 3, 4>>project_matrix_list,
    std::vector<Eigen::Vector2d>p_2d_obs_list,float lamda);

Eigen::Matrix<double, 2, 3> JacobianWrtPoint(
    const Eigen::Vector4d& P_3d,
    const Eigen::Matrix<double, 3, 4>& project_matrix);


Eigen::Matrix<double, 2, 3> JacobianWrtTranslation(
    Eigen::Matrix<double, 3, 4> project_matrix,
    const Eigen::Vector4d& P_3d);

Eigen::Matrix<double, 2, 3> JacobianWrtRotation
(Eigen::Matrix<double, 3, 4> J_T ,Eigen::Matrix<double, 3, 4> project_matrix,
    const Eigen::Vector4d& P_3d  );

void LM_update(std::vector<Eigen::Vector4d>& tripoints3d,
               std::vector<Eigen::Matrix<double, 3, 4>>& project_matrix_list,
               const std::vector<Eigen::Vector2d>& p_2d_obs_list,
               int iter_num);

Eigen::Matrix3d Skew(const Eigen::Vector3d& w);
Eigen::Matrix4d ExpSE3(const Eigen::Matrix<double,6,1>& xi);
void UpdateCameraPose(Eigen::Matrix<double,3,4>& project_matrix, const Eigen::Matrix<double,6,1>& dPose);


#endif
