#ifndef STRUCTURE_H
#define STRUCTURE_H
#include <Eigen/Dense>
#include<iostream>
#include <vector>
#include <cmath>
#include <cassert>

Eigen::Matrix3d compute_fundamental_matrix(std::vector<Eigen::Vector3d>npt_g1,std::vector<Eigen::Vector3d>npt_g2);

Eigen::Matrix3d scale_and_translate_points(std::vector<Eigen::Vector3d>&normalize_pts);

Eigen::MatrixXd corrosponding_matrix(std::vector<Eigen::Vector3d>npt_g1,std::vector<Eigen::Vector3d>npt_g2);

std::vector<std::pair<Eigen::Matrix3d,Eigen::Vector3d>> compute_P_from_Essential(Eigen::Matrix3d E);

Eigen::Vector4d reconstruct_one_3dpoint(Eigen::Vector3d pt1,Eigen::Vector3d pt2,Eigen::Matrix<double, 3, 4> m1,Eigen::Matrix<double, 3, 4> m2);

Eigen::Vector4d linear_triangulation(Eigen::Vector3d p1,Eigen::Vector3d p2,Eigen::Matrix<double, 3, 4> m1,Eigen::Matrix<double, 3, 4> m2);

#endif