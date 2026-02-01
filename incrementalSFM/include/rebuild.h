#pragma once
#include<iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include "commonview.h"
#include "camera.h"
#include <ceres/ceres.h>
#include "ceres/rotation.h"

struct observation{
        int cam_id; //corresponding camera id for the 2d point
        int p3d_id; //corresponding 3D point id for the 2d point
        double p2d[2] ; //coodinates in image plane
    };

class Rebuild {
public:

    Rebuild(const CommonView &commonView, const std::vector<Camera> &cameras, int image_num);

    

public:
    CommonView _commonView;
    int _images_num;
    std::vector<Camera> _cameras;
    std::vector<Eigen::Vector3d> _points_cloud; //記錄重建的所有三維點
    std::vector<int> _points_state; // 記錄每個三維點對應的的track
    std::vector<bool> _tracks_state; // 每個track是否已經重建
    std::vector<bool> _cameras_state; // 每個相機是否已經獲得參數
    std::vector<observation> _observations;
    void init();
    void save_point_cloud(const std::string &filename);
    
private:
    

    std::vector<cv::DMatch> get_success_matches(int id1, int id2);
    void init_reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches);
    void reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches);
    void ceres_bundle_adjustment();
    double reprojection_error(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera);
   
    

};