#include "camera.h"

Camera::Camera(const cv::Mat &K, const cv::Mat &dist)
{
    // for(int i = 0; i < 3; i++){
    //     for(int j = 0; j < 3; j++){
    //         this->_K(i,j) = K.at<double>(i,j);
    //     }
    // }
    cv::cv2eigen(K, this->_K);

    for(int i = 0; i < 5 ;i++){
        this->_dist(i) = dist.at<double>(0,i);

    }
}