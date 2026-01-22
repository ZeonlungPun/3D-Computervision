
#ifndef EXTRACT_FEATURES_H
#define EXTRACT_FEATURES_H
#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

std::vector<std::vector<cv::Point2f>> find_correspondence_points(
    cv::Mat image1, 
    cv::Mat image2, 
    bool visualize 
);

#endif // EXTRACT_FEATURES_H