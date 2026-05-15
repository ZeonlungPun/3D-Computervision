#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;

struct Edge{
    bool flag;
    std::vector<cv::DMatch> matches;
};
struct Node{
    cv::Mat img;
    std::vector<cv::KeyPoint> keyPoints; // 特徵點
    cv::Mat descriptors; // 特徵點描述向量
    std::vector<Edge> edges; // 當前視圖和其他視圖的匹配關係
    std::vector<int> track_id; 
};


class CommonView {
public:
    CommonView(std::vector<std::string> images_dirs,std::string base_path);
    CommonView(){};
    ~CommonView(){};

private:
    void create_graph();
    void create_tracks();

    /**
     * \  根據前驅節點將後繼節點插入到tracks中
     */
    void track_insert(std::pair<int,int> predecessor, std::pair<int,int> successor);

public:
    std::vector<Node> _graph;
    std::vector<cv::Mat> _images;
    //pair.first=Image_ID, pair.second=Feature_Index
    std::vector<std::list<std::pair<int,int>>> _tracks;
    int img_width;
    int img_height;

private:
    int _image_nums = 0; //圖片/攝像機數量
 
};