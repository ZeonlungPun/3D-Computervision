#include "./include/commonview.h"
#include "./include/camera.h"
#include "./include/rebuild.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;


int main()
{

    std::string base_path = "/home/zonekey/project/colmap/auto_reconstruction/dense/0/images";
    std::string depth_base_path ="/home/zonekey/project/colmap/auto_reconstruction/dense/0/stereo/depth_pngs";

    // to save image address
    std::vector<std::string> rgb_images_dirs,depth_images_dirs;

    if (fs::exists(base_path) && fs::is_directory(base_path)) 
    {
        for (const auto& entry : fs::directory_iterator(base_path)) 
        {
            if (entry.is_regular_file()) 
            {
                rgb_images_dirs.push_back(entry.path().filename().string());
            }
        }
    }

    if (fs::exists(depth_base_path) && fs::is_directory(depth_base_path)) 
    {
        for (const auto& entry : fs::directory_iterator(depth_base_path)) 
        {
            if (entry.is_regular_file()) 
            {
                depth_images_dirs.push_back(entry.path().filename().string());
                
            }
        }
    }

    std::vector<cv::Mat>depthimg_list;
    depthimg_list.reserve(depth_images_dirs.size());
    for (std::string& depth_path:depth_images_dirs)
    {
        fs::path full_path = fs::path(depth_base_path) / depth_path;
        cv::Mat depth_img = cv::imread(full_path.string(),cv::IMREAD_UNCHANGED);
        depthimg_list.push_back(depth_img);

    }

    std::cout<< "depth image size:"<<depthimg_list.size()<<std::endl;


    

    // create scene graph
    CommonView commonView(rgb_images_dirs, base_path);
    int image_height = commonView.img_height;
    int image_width = commonView.img_width;
    std::cout << "Image Width: " << image_width << ", Image Height: " << image_height << std::endl;

    // set camera parameters
    // cv::Mat K = (cv::Mat_<double>(3,3) << 517.3, 0.0, 318.6,
    //                                      0.0, 516.5, 255.3,
    //                                      0.0, 0.0, 1.0);
    cv::Mat K = (cv::Mat_<float>(3,3) << 2360.0, 0.0, image_width / 2.0,
                                         0.0, 2360.0, image_height / 2.0,
                                         0.0, 0.0, 1.0);
    //cv::Mat dist = (cv::Mat_<double>(1,5) << 0.2624, -0.9513, -0.0054, 0.0026, 1.1633);
    cv::Mat dist = (cv::Mat_<double>(1,5) << 0., 0., 0., 0., 0.);
    //every camera has the same intrinsics in this dataset
    std::vector<Camera>  cameras;
    cameras.reserve(rgb_images_dirs.size());
    for(int i = 0; i < rgb_images_dirs.size();i++){
        cameras.emplace_back(K, dist);
    }
    std::cout<<"camera load!"<<std::endl;
    // incremental SfM reconstruction
    Rebuild rebuild(commonView, cameras,depthimg_list, rgb_images_dirs.size());
    rebuild.save_point_cloud("desk.csv");
    
    
    return 0;
}