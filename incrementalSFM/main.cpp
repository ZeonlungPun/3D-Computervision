#include "./include/commonview.h"
#include "./include/rebuild.h"
#include "./include/camera.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

int main()
{
    std::string base_path = "/home/punzeonlung/PycharmProjects/3D-Reconstuction/images/dinos2";
  

    // to save image address
    std::vector<std::string> images_dirs;

    if (fs::exists(base_path) && fs::is_directory(base_path)) 
    {
        for (const auto& entry : fs::directory_iterator(base_path)) 
        {
            if (entry.is_regular_file()) 
            {
                images_dirs.push_back(entry.path().filename().string());
            }
        }
    }


    // create scene graph
    CommonView commonView(images_dirs, base_path);

    int image_height = commonView.img_height;
    int image_width = commonView.img_width;
    std::cout << "Image Width: " << image_width << ", Image Height: " << image_height << std::endl;
    // set camera parameters
    cv::Mat K = (cv::Mat_<float>(3,3) << 2360.0, 0.0, image_width / 2.0,
                                         0.0, 2360.0, image_height / 2.0,
                                         0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<float>(1,5) << 0.0, 0.0, 0.0, 0.0, 0.0);
    // every camera has the same intrinsics in this dataset
    std::vector<Camera>  cameras(int(images_dirs.size()*1.5));
    for(int i = 0; i < images_dirs.size();i++){
        Camera camera(K,dist);
        cameras[i] = camera;
    }

    // incremental SfM reconstruction
    Rebuild rebuild(commonView, cameras, images_dirs.size());
    //save
    rebuild.save_point_cloud("tripoints3d.csv");
  


    return 0;
}