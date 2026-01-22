#include <iostream>
#include <opencv2/opencv.hpp>
#include "extract_features.h" // 假設這些頭文件存在且正確
#include "processor.h"      // 假設這些頭文件存在且正確
#include <random>
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <algorithm> // for std::round

// 修正後的 stitch_images 函數，讓 img1 作為 source 變換到 img2 的視角
cv::Mat stitch_images(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& H_1_to_2_cv) {
    // H_1_to_2_cv 是從 img1 到 img2 的單應性矩陣
    // 我們需要變換 img1，並將 img2 放置在正確的位置

    // 獲取兩張圖片的尺寸
    int h1 = img1.rows;
    int w1 = img1.cols;
    int h2 = img2.rows;
    int w2 = img2.cols;

    // 定義 img1 的四個角點
    cv::Mat corners_img1(1, 4, CV_32FC2);
    corners_img1.at<cv::Point2f>(0, 0) = cv::Point2f(0, 0);       // 左上角
    corners_img1.at<cv::Point2f>(0, 1) = cv::Point2f(w1, 0);       // 右上角
    corners_img1.at<cv::Point2f>(0, 2) = cv::Point2f(w1, h1);       // 右下角
    corners_img1.at<cv::Point2f>(0, 3) = cv::Point2f(0, h1);       // 左下角

    // 將 img1 的四個角點進行透視變換 (從 img1 到 img2 的坐標系)
    cv::Mat transformed_corners_img1;
    cv::perspectiveTransform(corners_img1, transformed_corners_img1, H_1_to_2_cv);

    // 找到拼接後圖片的邊界
    // 初始邊界為 img2 的邊界
    float minX = 0, minY = 0, maxX = w2, maxY = h2;

    // 結合 img1 變換後的邊界
    for (int i = 0; i < 4; i++) {
        float x = transformed_corners_img1.at<cv::Point2f>(0, i).x;
        float y = transformed_corners_img1.at<cv::Point2f>(0, i).y;
        minX = std::min(minX, x);
        minY = std::min(minY, y);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
    }
    
    std::cout << "[stitch_images] Initial MinX: " << minX << ", MinY: " << minY << ", MaxX: " << maxX << ", MaxY: " << maxY << std::endl;

    // 計算拼接後圖片的總寬度和總高度
    // 確保尺寸為正且足夠大
    int result_width = static_cast<int>(std::round(maxX - minX));
    int result_height = static_cast<int>(std::round(maxY - minY));

    std::cout << "[stitch_images] Calculated Result Width: " << result_width << ", Result Height: " << result_height << std::endl;

    if (result_width <= 0 || result_height <= 0) {
        std::cerr << "[stitch_images] Error: Calculated result image dimensions are invalid (" 
                  << result_width << "x" << result_height << "). Returning empty image." << std::endl;
        return cv::Mat(); 
    }

    // 創建一個新的平移矩陣，用來確保所有圖像內容都在新的畫布中
    cv::Mat H_translate = cv::Mat::eye(3, 3, CV_64FC1);
    H_translate.at<double>(0, 2) = -minX; 
    H_translate.at<double>(1, 2) = -minY; 

    std::cout << "[stitch_images] H_translate:\n" << H_translate << std::endl;

    // 應用平移到 img1 的變換矩陣上
    cv::Mat H_total_img1 = H_translate * H_1_to_2_cv;

    // 應用平移到 img2 (img2 保持原始姿態，只進行畫布平移)
    cv::Mat H_total_img2 = H_translate; 

    // 創建結果畫布，初始化為黑色
    cv::Mat result_image(result_height, result_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // 將 img2 貼到結果畫布上 (作為背景)
    cv::Mat img2_warped;
    cv::warpPerspective(img2, img2_warped, H_total_img2, cv::Size(result_width, result_height));
    std::cout << "[stitch_images] img2_warped empty? " << img2_warped.empty() << std::endl;
    
    // 將 img1 變換並貼到結果畫布上
    cv::Mat img1_warped;
    cv::warpPerspective(img1, img1_warped, H_total_img1, cv::Size(result_width, result_height));
    std::cout << "[stitch_images] img1_warped empty? " << img1_warped.empty() << std::endl;


    // 融合兩張圖片
    // 現在 img2_warped 已經是帶有平移的背景，img1_warped 是變形並帶有平移的前景
    // 疊加時，讓 img1_warped 的內容優先顯示在重疊區域
    for (int y = 0; y < result_height; ++y) {
        for (int x = 0; x < result_width; ++x) {
            cv::Vec3b pixel1_warped = img1_warped.at<cv::Vec3b>(y, x);
            cv::Vec3b pixel2_warped = img2_warped.at<cv::Vec3b>(y, x);

            if (pixel1_warped != cv::Vec3b(0, 0, 0)) { // 如果變形後的 img1 有像素
                result_image.at<cv::Vec3b>(y, x) = pixel1_warped;
            } else { // 否則使用變形後的 img2 的像素
                result_image.at<cv::Vec3b>(y, x) = pixel2_warped;
            }
        }
    }

    return result_image;
}

std::vector<int> sample_indices_set(int n, int k) {
    std::unordered_set<int> selected;
    std::vector<int> result;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, n - 1);

    while ((int)selected.size() < k) {
        int idx = dist(gen);
        if (selected.insert(idx).second) {
            result.push_back(idx);
        }
    }
    return result;
}

std::vector<Eigen::Vector3d> project_points(std::vector<Eigen::Vector3d>hpt_g1,Eigen::Matrix3d h)
{
    std::vector<Eigen::Vector3d>hpt_hat;
    for(const Eigen::Vector3d& pt : hpt_g1) // 使用 const& 避免複製
    {
        Eigen::Vector3d pred_pt=h*pt;
        // 齊次歸一化
        pred_pt /= pred_pt(2);
        hpt_hat.push_back(pred_pt);

    }
    return hpt_hat;
}

double calculate_inliner_ratio(std::vector<Eigen::Vector3d>hpt_g2_pred,std::vector<Eigen::Vector3d>hpt_g2,double threshold)
{
    double total_inliner_num=0;
    for (int i=0;i<hpt_g2.size();i++)
    {
        Eigen::Vector3d true_vec=hpt_g2[i];
        Eigen::Vector3d pred_vec=hpt_g2_pred[i];
        double error= (true_vec- pred_vec).norm();
        if(error < threshold)
        {
            total_inliner_num+=1;
        }
    }
    double inliner_ratio= total_inliner_num/hpt_g2.size();
    return inliner_ratio;

}

std::vector<std::pair<Eigen::Matrix3d,double>>RANSAN_for_Homogeneous(std::vector<Eigen::Vector3d>hpt_g1,std::vector<Eigen::Vector3d>hpt_g2,
    double inliner_threshold=10.0,int RANSAC_num=5,int iter_num=1000)
{
    std::vector<std::pair<Eigen::Matrix3d,double>>result_dict;
    int num_points= hpt_g1.size();
    
    if (num_points < RANSAC_num) {
        std::cerr << "[RANSAC] Error: Not enough points (" << num_points << ") for RANSAC_num (" << RANSAC_num << "). Skipping RANSAC." << std::endl;
        return result_dict; // 返回空結果
    }

    for(int iter=0;iter<iter_num;iter++)
    {
        //3.1 get RANSAC_num random value (6 coordinate number for RANSAC)
        std::vector<int>sample_indices=sample_indices_set(num_points,RANSAC_num);
        Eigen::MatrixXd A(RANSAC_num*2, 9);
        int num=0;
        for (const int id : sample_indices) // 使用 const& 避免複製
        {
            double p1x = hpt_g1[id][0];
            double p1y = hpt_g1[id][1];
            double p2x = hpt_g2[id][0];
            double p2y = hpt_g2[id][1];
        //3.2, create the matrix Ah=0, where A is nx9 and h is 9x1
            A(2*num,0) = p1x;
            A(2*num,1) = p1y;
            A(2*num,2) = 1;
            A(2*num,3) = 0;
            A(2*num,4) = 0;
            A(2*num,5) = 0;
            A(2*num,6) = -p2x*p1x;
            A(2*num,7) = -p2x*p1y;
            A(2*num,8) = -p2x;

            A(2*num+1,0) = 0;
            A(2*num+1,1) = 0;
            A(2*num+1,2) = 0;
            A(2*num+1,3) = p1x;
            A(2*num+1,4) = p1y;
            A(2*num+1,5) = 1;
            A(2*num+1,6) = -p2y*p1x;
            A(2*num+1,7) = -p2y*p1y;
            A(2*num+1,8) = -p2y;
            num+=1;
        }
        //3.3, find the solution of Ah=0 using SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd V = svd.matrixV(); 
        Eigen::VectorXd h_hat_vector = V.col(V.cols() - 1);
        Eigen::Matrix3d h_hat_matrix = h_hat_vector.reshaped(3, 3).transpose();
        // 齊次化 (normalize)
        if (h_hat_matrix(2,2) != 0) { // 避免除以零
            h_hat_matrix /= h_hat_matrix(2, 2);
        } else {
            // 如果 h(2,2) 為零，說明單應性矩陣可能不合理，跳過此迭代
            continue; 
        }
        

        //3.4 reproject the source points to destination plane and compare them
        std::vector<Eigen::Vector3d>hpt_g2_pred =project_points(hpt_g1,h_hat_matrix);

        //3.5 calculate rMSE between projected points and real points
        // and get the inliner ratio
        double inliner_ratio = calculate_inliner_ratio(hpt_g2_pred,hpt_g2,inliner_threshold);
        //std::cout<<"ratio:"<<inliner_ratio<<std::endl;
        std::pair<Eigen::Matrix3d,double> temp_result= std::make_pair(h_hat_matrix,inliner_ratio);
        result_dict.push_back(temp_result);

    }
    return result_dict;
}


void main_stich_process(cv::Mat img1,cv::Mat img2)
{
    std::cout << "Starting main_stich_process..." << std::endl;
    //1, find corrosponding matching points in two image
    std::vector<std::vector<cv::Point2f>> final_points_group;
    final_points_group = find_correspondence_points(img1,img2,true);

    std::vector<cv::Point2f>pt_g1,pt_g2;
    pt_g1= final_points_group[0];
    pt_g2= final_points_group[1];
    
    std::cout << "Number of matched points found: " << pt_g1.size() << std::endl;
    if (pt_g1.size() < 4) { 
        std::cerr << "Error: Not enough matched points (" << pt_g1.size() << ") for homography estimation. Minimum 4 required." << std::endl;
        return;
    }
    
    //2, transform to homogeneous coordinates
    std::vector<Eigen::Vector3d>hpt_g1,hpt_g2;
    hpt_g1=cart2hom(pt_g1);
    hpt_g2=cart2hom(pt_g2);

    // 3, use RANSAC to solve the problem Ah=0, where A is nx9 and h is 9x1
    std::cout << "Running RANSAC for Homography estimation..." << std::endl;
    std::vector<std::pair<Eigen::Matrix3d,double>>result_dict=RANSAN_for_Homogeneous(hpt_g1,hpt_g2);

    if (result_dict.empty()) {
        std::cerr << "Error: RANSAC failed to estimate any valid homography matrix. Result dictionary is empty." << std::endl;
        return;
    }

    double max_ratio = -1.0;
    Eigen::Matrix3d h_hat_matrix;
    bool found_best_h = false;

    for (const auto& pair : result_dict) {
        if (pair.second > max_ratio) {
            max_ratio = pair.second;
            h_hat_matrix = pair.first;
            found_best_h = true;
        }
    }

    if (!found_best_h) {
        std::cerr << "Error: No best homography matrix found from RANSAC results." << std::endl;
        return;
    }
    std::cout << "Best RANSAC Inlier Ratio: " << max_ratio << std::endl;
    std::cout << "Best Estimated Homography Matrix (Eigen):\n" << h_hat_matrix << std::endl;


    // 4, use homogeneous matrix h to stich two images
    // 核心修正：將 Eigen::Matrix3d 複製到 cv::Mat 時，確保正確的元素對應
    cv::Mat H_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H_cv.at<double>(i, j) = h_hat_matrix(i, j);
        }
    }
    
    // 檢查 H_cv 是否包含 NaN 或 inf
    if (!cv::checkRange(H_cv, true)) {
        std::cerr << "Error: Final Homography matrix H_cv contains invalid values (NaN/Inf). Cannot proceed with stitching." << std::endl;
        return;
    }

    std::cout << "Homography Matrix (OpenCV format, H_1_to_2_cv):\n" << H_cv << std::endl;


    cv::Mat result_image= stitch_images( img1,  img2,  H_cv); // 將 H_cv 傳遞給 stitch_images
    
    if (result_image.empty()) {
        std::cerr << "Error: stitch_images returned an empty image. Stitching failed." << std::endl;
        return;
    }

    std::cout << "Stitching completed. Saving result to stich_result.png" << std::endl;
    cv::imwrite("stich_result.png",result_image);
    std::cout << "Image saved." << std::endl;
}

// int main()
// {   
//     cv::Mat img1=cv::imread("/home/punzeonlung/CPP/2-view-reconstruction/1.jpg");
//     cv::Mat img2=cv::imread("/home/punzeonlung/CPP/2-view-reconstruction/2.jpg");

//     // 檢查圖片是否成功讀取
//     if (img1.empty()) {
//         std::cerr << "Error: Image 1 not loaded from path. Please check the path and file existence." << std::endl;
//         return -1;
//     }
//     if (img2.empty()) {
//         std::cerr << "Error: Image 2 not loaded from path. Please check the path and file existence." << std::endl;
//         return -1;
//     }

//     std::cout << "Image 1 size: " << img1.cols << "x" << img1.rows << std::endl;
//     std::cout << "Image 2 size: " << img2.cols << "x" << img2.rows << std::endl;

//     main_stich_process(img1,img2);
//     return 0;
// }
