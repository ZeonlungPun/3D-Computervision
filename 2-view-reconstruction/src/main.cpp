#include "extract_features.h"
#include "processor.h"
#include "structure.h"
#include <iostream>
#include <fstream>


std::vector<Eigen::Vector3d> load_points_from_csv(const std::string& filename) {
    std::vector<Eigen::Vector3d> points;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "無法打開文件: " << filename << std::endl;
        return points;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string coord_str;
        Eigen::Vector3d p;

        // 讀取 x, y, w 三個座標
        if (std::getline(ss, coord_str, ',')) {
            p[0] = std::stod(coord_str);
        }
        if (std::getline(ss, coord_str, ',')) {
            p[1] = std::stod(coord_str);
        }
        if (std::getline(ss, coord_str, ',')) {
            p[2] = std::stod(coord_str);
        }
        points.push_back(p);
    }
    file.close();
    return points;
}

int main()
{

    // std::vector<Eigen::Vector3d> hpt_g1, hpt_g2;
    // hpt_g1.push_back(Eigen::Vector3d(0,0,1));
    // hpt_g1.push_back(Eigen::Vector3d(1,0,1));
    // hpt_g1.push_back(Eigen::Vector3d(0,1,1));
    // hpt_g1.push_back(Eigen::Vector3d(1,1,1));
    // hpt_g1.push_back(Eigen::Vector3d(2,0,1));
    // hpt_g1.push_back(Eigen::Vector3d(0,2,1));
    // hpt_g1.push_back(Eigen::Vector3d(2,2,1));
    // hpt_g1.push_back(Eigen::Vector3d(3,1,1));
    // hpt_g1.push_back(Eigen::Vector3d(4,1,1));

    // hpt_g2.push_back(Eigen::Vector3d(0,0,1));
    // hpt_g2.push_back(Eigen::Vector3d(1.0,0.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(0.1,1.0,1));
    // hpt_g2.push_back(Eigen::Vector3d(1.1,1.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(2.1,0.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(0.1,2.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(2.1,2.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(3.1,1.1,1));
    // hpt_g2.push_back(Eigen::Vector3d(4.1,1.1,1));




    cv::Mat img1=cv::imread("/home/punzeonlung/PycharmProjects/3D-Reconstuction/images/dinos/viff.000.ppm");
    cv::Mat img2=cv::imread("/home/punzeonlung/PycharmProjects/3D-Reconstuction/images/dinos/viff.001.ppm");

    
    float height=img1.size[0];
    float wdith=img1.size[1];
    std::vector<std::vector<cv::Point2f>> final_points_group;
    final_points_group = find_correspondence_points(img1,img2,true);

    std::vector<cv::Point2f>pt_g1,pt_g2;
    pt_g1= final_points_group[0];
    pt_g2= final_points_group[1];
    std::cout<<pt_g1[0]<<std::endl;
   
    std::vector<Eigen::Vector3d>hpt_g1,hpt_g2;
    hpt_g1=cart2hom(pt_g1);
    hpt_g2=cart2hom(pt_g2);
    std::cout<<hpt_g1[0]<<std::endl;


    //std::vector<Eigen::Vector3d> hpt_g1 = load_points_from_csv("/home/punzeonlung/PycharmProjects/3D-Reconstuction/points1.csv");
    //std::vector<Eigen::Vector3d> hpt_g2 = load_points_from_csv("/home/punzeonlung/PycharmProjects/3D-Reconstuction/points2.csv");

    if (hpt_g1.empty() || hpt_g2.empty()) {
        std::cerr << "未能成功讀取點，請檢查文件是否存在。" << std::endl;
        return 1;
    }

    Eigen::Matrix3d intrinsic;

    intrinsic << 2360.f,0,wdith/2,
                  0,2360.f, height/2,
                  0,0,1;


    std::vector<Eigen::Vector3d>npt_g1,npt_g2;

    Eigen::Matrix3d intrinsic_inv =intrinsic.inverse();

    std::cout<<"inv:"<<intrinsic_inv<<std::endl;
    
    // transform pixel coordinate to normalized camera coordinate
    for (const Eigen::Vector3d &pt1:hpt_g1)
    {
        
        Eigen::Vector3d pt1_n=intrinsic_inv*pt1;
        npt_g1.push_back(pt1_n);
    }

    for (const Eigen::Vector3d &pt2:hpt_g2)
    {
        Eigen::Vector3d pt2_n=intrinsic_inv*pt2;
        npt_g2.push_back(pt2_n);      

    }
    std::vector<Eigen::Vector3d> npt_g1_ = npt_g1;
    std::vector<Eigen::Vector3d> npt_g2_ = npt_g2;


    Eigen::Matrix3d norm3d_1= scale_and_translate_points(npt_g1);
    Eigen::Matrix3d norm3d_2= scale_and_translate_points(npt_g2);

    //Essential Matrix
    Eigen::Matrix3d E= compute_fundamental_matrix(npt_g1,npt_g2);
    //reverse preprocessing of coordinates
    Eigen::Matrix3d norm3d_1_T= norm3d_1.transpose();
    E =  E*norm3d_2;
    E = norm3d_1_T*E;
    E = E/ E(2,2);

    std::cout<<"E:"<<E<<std::endl;

    //Decompose the fundamental  matrix E-->R, T
    //Given we are at camera 1, calculate the parameters for camera 2
    //Using the essential matrix returns 4 possible camera paramters

    Eigen::Matrix<double, 3, 4> P1;
    
    P1 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0;

    std::vector<std::pair<Eigen::Matrix3d,Eigen::Vector3d>>P2s=compute_P_from_Essential(E);

    

    //Find the correct camera parameters: 3d points are in fornt of the camera
    // P2 : WORLD coordinate --> 2nd camera coordinate
    // P_hat_h_1 : 3d point reconstructed by 1st camera coordinate
    int best_ind = -1;
    int max_positive_count = -1;

    for (int i = 0; i < (int)P2s.size(); i++)
    {
        Eigen::Matrix3d R_2 = P2s[i].first;
        Eigen::Vector3d T_2 = P2s[i].second;
        //從世界坐標到第二個相機坐標的變換矩陣
        Eigen::Matrix<double, 3, 4> P2;
        P2.block<3,3>(0,0) = R_2;
        P2.col(3) = T_2;

        std::cout<<"r1:"<<R_2<<std::endl;
        std::cout<<"t2:"<<T_2<<std::endl;

        int positive_count = 0;

        for (size_t j = 0; j < npt_g1.size(); j++)
        {
            // triangulate
            //reconstruct the 3d point in 1st camera coordinate/ world coordinat
            Eigen::Vector4d P_hat_h_1 = reconstruct_one_3dpoint(npt_g1[j], npt_g2[j], P1, P2);
            if (std::abs(P_hat_h_1[3]) < 1e-12) continue;  // 避免除以0
            P_hat_h_1 /= P_hat_h_1[3];  // normalize

            // convert to second camera coordinates
            //convert 3D point from world(1st) coordinate to second coordinate
            Eigen::Vector3d P_hat_c2 = R_2 * P_hat_h_1.head<3>() + T_2;

            // check if point is in front of both cameras
            if (P_hat_h_1[2] > 0 && P_hat_c2[2] > 0)
            {
                positive_count++;
            }
        }

        // keep the best one
        if (positive_count > max_positive_count)
        {
            max_positive_count = positive_count;
            best_ind = i;
        }
    }

    std::cout << "Best camera index: " << best_ind 
          << " with " << max_positive_count << " points in front." << std::endl;

    //best_ind=3;
    Eigen::Matrix3d R_best = P2s[best_ind].first;
    Eigen::Vector3d t_best = P2s[best_ind].second;

    Eigen::Matrix<double, 3, 4> P2_hat;
    P2_hat.block<3,3>(0,0) = R_best;
    P2_hat.col(3) = t_best;

    //Eigen::Vector3d p1_single(1.0, 2.0, 1.0);
    //Eigen::Vector3d p2_single(3.0, 4.0, 1.0);

    //calculate the innitial 3d points
    std::vector<Eigen::Vector4d>tripoints3d;

    for (size_t jj = 0; jj < npt_g1.size(); jj++)
    {
        Eigen::Vector4d P_3d_hat= linear_triangulation(npt_g1_[jj], npt_g2_[jj] ,P1, P2_hat);
        
        tripoints3d.push_back(P_3d_hat);
        
        //std::cout<<"3d:"<<P_3d_hat<<std::endl;
    
    }

    //bindle ajustment
    int iter_num = 50;
    std::vector<Eigen::Matrix<double, 3, 4>> project_matrix_list;
    project_matrix_list.push_back(P1);
    project_matrix_list.push_back(P2_hat);
    std::vector<Eigen::Vector2d> p_2d_obs_list;
    int num_points = pt_g1.size();  // 假設 pt_g1 和 pt_g2 對應點數一致
    for (int i = 0; i < num_points; i++) {
        // 第一個相機的觀測
        p_2d_obs_list.push_back(Eigen::Vector2d(pt_g1[i].x, pt_g1[i].y));
    }
    // 第二個相機的觀測
    for (int i = 0; i < num_points; i++) {
        p_2d_obs_list.push_back(Eigen::Vector2d(pt_g2[i].x, pt_g2[i].y));
    }

    LM_update(tripoints3d,project_matrix_list, p_2d_obs_list,iter_num);




    std::ofstream ofs("tripoints3d.ply");
    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << tripoints3d.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "end_header\n";

    for (auto &p : tripoints3d) {
        Eigen::Vector3d pt = p.head<3>() ;
        ofs << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    ofs.close();

    std::ofstream outfile("tripoints3d.csv");
    outfile << "X,Y,Z\n";  // CSV 標題

    for (const auto& pt : tripoints3d)
    {
        outfile << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
    }

    outfile.close();
    std::cout << "Saved tripoints3d to tripoints3d.csv" << std::endl;

    

    


    return 0;
}




