#include "commonview.h"
#include "rebuild.h"

bool edegeExists(const CommonView &commonView)
{
    for (int i = 0; i < commonView._graph.size(); i++) {
        for (int j = 0; j < commonView._graph[i].edges.size(); j++) {
            if (commonView._graph[i].edges[j].flag == true) {
                return true;
            }
        }
    }
    return false;
}



Rebuild::Rebuild(const CommonView &commonView, const std::vector<Camera> &cameras,std::vector<cv::Mat>&_depth_images, int image_num)
{
    this->_commonView = commonView;
    this->_images_num = image_num;
    this->_cameras = cameras;
    this->_depth_images =_depth_images;
    this->_view_clouds.resize(this->_images_num);

    this->_tracks_state.resize(this->_commonView._tracks.size());
    for (int i = 0; i < this->_tracks_state.size(); i++) {
        this->_tracks_state[i] = false;
    }
    this->_cameras_state.resize(this->_images_num);
    for (int i = 0; i < this->_cameras_state.size(); i++) {
        this->_cameras_state[i] = false;
    } 
    
    this->init();

};

void Rebuild::init()
{
    // --------------------------------------------------
    // 1) 找初始影像對
    // --------------------------------------------------
    int id1 = -1, id2 = -1;
    std::vector<cv::DMatch> success_matches;

    for (int i = 0; i < this->_images_num; i++) {
        for (int j = i + 1; j < this->_images_num; j++) {
            std::vector<cv::DMatch> temp_matches = this->get_success_matches(i, j);
            if (temp_matches.size() > success_matches.size()) {
                success_matches = temp_matches;
                id1 = i;
                id2 = j;
            }
        }
    }

    if (id1 < 0 || id2 < 0 || success_matches.empty()) {
        std::cerr << "No valid initial image pair found." << std::endl;
        return;
    }

    std::cout << "Initial image pair: " << id1
              << " and " << id2
              << " with " << success_matches.size()
              << " matches." << std::endl;

    // 初始重建
    this->_cameras_state[id1] = true;
    this->_cameras_state[id2] = true;
    this->init_reconstruct(id1, id2, success_matches,1.0);

    // 記錄哪些視圖已經嘗試過但失敗了，避免一直重複選到同一張
    std::vector<bool> camera_tried(this->_images_num, false);

    // --------------------------------------------------
    // 2) 逐步加入更多視圖
    // --------------------------------------------------
    int reconstructed_counts = 1;

    while (true)
    {
        reconstructed_counts += 1;
        std::cout << "Reconstruction iteration: "
                  << reconstructed_counts << std::endl;

        // 計算每個 view 被已重建 3D 點觀測到的次數
        std::vector<int> views_num(this->_images_num, 0);

        for (int i = 0; i < this->_points_state.size(); i++) {
            int track_id = this->_points_state[i];
            if (track_id < 0 || track_id >= (int)this->_tracks_state.size())
                continue;

            this->_tracks_state[track_id] = true;

            for (auto &e : this->_commonView._tracks[track_id]) {
                int img_id = e.first;
                if (img_id >= 0 && img_id < this->_images_num) {
                    views_num[img_id] += 1;
                }
            }
        }

        // 選擇尚未重建、尚未嘗試失敗、且 track 最多的視圖
        int max_val = 0;
        int max_img_id = -1;

        for (int i = 0; i < this->_images_num; i++) {
            if (!this->_cameras_state[i] &&
                !camera_tried[i] &&
                views_num[i] > max_val)
            {
                max_img_id = i;
                max_val = views_num[i];
            }
        }

        // 沒有可加入的視圖了
        if (max_img_id == -1 || max_val == 0) {
            std::cout << "No more valid views to add. Stop." << std::endl;
            break;
        }

        std::cout << "Adding view " << max_img_id
                  << " with " << max_val
                  << " tracks." << std::endl;

        // 3) 建立 3D-2D correspondences
        std::vector<cv::Point3d> p3ds;
        std::vector<cv::Point2d> p2ds;

        for (int i = 0; i < this->_points_state.size(); i++)
        {
            int track_id = this->_points_state[i];
            if (track_id < 0 || track_id >= (int)this->_commonView._tracks.size())
                continue;

            for (auto &e : this->_commonView._tracks[track_id])
            {
                int img_id = e.first;
                int kp_id  = e.second;

                if (img_id == max_img_id) {
                    if (kp_id < 0 || kp_id >= (int)this->_commonView._graph[max_img_id].keyPoints.size())
                        continue;

                    const Eigen::Vector3d &p3d = this->_points_cloud[i];
                    p3ds.push_back(cv::Point3d(p3d(0), p3d(1), p3d(2)));

                    double p2d_x = this->_commonView._graph[max_img_id].keyPoints[kp_id].pt.x;
                    double p2d_y = this->_commonView._graph[max_img_id].keyPoints[kp_id].pt.y;
                    p2ds.push_back(cv::Point2d(p2d_x, p2d_y));
                }
            }
        }

        std::cout << "Found " << p3ds.size()
                  << " valid 3D-2D correspondences." << std::endl;

        // solvePnPRansac 至少需要 4 組對應點
        if (p3ds.size() < 4 || p2ds.size() < 4 || p3ds.size() != p2ds.size()) {
            std::cout << "Skip view " << max_img_id
                      << " because correspondences are insufficient." << std::endl;
            camera_tried[max_img_id] = true;
            continue;
        }

        // --------------------------------------------------
        // 4) Solve PnP
        // --------------------------------------------------
        Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
        cv::Mat rvec, tvec, inliers;
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            K_eigen(0,0), K_eigen(0,1), K_eigen(0,2),
            K_eigen(1,0), K_eigen(1,1), K_eigen(1,2),
            K_eigen(2,0), K_eigen(2,1), K_eigen(2,2));

        try {
            cv::solvePnPRansac(
                p3ds,
                p2ds,
                K,
                cv::Mat(),
                rvec,
                tvec,
                false,
                100,
                8.0,
                0.99,
                inliers,
                cv::SOLVEPNP_EPNP
            );
        }
        catch (const cv::Exception &e) {
            std::cerr << "solvePnPRansac failed for view "
                      << max_img_id << ": " << e.what() << std::endl;
            camera_tried[max_img_id] = true;
            continue;
        }

        if (inliers.rows < 4) {
            std::cout << "Skip view " << max_img_id
                      << " because inliers < 4." << std::endl;
            camera_tried[max_img_id] = true;
            continue;
        }

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R_eigen;
        Eigen::Vector3d t_eigen;

        for (int r = 0; r < 3; r++) {
            t_eigen(r) = tvec.at<double>(r, 0);
            for (int c = 0; c < 3; c++) {
                R_eigen(r, c) = R_cv.at<double>(r, c);
            }
        }

        // update camera pose
        this->_cameras[max_img_id]._R = R_eigen;
        this->_cameras[max_img_id]._t = t_eigen;
        this->_cameras_state[max_img_id] = true;

        std::cout << "Camera " << max_img_id
                  << " pose estimated successfully." << std::endl;

        // --------------------------------------------------
        // 5) 用新相機與已知相機重建新點
        // --------------------------------------------------
        for (int i = 0; i < this->_images_num; i++)
        {
            if (!this->_cameras_state[i]) continue;
            if (i == max_img_id) continue;

            int a = i;
            int b = max_img_id;
            if (a > b) std::swap(a, b);

            if (this->_commonView._graph[a].edges[b].flag == true) {
                this->_commonView._graph[a].edges[b].flag = false;
                this->_commonView._graph[b].edges[a].flag = false;

                success_matches = this->get_success_matches(a, b);
                if (success_matches.size() > 100) {
                    this->reconstruct(max_img_id, i, success_matches,3000.0);
                }
            }
        }
    }

    for (int i=0;i< this->_images_num;i++)
    {
        this->refinePoseByICP(i);
    }
}

std::vector<cv::DMatch> Rebuild::get_success_matches(int id1, int id2)
{
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }

    std::vector<cv::DMatch> temp_matches1;
    Node node1 = this->_commonView._graph[id1];
    Node node2 = this->_commonView._graph[id2];


    std::vector<cv::DMatch> success_matches;
    std::vector<cv::DMatch> matches = this->_commonView._graph[id1].edges[id2].matches;
    for (int i = 0; i < matches.size(); i++) {
        int queryIdx = matches[i].queryIdx;//第一幅圖的特徵點id;
        int trainIdx = matches[i].trainIdx;//第二幅圖的特徵點id;
        int track_id1 = node1.track_id[queryIdx];
        int track_id2 = node2.track_id[trainIdx];
        if (track_id1 != track_id2) {
            continue;
        }
        if (this->_commonView._tracks[track_id1].size() > 2 && this->_tracks_state[track_id1] == false) {
            success_matches.push_back(matches[i]);
        }
    }
    return success_matches;
}

void Rebuild::init_reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches,double depth_scale)
{
    if (id1 > id2) {
        std::swap(id1, id2);
    }
    
    Node node1 = this->_commonView._graph[id1];
    Node node2 = this->_commonView._graph[id2];
    
    // 取得相機內參
    Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
    cv::Mat K = (cv::Mat_<double>(3,3) << 
        K_eigen(0,0), K_eigen(0,1), K_eigen(0,2),
        K_eigen(1,0), K_eigen(1,1), K_eigen(1,2),
        K_eigen(2,0), K_eigen(2,1), K_eigen(2,2));

    double fx = K_eigen(0,0), fy = K_eigen(1,1);
    double cx = K_eigen(0,2), cy = K_eigen(1,2);
   

    // 假設深度圖存在這裡
    cv::Mat depth1 = this->_depth_images[id1];

    std::vector<cv::Point3d> p3ds;
    std::vector<cv::Point2d> p2ds_img2;
    std::vector<cv::Point2d> p2ds_img1;
    std::vector<int> valid_match_indices; // 記錄有效的 match index，方便後續更新 state

    // 1. 利用 Image 1 的深度圖生成 3D 點
    for (int i = 0; i < success_matches.size(); i++) {
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx = success_matches[i].trainIdx;
        
        cv::Point2d pt1 = node1.keyPoints[queryIdx].pt;
        cv::Point2d pt2 = node2.keyPoints[trainIdx].pt;

        // 讀取深度值 (需對浮點數座標四捨五入)
        int u = std::round(pt1.x);
        int v = std::round(pt1.y);
        
        if (u < 0 || u >= depth1.cols || v < 0 || v >= depth1.rows) continue;
        
        unsigned short d_raw = depth1.at<unsigned short>(v, u);
        if (d_raw == 0) continue; // 深度無效

        double d = d_raw / depth_scale; // 轉換為米(m)

        // 相機 1 座標系下的 3D 點 (因為 Cam 1 是原點，所以這也就是世界座標)
        double X = (pt1.x - cx) * d / fx;
        double Y = (pt1.y - cy) * d / fy;
        double Z = d;

        p3ds.push_back(cv::Point3d(X, Y, Z));
        p2ds_img2.push_back(pt2);
        p2ds_img1.push_back(pt1);
        valid_match_indices.push_back(i);
    }

    if (p3ds.size() < 10) {
        std::cerr << "Not enough valid depth points for initial PnP!" << std::endl;
        return;
    }

    // 2. 設置相機 1 的姿態為原點
    this->_cameras[id1]._R = Eigen::Matrix3d::Identity();
    this->_cameras[id1]._t = Eigen::Vector3d::Zero();

    // 3. 用 PnP 算出包含真實尺度的相機 2 外參
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(p3ds, p2ds_img2, K, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);

    cv::Mat R12;
    cv::Rodrigues(rvec, R12);

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    for (int r = 0; r < 3; r++) {
        t_eigen(r) = tvec.at<double>(r, 0);
        for (int c = 0; c < 3; c++) {
            R_eigen(r, c) = R12.at<double>(r, c);
        }
    }
    this->_cameras[id2]._R = R_eigen;
    this->_cameras[id2]._t = t_eigen;   

    // 4. 將 PnP 的 Inliers 存入global cloud / each-view cloud
    for (int i = 0; i < inliers.rows; i++) {
        int idx = inliers.at<int>(i, 0);
        int match_idx = valid_match_indices[idx];
        
        cv::Point3d& p = p3ds[idx];
        Eigen::Vector3d point3d(p.x, p.y, p.z);
        // global cloud
        this->_points_cloud.push_back(point3d);
        const cv::DMatch& match = success_matches[match_idx];

        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;

        // world point: 直接用 id1 相機座標下的點，
        // 因為 init 時 id1 被設為世界原點
        Eigen::Vector3d point_c1(p.x, p.y, p.z);
        Eigen::Vector3d point_w  = point_c1;
        
        // cam2 座標下的同一個點
        Eigen::Vector3d point_c2 = this->_cameras[id2]._R * point_w + this->_cameras[id2]._t;
        // global cloud
        this->_points_cloud.push_back(point_w);

        // view id1 cloud
        this->_view_clouds[id1].points_c.push_back(point_c1);
        this->_view_clouds[id1].points_w.push_back(point_w);
        this->_view_clouds[id1].pixels.push_back(node1.keyPoints[queryIdx].pt);
        this->_view_clouds[id1].track_ids.push_back(this->_commonView._graph[id1].track_id[queryIdx]);

         // view id2 cloud
        this->_view_clouds[id2].points_c.push_back(point_c2);
        this->_view_clouds[id2].points_w.push_back(point_w);
        this->_view_clouds[id2].pixels.push_back(node2.keyPoints[trainIdx].pt);
        this->_view_clouds[id2].track_ids.push_back(this->_commonView._graph[id1].track_id[queryIdx]);

        // track / point state
        int track_id = this->_commonView._graph[id1].track_id[queryIdx];
        this->_points_state.push_back(track_id);
        this->_tracks_state[track_id] = true;   
    }

    // 取消這兩視圖的邊
    this->_commonView._graph[id1].edges[id2].flag = false;
    this->_commonView._graph[id2].edges[id1].flag = false;  

    std::cout << "Initial reconstruction completed with " << inliers.rows << " 3D points." << std::endl;
}


void Rebuild::reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches,double depth_scale)
{
    if (id1 > id2) {
        std::swap(id1, id2);
    }
    
    Node node1 = this->_commonView._graph[id1];
    Node node2 = this->_commonView._graph[id2];
    
    Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
    double fx = K_eigen(0,0), fy = K_eigen(1,1);
    double cx = K_eigen(0,2), cy = K_eigen(1,2);
   

    cv::Mat depth1 = this->_depth_images[id1];
    cv::Mat depth2 = this->_depth_images[id2];

    Eigen::Matrix3d R1 = this->_cameras[id1]._R;
    Eigen::Vector3d t1 = this->_cameras[id1]._t;
    
    Eigen::Matrix3d R2 = this->_cameras[id2]._R;
    Eigen::Vector3d t2 = this->_cameras[id2]._t;

    for (int i = 0; i < success_matches.size(); i++) {
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx =success_matches[i].trainIdx;
        int track_id = this->_commonView._graph[id1].track_id[queryIdx];
        
        // 如果這個 track 已經重建過了，就跳過
        if (this->_tracks_state[track_id]) continue;
        // image points
        cv::Point2d pt1 = node1.keyPoints[queryIdx].pt;
        cv::Point2d pt2 = node2.keyPoints[trainIdx].pt;
        
        int u = std::round(pt1.x);
        int v = std::round(pt1.y);
        
        if (u < 0 || u >= depth1.cols || v < 0 || v >= depth1.rows) continue;
        
        unsigned short d_raw = depth1.at<unsigned short>(v, u);
        
        // 如果 id1 深度缺失，你可以選擇用 id2 的深度反推 (這裡為了簡潔僅檢查 id1)
        if (d_raw == 0) continue; 

        double d = d_raw / depth_scale;

        // 計算在相機 1 座標系下的位置 P_c1
        Eigen::Vector3d p_c1;
        p_c1(0) = (pt1.x - cx) * d / fx;
        p_c1(1) = (pt1.y - cy) * d / fy;
        p_c1(2) = d;

        // 轉換為世界座標系 P_w = R^T * (P_c - t)
        Eigen::Vector3d p_w = R1.transpose() * (p_c1 - t1);
        // transform to camera2 coordinate
        Eigen::Vector3d p_c2 = R2 * p_w + t2;
        // --- 深度與視角檢查 (Cheirality Check) ---
        // 確保此點在相機 2 的前方
        if (p_c2.z() <= 0.001) {
            continue; 
        }

        // 更新點雲與狀態
        // global sparse cloud
        this->_points_cloud.push_back(p_w);
        this->_points_state.push_back(track_id);
        this->_tracks_state[track_id] = true; 
        // view cloud : id1
        this->_view_clouds[id1].points_c.push_back(p_c1);
        this->_view_clouds[id1].points_w.push_back(p_w);
        this->_view_clouds[id1].pixels.push_back(pt1);
        this->_view_clouds[id1].track_ids.push_back(track_id);  
        // view cloud : id2
        this->_view_clouds[id2].points_c.push_back(p_c2);
        this->_view_clouds[id2].points_w.push_back(p_w);
        this->_view_clouds[id2].pixels.push_back(pt2);
        this->_view_clouds[id2].track_ids.push_back(track_id);

    }
    std::cout
        << "Reconstruction between "
        << id1
        << " and "
        << id2
        << " finished."
        << std::endl;

}


void pose_estimation_3d3d(const std::vector<Eigen::Vector3d>& pts_src,
                          const std::vector<Eigen::Vector3d>& pts_tar,
                          Eigen::Matrix3d& R,Eigen::Vector3d& t)
{
    // center of mass
    Eigen::Vector3d p1, p2;
    int N = pts_src.size();
    for (int i=0; i<N; i++)
    {
        p1 += pts_src[i];
        p2 += pts_tar[i];
    }
    p1 /= N;
    p2 /= N;

    // subtract COM
    std::vector<Eigen::Vector3d> q1(N), q2(N);
    for (int i=0; i<N; i++)
    {
        q1[i] = pts_src[i] - p1;
        q2[i] = pts_tar[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i=0; i<N; i++)
    {
        W += Eigen::Vector3d(q1[i][0], q1[i][1], q1[i][2]) * Eigen::Vector3d(q2[i][0],
                q2[i][1], q2[i][2]).transpose();
    }
   

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    

    R = U * (V.transpose());
    t = Eigen::Vector3d(p1[0], p1[1], p1[2]) - R * Eigen::Vector3d(p2[0], p2[1], p2[2]);

 
}

void Rebuild::refinePoseByICP(int img_id)
{
    
    // check
    if (!this->_cameras_state[img_id]) {
        std::cerr
            << "Camera not initialized."
            << std::endl;
        return;
    }

    if (this->_view_clouds[img_id].points_c.empty()) {
        std::cerr
            << "No local cloud for ICP."
            << std::endl;
        return;
    }

  
    // source cloud:current frame cloud
    std::vector<Eigen::Vector3d>& src_pts_c =this->_view_clouds[img_id].points_c;

   
    // target cloud: global world cloud
    std::vector<Eigen::Vector3d>& tgt_pts_w = this->_points_cloud;

    if (tgt_pts_w.size() < 50) {
        std::cerr
            << "Global cloud too small."
            << std::endl;
        return;
    }

    // current pose: Tcw
    // convert to Twc (camera-->world)
    Eigen::Matrix3d Rcw =this->_cameras[img_id]._R;
    Eigen::Vector3d tcw =this->_cameras[img_id]._t;
    Eigen::Matrix3d Rwc =Rcw.transpose();
    Eigen::Vector3d twc = -Rcw.transpose() * tcw;

    // initial transform
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    T_wc.block<3,3>(0,0) = Rwc;
    T_wc.block<3,1>(0,3) = twc;


    // ICP iteration
    const int max_iterations = 20;
    const double max_correspond_distance = 0.05;

    for (int iter = 0;iter < max_iterations;iter++)
    {
        // transformed source points
        std::vector<Eigen::Vector3d>transformed_src;
        transformed_src.reserve(src_pts_c.size());

        for (const auto& p_c : src_pts_c)
        {
            Eigen::Vector4d p_h;
            p_h <<
                p_c(0),
                p_c(1),
                p_c(2),
                1.0;

            Eigen::Vector4d p_w_h =T_wc * p_h;
            transformed_src.push_back(p_w_h.head<3>());
        }
        // nearest neighbor correspondences:brute force
        std::vector<Eigen::Vector3d> pts_src;
        std::vector<Eigen::Vector3d> pts_tgt;

        for (size_t i = 0; i < transformed_src.size();i++)
        {
            const auto& p =transformed_src[i];

            double best_dist =1e9;
            int best_idx = -1;

            for (size_t j = 0;j < tgt_pts_w.size();j++)
            {
                double dist =(p - tgt_pts_w[j]).norm();

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            if (best_idx >= 0 && best_dist < max_correspond_distance)
            {
                const auto& src_p = transformed_src[i];
                const auto& tgt_p = tgt_pts_w[best_idx];

                pts_src.push_back(Eigen::Vector3d(src_p(0),src_p(1),src_p(2)));
                pts_tgt.push_back(Eigen::Vector3d( tgt_p(0),tgt_p(1), tgt_p(2)));
            }
        }

        std::cout
            << "[ICP] iteration "
            << iter
            << " correspondences = "
            << pts_src.size()
            << std::endl;

      
        // insufficient correspondences

        if (pts_src.size() < 10) {
            break;
        }

     
        // solve delta transform  SVD method
        Eigen::Matrix3d dR;
        Eigen::Vector3d dt;
        pose_estimation_3d3d(pts_src,pts_tgt,dR,dt);

        // delta transform
        Eigen::Matrix4d dT = Eigen::Matrix4d::Identity();

        dT.block<3,3>(0,0) = dR;
        dT.block<3,1>(0,3) = dt;

        // update
        T_wc = dT * T_wc;


        // convergence check
        double trans_norm =dt.norm();
        double rot_diff = (dR - Eigen::Matrix3d::Identity()).norm();

        std::cout
            << "[ICP] trans = "
            << trans_norm
            << " rot = "
            << rot_diff
            << std::endl;

        if (trans_norm < 1e-4 &&
            rot_diff < 1e-4)
        {
            std::cout
                << "[ICP] converged."
                << std::endl;

            break;
        }
    }

    
    // convert back to Tcw
    // Pc = Rcw * Pw + tcw
    Eigen::Matrix3d Rwc_final = T_wc.block<3,3>(0,0);
    Eigen::Vector3d twc_final = T_wc.block<3,1>(0,3);

    Eigen::Matrix3d Rcw_final =Rwc_final.transpose();
    Eigen::Vector3d tcw_final =-Rwc_final.transpose() * twc_final;

  
    // update camera pose
    this->_cameras[img_id]._R =Rcw_final;
    this->_cameras[img_id]._t =tcw_final;

    
    // update world cloud
    this->_view_clouds[img_id].points_w.clear();
    for (const auto& p_c : src_pts_c)
    {
        Eigen::Vector3d p_w =Rwc_final * p_c +twc_final;
        this->_view_clouds[img_id].points_w.push_back(p_w);
    }

    std::cout
        << "ICP refinement finished for image "
        << img_id
        << std::endl;
}


void Rebuild::save_point_cloud(const std::string &filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 1. 寫入標題列 (CSV 標準格式)
    outfile << "x,y,z" << std::endl;

    int saved_count = 0;
    for (const auto &point : this->_points_cloud) {
        // 2. 數據有效性檢查：防止輸出 nan 或 inf 導致 CSV 損壞
        if (std::isfinite(point(0)) && std::isfinite(point(1)) && std::isfinite(point(2))) {
            
            // 3. 使用逗號分隔
            outfile << point(0) << "," 
                    << point(1) << "," 
                    << point(2) << std::endl;
            
            saved_count++;
        }
    }

    outfile.close();
    std::cout << "Point cloud saved as CSV to " << filename 
              << " (" << saved_count << " points)" << std::endl;
}







