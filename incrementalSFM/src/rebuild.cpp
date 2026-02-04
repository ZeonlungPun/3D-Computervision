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



Rebuild::Rebuild(const CommonView &commonView, const std::vector<Camera> &cameras, int image_num)
{
    this->_commonView = commonView;
    this->_images_num = image_num;
    this->_cameras = cameras;

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
    // 查找匹配特徵點最多的2個影像作為初始影像對
    // find the image pair with the most matches
    int id1, id2;
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

    std::cout << "Initial image pair: " << id1 << " and " << id2 << " with " << success_matches.size() << " matches." << std::endl;

    //初始重建
    // Initial reconstruction   
    this->_cameras_state[id1] = true;
    this->_cameras_state[id2] = true;   
    this->init_reconstruct(id1, id2, success_matches);


    //add more views
    int reconstructed_counts= 1;
    while (edegeExists(this->_commonView))
    {

        reconstructed_counts += 1;
        std::cout << "Reconstruction iteration: " << reconstructed_counts << std::endl;
        //從已經重建的點中選擇track最多的視圖
        std::vector<int> views_num(this->_images_num, 0);
       
        for (int i = 0; i < this->_points_state.size(); i++) {
            //取出已重建的三維點所對應的track id
            int track_id = this->_points_state[i];
            this->_tracks_state[track_id] = true;
            for (auto e: this->_commonView._tracks[track_id]) {
                int img_id = e.first;
                views_num[img_id] += 1;
                
            }
        }

        int max_val = 0, max_img_id = 0;
        for (int i = 0; i < this->_images_num; i++) {
            if (views_num[i] > max_val && this->_cameras_state[i] == false) {
                max_img_id = i;
                max_val = views_num[i];
            }
        }
        
        std::cout << "Adding view " << max_img_id << " with " << max_val << " tracks." << std::endl;
        // get the external paremeters use ePnP
        
        // 3d-2d correspondences
        std::vector<cv::Point3d> p3ds;
        std::vector<cv::Point2d> p2ds;
        for (int i = 0; i < this->_points_state.size(); i++) 
        {
            int track_id = this->_points_state[i];
            for (auto e: this->_commonView._tracks[track_id]) 
            {
                int img_id = e.first;
                int kp_id = e.second;
                if (img_id == max_img_id) {
                    Eigen::Vector3d p3d = this->_points_cloud[i];
                    p3ds.push_back(cv::Point3d(p3d(0), p3d(1), p3d(2)));
                    double p2d_x = this->_commonView._graph[max_img_id].keyPoints[kp_id].pt.x;
                    double p2d_y = this->_commonView._graph[max_img_id].keyPoints[kp_id].pt.y;
                    p2ds.push_back(cv::Point2d(p2d_x, p2d_y));
                }
            }
        }
        
        //solve PnP
        Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
        cv::Mat rvec, tvec,inliers;
        cv::Mat K = (cv::Mat_<double>(3,3) << 
        K_eigen(0,0), K_eigen(0,1), K_eigen(0,2),
        K_eigen(1,0), K_eigen(1,1), K_eigen(1,2),
        K_eigen(2,0), K_eigen(2,1), K_eigen(2,2));
       

        cv::solvePnPRansac(
            p3ds,               // std::vector<cv::Point3d>
            p2ds,               // std::vector<cv::Point2d>
            K,                  // 相機內參
            cv::Mat(),          // 畸變係數 (如果已校正，傳空)
            rvec,               // 輸出：旋轉向量
            tvec,               // 輸出：平移向量
            false,              // useExtrinsicGuess
            100,                // iterationsCount: RANSAC 迭代次數
            8.0,                // reprojectionError: 重投影誤差閾值 (單位：像素)
            0.99,               // confidence: 置信度
            inliers,            // out: 內點索引
            cv::SOLVEPNP_EPNP   // 具體算法
        );

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        std::cout<< "R_cv type:"<<  R_cv.type() << std::endl;
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

        //use max_id camera and other usable cameras to triangulate new points
        for (int i=0;i< this->_images_num;i++)
        {
            id1 = i;
            id2 = max_img_id;
            if (id1 > id2) {
                int tmp = id1;
                id1 = id2;
                id2 = tmp;
            }
            if (this->_cameras_state[i] && this->_commonView._graph[id1].edges[id2].flag == true) {
                this->_commonView._graph[id1].edges[id2].flag = false;
                this->_commonView._graph[id2].edges[id1].flag = false;
                success_matches = this->get_success_matches(id1, id2);
                if (success_matches.size() > 100) {
                    this->reconstruct(max_img_id, i, success_matches);
                    //this->ceres_bundle_adjustment();
                    
                }
            }
        }
        
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

void Rebuild::init_reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches)
{
    /* Initial reconstruction using two views and their matching points
    param id1: first image id
    param id2: second image id
    param success_matches: matching points between two images
    */
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }
     //get the matching points
    
     //record  the  key points id
    std::map<int, int> state1, state2;
    
    Node node1 = this->_commonView._graph[id1];
    Node node2 = this->_commonView._graph[id2];
    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;
    for (int i = 0; i < success_matches.size(); i++) {
      
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx = success_matches[i].trainIdx;
        auto point1 = node1.keyPoints[queryIdx];
        auto point2 = node2.keyPoints[trainIdx];
        points1.push_back(point1.pt);
        points2.push_back(point2.pt);
        state1[i] = queryIdx;
        state2[i] = trainIdx;
    }
  
    // compute essential matrix
    Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
    cv::Mat K = (cv::Mat_<double>(3,3) << 
    K_eigen(0,0), K_eigen(0,1), K_eigen(0,2),
    K_eigen(1,0), K_eigen(1,1), K_eigen(1,2),
    K_eigen(2,0), K_eigen(2,1), K_eigen(2,2));
    cv::Mat E, mask, R12,t12;
    E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.99, 1.0, mask);
    int inliers = cv::recoverPose(E, points1, points2, K, R12,t12,mask);
   

    //update the camera pose
    this->_cameras[id1]._R = Eigen::Matrix3d::Identity();
    this->_cameras[id1]._t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    for (int r = 0; r < 3; r++) {
        t_eigen(r) = t12.at<double>(r, 0);
        for (int c = 0; c < 3; c++) {
            R_eigen(r, c) = R12.at<double>(r, c);
        }
    }
    this->_cameras[id2]._R = R_eigen;
    this->_cameras[id2]._t = t_eigen;   


    //set camerea projection matrix
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt;
    cv::hconcat(R12, t12, Rt); // [R|t] R(3x3) 和 t(3x1)
    cv::Mat P2 = K * Rt;
    //triangulate points
    cv::Mat points4D;// 輸出：3D 點座標 (4xN, 齊次座標)
    cv::triangulatePoints(P1, P2, points1, points2, points4D);
    CV_Assert(points4D.type() == CV_64F);
    std::cout << "points4D type after triangulate = "
          << points4D.type() << std::endl;
    
    //convert to euclidean coordinates
    
    for (int i = 0; i < points4D.cols; i++) 
    {
        // only keep the inliner
        if (mask.at<uchar>(i) == 0) continue;
        double w = points4D.at<double>(3, i);
        
        if (std::abs(w) < 1e-6) continue;
        Eigen::Vector3d point3d;
        point3d(0) = points4D.at<double>(0, i) / w;
        point3d(1) = points4D.at<double>(1, i) / w;
        point3d(2) = points4D.at<double>(2, i) / w;
        // --- 深度檢查 (Depth Check / Cheirality Check) ---
        // 計算在相機 1 座標系下的位置: P_c1 = R1 * P_w + t1
        Eigen::Vector3d p_c1 = this->_cameras[id1]._R * point3d + this->_cameras[id1]._t;
        // 計算在相機 2 座標系下的位置: P_c2 = R2 * P_w + t2
        Eigen::Vector3d p_c2 = this->_cameras[id2]._R * point3d + this->_cameras[id2]._t;

        // 點必須在兩個相機的前方 (Z > 0)
        if (p_c1.z() <= 0.0001 || p_c2.z() <= 0.0001) {
            continue; 
        }


        // update the point cloud and point state
        this->_points_cloud.push_back(point3d);
        int track_id = this->_commonView._graph[id1].track_id[state1[i]];
        this->_points_state.push_back(track_id);
        // update the track state 標記該 track 已經成功 3D 化
        this->_tracks_state[track_id] = true;   
        
    }

    //cancel the edge of the two views
    this->_commonView._graph[id1].edges[id2].flag = false;
    this->_commonView._graph[id2].edges[id1].flag = false;  

    std::cout << "Initial reconstruction completed with " << this->_points_cloud.size() << " 3D points." << std::endl;

}

void Rebuild::reconstruct(int id1, int id2, std::vector<cv::DMatch> success_matches)
{
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }
    //get the matching points
    Node node1 = this->_commonView._graph[id1];
    Node node2 = this->_commonView._graph[id2];
    std::vector<cv::Point2d> points1, points2;
     //record  the  key points id
    std::vector<int> state1, state2;
    for (int i = 0; i < success_matches.size(); i++) {
      
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx = success_matches[i].trainIdx;
        auto point1 = node1.keyPoints[queryIdx];
        auto point2 = node2.keyPoints[trainIdx];
        points1.push_back(point1.pt);
        points2.push_back(point2.pt);
        state1.push_back(queryIdx);
        state2.push_back(trainIdx);
    }
    
    Eigen::Matrix3d K_eigen = this->_cameras[0]._K;
    cv::Mat K = (cv::Mat_<double>(3,3) << 
    K_eigen(0,0), K_eigen(0,1), K_eigen(0,2),
    K_eigen(1,0), K_eigen(1,1), K_eigen(1,2),
    K_eigen(2,0), K_eigen(2,1), K_eigen(2,2));

    //set camerea projection matrix
    cv::Mat R1_cv, t1_cv, R2_cv, t2_cv;
    R1_cv = cv::Mat(3,3,CV_64F);
    t1_cv = cv::Mat(3,1,CV_64F);
    R2_cv = cv::Mat(3,3,CV_64F);
    t2_cv = cv::Mat(3,1,CV_64F);
    for (int r = 0; r < 3; r++) {
        t1_cv.at<double>(r, 0) = this->_cameras[id1]._t(r);
        t2_cv.at<double>(r, 0) = this->_cameras[id2]._t(r);
        for (int c = 0; c < 3; c++) {
            R1_cv.at<double>(r, c) = this->_cameras[id1]._R(r, c);
            R2_cv.at<double>(r, c) = this->_cameras[id2]._R(r, c);
        }                                               
    }
    cv::Mat Rt1, Rt2;
    cv::hconcat(R1_cv, t1_cv, Rt1); // [R|t] R(3x3) 和 t(3x1)
    cv::hconcat(R2_cv, t2_cv, Rt2); // [R|t] R(3x3) 和 t(3x1)
    cv::Mat P1 = K * Rt1;
    cv::Mat P2 = K * Rt2;
    //triangulate points
    cv::Mat points4D;// 輸出：3D 點座標 (4xN, 齊次座標) 

    cv::triangulatePoints(P1, P2, points1, points2, points4D);
    for (int i = 0; i < points4D.cols; i++) 
    {
        int track_id = this->_commonView._graph[id1].track_id[state1[i]];
        // 如果這個 track 已經重建過了，就跳過 ---
        if (this->_tracks_state[track_id]) continue;

        double w = points4D.at<double>(3, i);
        
        // 簡單的數值穩定性檢查，防止除以 0 或過小的 w
        if (std::abs(w) < 1e-6) continue;

        Eigen::Vector3d point3d;
        point3d(0) = points4D.at<double>(0, i) / w;
        point3d(1) = points4D.at<double>(1, i) / w;
        point3d(2) = points4D.at<double>(2, i) / w;
        // --- 深度檢查 (Depth Check / Cheirality Check) ---
        // 計算在相機 1 座標系下的位置: P_c1 = R1 * P_w + t1
        Eigen::Vector3d p_c1 = this->_cameras[id1]._R * point3d + this->_cameras[id1]._t;
        // 計算在相機 2 座標系下的位置: P_c2 = R2 * P_w + t2
        Eigen::Vector3d p_c2 = this->_cameras[id2]._R * point3d + this->_cameras[id2]._t;

        // 點必須在兩個相機的前方 (Z > 0)
        if (p_c1.z() <= 0.001 || p_c2.z() <= 0.001) {
            continue; 
        }


        // update the point cloud and point state
        this->_points_cloud.push_back(point3d);
        this->_points_state.push_back(track_id);
        // update the track state 標記該 track 已經成功 3D 化
        this->_tracks_state[track_id] = true;   
        
    }

}

double Rebuild::reprojection_error(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera)
{
    Eigen::Matrix3d R = camera._R;
    Eigen::Vector3d t = camera._t;
    Eigen::Matrix3d K = camera._K;  
    Eigen::Vector3d p_c = R * p3d + t;
    // --- 如果點在相機平面後方，返回一個極大值而不是讓它變成 nan ---
    if (p_c.z() < 0.001) {
        return 1000.0; // 給予一個很大的懲罰誤差
    }
    Eigen::Vector3d p_img = K * p_c;
    Eigen::Vector2d p_proj;
    p_proj(0) = p_img(0) / p_img(2);
    p_proj(1) = p_img(1) / p_img(2);
    double error_x = p_proj(0) - p2d(0);;
    double error_y = p_proj(1) - p2d(1);
    return std::sqrt(error_x * error_x + error_y * error_y);
}

struct ReprojectionError {
    ReprojectionError(double u, double v, double fx, double fy, double cx, double cy)
        : u_(u), v_(v), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const cam, const T* const point, T* residuals) const {
        // 1. World -> Camera (R*P + t)
        T p[3];
        ceres::AngleAxisRotatePoint(cam, point, p);
        p[0] += cam[3]; p[1] += cam[4]; p[2] += cam[5];

        // 2. Projection to normalized plane
        T xp = p[0] / (p[2] + T(1e-10));
        T yp = p[1] / (p[2] + T(1e-10));


        // 3. Normalized -> Pixel
        T predicted_u = T(fx_) * xp + T(cx_);
        T predicted_v = T(fy_) * yp + T(cy_);

        residuals[0] = predicted_u - T(u_);
        residuals[1] = predicted_v - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(double u, double v, const Camera& camera) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(u, v, 
                                  camera._K(0,0), camera._K(1,1), 
                                  camera._K(0,2), camera._K(1,2)));
    }

    double u_, v_, fx_, fy_, cx_, cy_;
};

void Rebuild::ceres_bundle_adjustment()
{
    //3d-2d correspondences
    for (int i=0; i< this->_points_cloud.size(); i++) 
    {   
        int p3d_id = i;
        int track_id = this->_points_state[i];
        for (auto e: this->_commonView._tracks[track_id]) 
        {
            int cam_id = e.first;
            int kp_id = e.second;
            float p2d_x = this->_commonView._graph[cam_id].keyPoints[kp_id].pt.x;
            float p2d_y = this->_commonView._graph[cam_id].keyPoints[kp_id].pt.y;
            observation obs;
            obs.cam_id = cam_id;
            obs.p3d_id = p3d_id;
            obs.p2d[0] = p2d_x;
            obs.p2d[1] = p2d_y;
            this->_observations.push_back(obs);
        }
    }

    // set the reconstruction error
    double err = 0;
    int valid_count = 0;
    for (int i=0;i< this->_observations.size(); i++) 
    {
        observation obs = this->_observations[i];
        Eigen::Vector3d p3d = this->_points_cloud[obs.p3d_id];
        Eigen::Vector3d p2d;
        p2d(0) = obs.p2d[0];
        p2d(1) = obs.p2d[1];
        double error = this->reprojection_error(p3d, p2d, this->_cameras[obs.cam_id]);
        if (!std::isnan(error) && !std::isinf(error)) {
            err += error;
            valid_count++;
        }
        
    }
    err /= valid_count;
    std::cout << "Before BA, reprojection error: " << err << std::endl;


    ceres::Problem problem;
    // convert the parameters to be optimized
    // camera parameters: 6 parameters for each camera (3 for rotation as angle-axis, 3 for translation)
    std::vector<double> cam_params(this->_cameras.size() * 6);

    for (int i = 0; i < this->_cameras.size(); ++i) {
        Camera& camera = this->_cameras[i];

        // R -> angle-axis
        Eigen::AngleAxisd aa(camera._R);
        cam_params[i * 6 + 0] = aa.axis()(0) * aa.angle();
        cam_params[i * 6 + 1] = aa.axis()(1) * aa.angle();
        cam_params[i * 6 + 2] = aa.axis()(2) * aa.angle();

        cam_params[i * 6 + 3] = camera._t(0);
        cam_params[i * 6  + 4] = camera._t(1);
        cam_params[i * 6 + 5] = camera._t(2);
       
    } 
    // 3d point parameters: 3 parameters for each 3d point (X, Y, Z)
    std::vector<double> point_params(3 *this->_points_cloud.size());
    for (int i = 0; i < this->_points_cloud.size(); ++i) {
        point_params[i * 3 + 0] = this->_points_cloud[i](0);
        point_params[i * 3 + 1] = this->_points_cloud[i](1);
        point_params[i * 3 + 2] = this->_points_cloud[i](2);
    }
    
    // Ceres problem setup and solving 
    
    for (const auto& obs : this->_observations) {
        // 1. 創建立 CostFunction
        // 傳入 2D 座標 (u, v) 和 當前相機的內參 (fx, fy, cx, cy)
        ceres::CostFunction* cost_function = ReprojectionError::Create(
            obs.p2d[0], 
            obs.p2d[1], 
            this->_cameras[obs.cam_id] // 傳入相機物件以取得內參
        );

        // 2. 選擇損失函數 (LossFunction)
        // HuberLoss 可以降低誤匹配 (Outliers) 對優化的影響
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        // 3. 取得參數塊的指標
        // 相機參數指標：跳過前面的相機，指向當前 cam_id 的位置
        double* camera_ptr = cam_params.data() + obs.cam_id * 6;
        
        // 3D 點參數指標：指向當前 p3d_id 的位置
        double* point_ptr = point_params.data() + obs.p3d_id * 3;

        // 4. 加入問題
        problem.AddResidualBlock(
            cost_function,
            loss_function,
            camera_ptr,   // 對應 operator() 中的 const T* const cam
            point_ptr     // 對應 operator() 中的 const T* const point
        );
    }
    // 固定第一張相機避免尺度漂移
    if (!this->_observations.empty()) {
        problem.SetParameterBlockConstant(cam_params.data() + this->_observations[0].cam_id * 6);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout<< "--------------------------------" << std::endl;
    std::cout << summary.BriefReport() << std::endl;

    //update the optimized parameters back to cameras and points
    for (int i = 0; i < this->_cameras.size(); i++) {
        Camera& camera = this->_cameras[i];

        Eigen::Vector3d aa(
            cam_params[i*6+0],
            cam_params[i*6+1],
            cam_params[i*6+2]
        );
        double angle = aa.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) {
            R = Eigen::AngleAxisd(angle, aa.normalized()).toRotationMatrix();
        }

        camera._R = R;
        camera._t << cam_params[i*6+3],
                cam_params[i*6+4],
                cam_params[i*6+5];

    }

    // points
    for (int i = 0; i < this->_points_cloud.size(); i++) {
        this->_points_cloud[i] <<
            point_params[3*i+0],
            point_params[3*i+1],
            point_params[3*i+2];
    }
    // compute the final reprojection error
    double after_err = 0;
    for (int i=0;i< this->_observations.size(); i++) 
    {
        observation obs = this->_observations[i];
        Eigen::Vector3d p3d = this->_points_cloud[obs.p3d_id];
        Eigen::Vector3d p2d;
        p2d(0) = obs.p2d[0];
        p2d(1) = obs.p2d[1];
        double error = this->reprojection_error(p3d, p2d, this->_cameras[obs.cam_id]);
        after_err += error;
    }
    after_err /= this->_observations.size();       
    std::cout << "After BA, reprojection error: " << after_err << std::endl;   
    this->_observations.clear(); 
        

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


cv::Mat eigen2cv(const Eigen::Matrix3d& mat) {
    cv::Mat cv_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cv_mat.at<double>(i, j) = mat(i, j);
    return cv_mat;
}

cv::Mat eigen2cv_vec(const Eigen::Vector3d& vec) {
    cv::Mat cv_vec(3, 1, CV_64F);
    for (int i = 0; i < 3; i++) cv_vec.at<double>(i, 0) = vec(i);
    return cv_vec;
}

cv::Mat eigen2cv_dist(const Eigen::VectorXd& vec) {
    // 假設是 5 個畸變參數 (k1, k2, p1, p2, k3)
    cv::Mat cv_dist(1, 5, CV_64F);
    for (int i = 0; i < vec.size() && i < 5; i++) cv_dist.at<double>(0, i) = vec(i);
    return cv_dist;
}

void Rebuild::compute_stereo_depth(
    const cv::Mat& img1, const cv::Mat& img2, 
    const Camera& cam1, const Camera& cam2, 
    cv::Mat& out_depth, cv::Mat& out_rectified_img1,
    Eigen::Matrix3d& out_new_K, Eigen::Matrix3d& out_rect_R) 
{
    // 1. 準備相機參數
    cv::Mat K1 = eigen2cv(cam1._K);
    cv::Mat D1 = eigen2cv_dist(cam1._dist);
    cv::Mat K2 = eigen2cv(cam2._K);
    cv::Mat D2 = eigen2cv_dist(cam2._dist);

    // 2. 計算相對位姿 (Cam1 -> Cam2)
    // P_c2 = R_rel * P_c1 + T_rel
    // R_rel = R2 * R1^T
    // T_rel = t2 - R_rel * t1
    Eigen::Matrix3d R_rel_eig = cam2._R * cam1._R.transpose();
    Eigen::Vector3d T_rel_eig = cam2._t - R_rel_eig * cam1._t;

    cv::Mat R_rel = eigen2cv(R_rel_eig);
    cv::Mat T_rel = eigen2cv_vec(T_rel_eig);

    // 3. 立體校正 (Stereo Rectify)
    // 這會計算讓兩張圖極線平行的變換矩陣 (R1, R2) 和新的投影矩陣 (P1, P2)
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K2, D2, img1.size(), R_rel, T_rel, 
                      R1, R2, P1, P2, Q, 
                      cv::CALIB_ZERO_DISPARITY, 0, img1.size()); // alpha=0 盡量保留有效像素

    // 4. 建立校正映射表 (Undistort & Rectify Map)
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_16SC2, map21, map22);

    // 5. 影像重映射 (Remap) - 這是 Open3D 之後要用的 RGB
    cv::Mat img1_rect, img2_rect;
    cv::remap(img1, img1_rect, map11, map12, cv::INTER_LINEAR);
    cv::remap(img2, img2_rect, map21, map22, cv::INTER_LINEAR);
    
    // 轉換為灰階進行匹配
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1_rect, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2_rect, img2_gray, cv::COLOR_BGR2GRAY);

    // 6. SGBM 視差計算
    // 參數需要根據圖像解析度調整，這裡給出一組通用參數
    int minDisparity = 0;
    int numDisparities = 16 * 8; // 必須是16的倍數
    int blockSize = 5;
    
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        minDisparity, numDisparities, blockSize,
        8 * 3 * blockSize * blockSize,   // P1
        32 * 3 * blockSize * blockSize,  // P2
        1, 63, 10, 100, 32, cv::StereoSGBM::MODE_SGBM_3WAY
    );

    cv::Mat disp;
    sgbm->compute(img1_gray, img2_gray, disp);

    // 7. 視差轉深度 (Disparity to Depth)
    // Z = f * B / d
    // Q矩陣包含了 f*B 的資訊，可以直接用 reprojectImageTo3D
    // 但為了更精確控制，我們通常手動計算或過濾
    cv::Mat points3D;
    cv::reprojectImageTo3D(disp, points3D, Q, true); // handleMissingValues=true

    // 提取 Z 通道 (深度)
    std::vector<cv::Mat> channels;
    cv::split(points3D, channels);
    cv::Mat depth_map = channels[2]; // Z channel

    // 過濾無效值 (SGBM 失敗處通常為極大值或負值)
    // 設為 0 或 NaN
    cv::patchNaNs(depth_map, 0.0);
    cv::threshold(depth_map, depth_map, 1000.0, 0.0, cv::THRESH_TOZERO_INV); // 過濾掉 > 1000m 的點
    cv::threshold(depth_map, depth_map, 0.0, 0.0, cv::THRESH_TOZERO);      // 過濾掉 < 0 的點

    // 輸出
    out_depth = depth_map;           // CV_32F, 單位通常是 meters (取決於 t 的單位)
    out_rectified_img1 = img1_rect;  // 校正後的 RGB

    // 輸出新的相機內參 (P1 的前 3x3)
    out_new_K << P1.at<double>(0,0), P1.at<double>(0,1), P1.at<double>(0,2),
                 P1.at<double>(1,0), P1.at<double>(1,1), P1.at<double>(1,2),
                 P1.at<double>(2,0), P1.at<double>(2,1), P1.at<double>(2,2);

    // 輸出校正旋轉矩陣 (用於修正 Open3D 的 Extrinsics)
    // R1 是將原始相機旋轉到校正相機的矩陣
    for(int r=0; r<3; r++)
        for(int c=0; c<3; c++)
            out_rect_R(r,c) = R1.at<double>(r,c);
}
void save_depth_visualization(const cv::Mat& depth_map, const std::string& filename) {
    if (depth_map.empty()) return;

    cv::Mat adjMap;
    // 1. 將深度歸一化到 0-255 (CV_8U)
    // 注意：這裡要排除掉深度為 0 (無效點) 的部分
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal, NULL, NULL, depth_map > 0);
    
    depth_map.convertTo(adjMap, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // 2. 加上偽彩色 (Jet 效果：近紅遠藍，或者 Magma 效果)
    cv::Mat colorMap;
    cv::applyColorMap(adjMap, colorMap, cv::COLORMAP_JET);

    // 3. 將無效點（深度為 0）標記為黑色
    colorMap.setTo(cv::Scalar(0, 0, 0), depth_map <= 0);

    cv::imwrite(filename, colorMap);
}

void Rebuild::generate_depth_maps(const std::vector<cv::Mat>& images) {
    std::cout << "Starting dense reconstruction..." << std::endl;

    for (int i = 0; i < this->_images_num; i++) {
        if (!this->_cameras_state[i]) continue; // Skip 未重建的相機

        // 1. 尋找最佳匹配視圖 (Simple heuristic: most shared points)
        int best_neighbor = -1;
        int max_shared_points = 0;
        const int MIN_MATCH_THRESHOLD = 30; // 密集重建通常需要至少一定數量的匹配點來保證校正品質

        for (int j = 0; j < this->_images_num; j++) {
            if (i == j || !this->_cameras_state[j]) continue;

            // 處理上三角存儲
            int u = std::min(i, j);
            int v = std::max(i, j);

            // 直接檢查 matches 的數量，無視 flag
            const auto& edge_matches = this->_commonView._graph[u].edges[v].matches;
            int current_matches_count = static_cast<int>(edge_matches.size());

            if (current_matches_count > max_shared_points && current_matches_count > MIN_MATCH_THRESHOLD) {
                max_shared_points = current_matches_count;
                best_neighbor = j;
            }
        }

        if (best_neighbor == -1 || max_shared_points < 20) {
            std::cout << "Skipping view " << i << ": not enough overlap." << std::endl;
            continue;
        }

        std::cout << "Processing View " << i << " (Neighbor: " << best_neighbor << ")..." << std::endl;

        // 2. 計算深度
        cv::Mat depth_map, rect_rgb;
        Eigen::Matrix3d new_K, rect_R;
        
        this->compute_stereo_depth(images[i], images[best_neighbor], 
                             _cameras[i], _cameras[best_neighbor], 
                             depth_map, rect_rgb, new_K, rect_R);
                    

        // 3. 保存數據供 Open3D 使用
        std::string folder_path = "./output";
        if (!fs::exists(folder_path)) {
            if (fs::create_directories(folder_path)) {
                std::cout << "Successfully created directory: " << folder_path << std::endl;
            } else {
                std::cerr << "Critical: Could not create directory!" << std::endl;
                return;
            }
        }
        // 格式: 0000.jpg (RGB), 0000.png (Depth 16bit), 0000.json (Pose)
        std::string prefix = "./output/view_" + std::to_string(i);

     

        // A. 保存校正後的 RGB
        bool success_rgb = cv::imwrite(prefix + ".jpg", rect_rgb);

        // B. 保存深度圖 (Open3D 推薦存為 16-bit PNG, 單位毫米)
        // 假設 depth_map 單位是米 (根據你的 t 單位)
        cv::Mat depth_mm;
        depth_map.convertTo(depth_mm, CV_16U, 1000.0); // m -> mm
        bool success_depth =  cv::imwrite(prefix + ".png", depth_mm);
        if (!success_rgb || !success_depth) {
            std::cerr << "Failed to save images to: " << prefix << std::endl;
        } else {
            std::cout << "Successfully saved: " << prefix << std::endl;
        }
        // 保存給人類看的可視化 JPG
        save_depth_visualization(depth_map, prefix + "_vis.jpg"); 

        // 保存校正後的 RGB 影像
        bool success_vis= cv::imwrite(prefix + "_rect.jpg", rect_rgb);
        if (!success_vis)
        {
            std::cerr << "Failed to save images to: " << prefix << std::endl;
        }
        else
        {
            std::cout << "Successfully saved: " << prefix << std::endl;
        }

        // C. 計算並保存 Open3D 需要的 Pose
        // 原始 Pose (World -> Camera): [R_old | t_old]
        // 校正旋轉 R_rect (Camera -> Rectified Camera)
        // 新 Pose (World -> Rectified Camera): R_new = R_rect * R_old
        // 新 Translation: t_new = R_rect * t_old
        
        Eigen::Matrix3d R_new = rect_R * _cameras[i]._R;
        Eigen::Vector3d t_new = rect_R * _cameras[i]._t;
        
        // Open3D 讀取的軌跡通常是 Camera-to-World (Extrinsics 的逆矩陣)
        // 或者你需要存儲 World-to-Camera 的 Extrinsics
        // 這裡將 Extrinsics 寫入簡單文本供讀取
        
        std::ofstream pose_file(prefix + ".txt");
        pose_file << "Intrinsic (K):" << std::endl << new_K << std::endl;
        pose_file << "Extrinsic (R):" << std::endl << R_new << std::endl;
        pose_file << "Extrinsic (t):" << std::endl << t_new.transpose() << std::endl;
        pose_file.close();
    }
}