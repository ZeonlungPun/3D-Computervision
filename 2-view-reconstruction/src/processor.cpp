#include "processor.h"

std::vector<Eigen::Vector3d>cart2hom(std::vector<cv::Point2f>pt_group)
{
    //Convert catesian to homogenous points by appending a row of 1s
    std::vector<Eigen::Vector3d>hompt_group;
    for (const cv::Point2f &pt: pt_group)
    {
        float x=pt.x;
        float y=pt.y;
        Eigen::Vector3d hom_pt;
        hom_pt(0)=x;
        hom_pt(1)=y;
        hom_pt(2)=1;
        hompt_group.push_back(hom_pt);
    }


    return hompt_group;

}

// 將旋轉向量轉換為 3x3 反對稱矩陣
Eigen::Matrix3d Skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W <<     0, -w(2),  w(1),
          w(2),     0, -w(0),
         -w(1),  w(0),     0;
    return W;
}


Eigen::Vector2d ReconstructionError(Eigen::Vector4d P_3d,Eigen::Matrix<double, 3, 4> project_matrix,Eigen::Vector2d p_2d_obs)
{
    /*
    calculate the Reconstruction Error; input : homogeneous vector in 3d and 2d space, projection matrix
    */
    //get 2d space homogeous points
    Eigen::Vector3d p_2d_h = project_matrix* P_3d;

   
    // 歸一化到影像平面
    Eigen::Vector2d p_2d_hat(p_2d_h[0]/p_2d_h[2], p_2d_h[1]/p_2d_h[2]);

    Eigen::Vector2d error_vec = p_2d_obs-p_2d_hat;
    

    return error_vec;

}

Eigen::Matrix<double, 2, 3> JacobianWrtPoint(
    const Eigen::Vector4d& P_3d,
    const Eigen::Matrix<double, 3, 4>& project_matrix)
{
    Eigen::Vector3d p = project_matrix * P_3d;
    double px = p(0), py = p(1), pz = p(2);

    if (std::abs(pz) < 1e-12)
        return Eigen::Matrix<double, 2, 3>::Zero();

    Eigen::Matrix<double, 2, 3> J;

    // 第一行: e_x 對 X,Y,Z
    J(0,0) = - (project_matrix(0,0)*pz - project_matrix(2,0)*px) / (pz*pz);
    J(0,1) = - (project_matrix(0,1)*pz - project_matrix(2,1)*px) / (pz*pz);
    J(0,2) = - (project_matrix(0,2)*pz - project_matrix(2,2)*px) / (pz*pz);

    // 第二行: e_y 對 X,Y,Z
    J(1,0) = - (project_matrix(1,0)*pz - project_matrix(2,0)*py) / (pz*pz);
    J(1,1) = - (project_matrix(1,1)*pz - project_matrix(2,1)*py) / (pz*pz);
    J(1,2) = - (project_matrix(1,2)*pz - project_matrix(2,2)*py) / (pz*pz);

    return J;
}

Eigen::Matrix<double, 2, 3> JacobianWrtTranslation(Eigen::Matrix<double, 3, 4> project_matrix,const Eigen::Vector4d& P_3d)
{
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Vector3d p = project_matrix * P_3d;
    double px = p(0), py = p(1), pz = p(2);
    J(0,0) = - 1.0/ pz;
    J(0,1) = 0.0;
    J(0,2) = px / (pz*pz);
    J(1,0) =0.0;
    J(1,1)= - 1.0/ pz;
    J(1,2)=py/(pz*pz);
    return J;

}

Eigen::Matrix<double, 2, 3> JacobianWrtRotation
(Eigen::Matrix<double, 2, 3>& J_pt ,Eigen::Matrix<double, 3, 3>& Rotation_matrix,
    const Eigen::Vector3d& P_3d  )
{
    Eigen::Vector3d temp_vec = Rotation_matrix*P_3d;
    Eigen::Matrix3d skew_m=Skew(temp_vec);
    Eigen::Matrix<double, 2, 3> J=J_pt*(-skew_m);

    return J;


}

Eigen::VectorXd BA_CostFunction(
    const std::vector<Eigen::Vector4d>& tripoints3d,
    const std::vector<Eigen::Matrix<double, 3, 4>>& project_matrix_list,
    const std::vector<Eigen::Vector2d>& p_2d_obs_list,
    double lambda)
{
    int num_points = int(tripoints3d.size());
    int num_cameras = int(project_matrix_list.size());

    // 參數維度：3*num_points + 6*(num_cameras-1)  （第一相機固定）
    int param_dim = 3 * num_points + 6 * (num_cameras - 1);
    int num_observations = num_points * num_cameras; // 假設每個點在每個相機都有觀測
    int resid_dim = 2 * num_observations;

    Eigen::MatrixXd J_system = Eigen::MatrixXd::Zero(resid_dim, param_dim);
    Eigen::VectorXd error_vec = Eigen::VectorXd::Zero(resid_dim);

    for (int i = 0; i < num_points; ++i) {
        const Eigen::Vector4d& P_3d = tripoints3d[i];
        Eigen::Vector3d P_nonhomo_3d = P_3d.hnormalized();

        for (int j = 0; j < num_cameras; ++j) {
            const Eigen::Matrix<double, 3, 4>& P = project_matrix_list[j];

            Eigen::Matrix<double,3,3> Rotate_matrix = P.block<3, 3>(0, 0);

            int row = 2 * (i * num_cameras + j);   // 2 行對應 (i,j)
            int col_point = 3 * i;                 // 3 列對應第 i 個 3D 點
            // col_pose 要以 (j-1) 計算：因為第一個相機 (j==0) 不在參數中
            int col_pose = 3 * num_points + 6 * (j - 1); 

            // 先計算殘差
            Eigen::Vector2d p_2d_obs = p_2d_obs_list[j * num_points + i];
            Eigen::Vector2d resid = ReconstructionError(P_3d, P, p_2d_obs);
            error_vec(row)     = resid(0);
            error_vec(row + 1) = resid(1);

            // 對 3D 點的 Jacobian（2x3）
            Eigen::Matrix<double, 2, 3> J_pt = JacobianWrtPoint(P_3d, P);
            J_system.block<2, 3>(row, col_point) = J_pt;

            // 對相機 pose 的 Jacobian（2x6），但若 j==0 則跳過（第一相機固定）
            if (j > 0) {
                Eigen::Matrix<double, 2, 3> J_T = JacobianWrtTranslation(P, P_3d);
                Eigen::Matrix<double, 2, 3> J_R = JacobianWrtRotation(J_pt, Rotate_matrix, P_nonhomo_3d);
                Eigen::Matrix<double, 2, 6> J_pose;
                J_pose.block<2, 3>(0, 0) = J_R;
                J_pose.block<2, 3>(0, 3) = J_T;
                J_system.block<2, 6>(row, col_pose) = J_pose;
            }
        }
    }

    // normal equations: (J^T J + lambda I) delta = J^T r
    Eigen::MatrixXd H = J_system.transpose() * J_system;
    // damping only on diagonal (typical LM)
    H.diagonal().array() += lambda;

    Eigen::VectorXd g = J_system.transpose() * error_vec;

    // solve H delta = -g  (note sign convention; earlier you used +, here we solve for delta = -H^{-1} g)
    // We can do delta = - H.ldlt().solve(g)
    Eigen::VectorXd delta;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
    if (ldlt.info() == Eigen::Success) {
        delta = - ldlt.solve(g);
    } else {
        // fallback —如果 LDLT 失敗，用 pseudo-inverse 的 solve（不建議但保底）
        delta = - H.completeOrthogonalDecomposition().solve(g);
    }

    return delta;
}

void LM_update(std::vector<Eigen::Vector4d>& tripoints3d,
               std::vector<Eigen::Matrix<double, 3, 4>>& project_matrix_list,
               const std::vector<Eigen::Vector2d>& p_2d_obs_list,
               int iter_num)
{
    double lambda = 1e-2; // 起始阻尼，可視情況調
    int num_points = int(tripoints3d.size());
    int num_cameras = int(project_matrix_list.size());

    for (int it = 0; it < iter_num; ++it) {
        Eigen::VectorXd delta = BA_CostFunction(tripoints3d, project_matrix_list, p_2d_obs_list, lambda);

        // 拆分 delta
        int param_point_dim = 3 * num_points;
        int param_pose_dim  = 6 * (num_cameras - 1); // exclude first camera

        if ((int)delta.size() != param_point_dim + param_pose_dim) {
            std::cerr << "delta size mismatch: " << delta.size() << " vs expected "
                      << param_point_dim + param_pose_dim << std::endl;
            return;
        }

        double old_cost = TotalReconstructionError(tripoints3d, project_matrix_list, const_cast<std::vector<Eigen::Vector2d>&>(p_2d_obs_list));

        // 保存 old state in case we need to rollback
        auto old_points = tripoints3d;
        auto old_proj   = project_matrix_list;

        // 更新 3D 點
        for (int i = 0; i < num_points; ++i) {
            Eigen::Matrix<double, 3, 1> dX = delta.segment<3>(3 * i);
            tripoints3d[i].head<3>() += dX;
        }

        // 更新相機 pose (從 j=1 開始)
        for (int j = 1; j < num_cameras; ++j) {
            int offset = param_point_dim + 6 * (j - 1);
            Eigen::Matrix<double, 6, 1> dPose = delta.segment<6>(offset);
            UpdateCameraPose(project_matrix_list[j], dPose);
        }

        double new_cost = TotalReconstructionError(tripoints3d, project_matrix_list, const_cast<std::vector<Eigen::Vector2d>&>(p_2d_obs_list));

        std::cout << "iter " << it << " cost: " << old_cost << " -> " << new_cost << std::endl;

        if (new_cost < old_cost) {
            // 接受更新，減小 lambda
            lambda *= 0.7;
        } else {
            // 回滾並增加 lambda
            tripoints3d = old_points;
            project_matrix_list = old_proj;
            lambda *= 2.0;
        }
    }
}

double TotalReconstructionError(std::vector<Eigen::Vector4d>& tripoints3d,
    std::vector<Eigen::Matrix<double, 3, 4>>& project_matrix_list,
    std::vector<Eigen::Vector2d>& p_2d_obs_list)
{
    double total_error=0;
    int num_points = tripoints3d.size();
    int num_cameras = project_matrix_list.size();

    for (int i=0; i<num_points; i++) {
        Eigen::Vector4d P_3d = tripoints3d[i];

        for (int j=0; j<num_cameras; j++) {
            Eigen::Matrix<double, 3, 4> project_matrix = project_matrix_list[j];
            Eigen::Vector2d p_2d_obs = p_2d_obs_list[j*num_points + i];
            Eigen::Vector2d error_vec= ReconstructionError(P_3d, project_matrix, p_2d_obs);
            total_error += error_vec.squaredNorm();  // 建議用平方和
        }
    }
    return total_error;
}





// 更新相機投影矩陣 [R|t]
void UpdateCameraPose(Eigen::Matrix<double,3,4>& project_matrix, const Eigen::Matrix<double,6,1>& dPose) 
{
    Eigen::Vector3d delta_omega = dPose.segment<3>( 0);
    Eigen::Vector3d delta_t     = dPose.segment<3>( 3);

    Eigen::Matrix3d R = project_matrix.block<3,3>(0,0);
    Eigen::Vector3d t =  project_matrix.block<3,1>(0,3);

    Eigen::Matrix3d dR =
        Eigen::Matrix3d::Identity() + Skew(delta_omega);

    R = dR * R;
    t = t + delta_t;

     //  保證 R ∈ SO(3)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();


    project_matrix.block<3,3>(0,0) = R;
    project_matrix.block<3,1>(0,3) = t;

    
}