#include "structure.h"

Eigen::MatrixXd corrosponding_matrix(std::vector<Eigen::Vector3d>npt_g1,std::vector<Eigen::Vector3d>npt_g2)
{
    /*Each row in the  matrix below is constructed as
    p1 =(x',y',1)
    p2 = (x,y,1)
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]*/
    
    assert (npt_g1.size()==npt_g2.size());
    int num_points=npt_g1.size();

    // The correspondence matrix W will have 9 columns and 'num_points' rows
    Eigen::MatrixXd W(num_points, 9);

    for (int i=0;i<num_points;i++)
    {
        double p1x = npt_g1[i][0];
        double p1y = npt_g1[i][1];
        double p2x = npt_g2[i][0];
        double p2y = npt_g2[i][1];

        W(i,0)=p1x*p2x;
        W(i,1)=p1x*p2y;
        W(i,2)=p1x;
        W(i,3)=p1y*p2x;
        W(i,4)=p1y*p2y;
        W(i,5)=p1y;
        W(i,6)=p2x;
        W(i,7)=p2y;
        W(i,8)=1;

    }

    return W;

}


Eigen::Matrix3d compute_fundamental_matrix(std::vector<Eigen::Vector3d>npt_g1,std::vector<Eigen::Vector3d>npt_g2)
{
    
    /*Computes the  fundamental matrix from corresponding points
      using the normalized 8 point algorithm.

    input npt_g1, npt_g2: corresponding points vector with shape n x3
    returns:  fundamental matrix with shape 3 x 3 */
    
    Eigen::MatrixXd W = corrosponding_matrix(npt_g1,npt_g2);
    std::cout<<"w:"<<W<<std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV(); 
    Eigen::VectorXd S = svd.singularValues();
    
    
    Eigen::VectorXd F_hat_vector = V.col(V.cols()-1).transpose();;
    std::cout<<"w:"<<F_hat_vector<<std::endl;
    Eigen::Matrix3d F_hat_matrix;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            F_hat_matrix(i, j) = F_hat_vector(i * 3 + j); // row-major 對齊 Python reshape
        }
    }
    std::cout<<"w:"<<F_hat_matrix<<std::endl;
    //constrain F. Make rank 2 by zeroing out last singular value
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_2(F_hat_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U_2 = svd_2.matrixU();
    Eigen::MatrixXd V_2 = svd_2.matrixV(); 
    Eigen::VectorXd S_2 = svd_2.singularValues();
    Eigen::MatrixXd V_2_t =V_2.transpose();

    std::cout<<"U:"<<U_2<<std::endl;
    std::cout<<"V:"<<V_2_t<<std::endl;
    std::cout<<"U:"<<S_2<<std::endl;

    S_2(S_2.size() - 1) = 0.0;
    S_2(0)=1.0;
    S_2(1)=1.0;
    Eigen::DiagonalMatrix<double, 3> S_diag(S_2);
    Eigen::Matrix3d F_hat_matrix_final= U_2* (S_diag *V_2_t);
    std::cout<<"w:"<<F_hat_matrix_final<<std::endl;
    return F_hat_matrix_final;
       
}

Eigen::Matrix3d scale_and_translate_points(std::vector<Eigen::Vector3d>&normalize_pts)
{
    /*Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    param normalize_pts: array of homogenous point ( n x3)
    returns: array of same input shape and its normalization matrix  */

    // find the centroid 
    double center_x = 0.0, center_y = 0.0;
    for (const Eigen::Vector3d &pts:normalize_pts)
    {
        double x=pts[0];
        double y=pts[1];

        center_x+=x;
        center_y+=y;

    }
    center_x= center_x/ static_cast<double> (normalize_pts.size());
    center_y= center_y/ static_cast<double> (normalize_pts.size());

    // centralize and calculate the distance
    double dist=0;
    for (const Eigen::Vector3d &pts_:normalize_pts)
    {
        double x_=pts_[0];
        double y_=pts_[1];

        x_= x_ - center_x;
        y_= y_ - center_y;
        dist += std::sqrt(x_ * x_ + y_ * y_);

    }
    double dist_mean= dist/ static_cast<double> (normalize_pts.size()); 
    double scale= M_SQRT2/ dist_mean; 

    // Construct the normalization matrix
    Eigen::Matrix3d norm3d_T;
    norm3d_T<< scale, 0, -scale * center_x,
    0, scale, -scale * center_y,
    0,0,1;

    for (int i=0;i< normalize_pts.size();i++)
    {
        Eigen::Vector3d pt= normalize_pts[i];
        normalize_pts[i]= norm3d_T*pt;
    }

    return norm3d_T;



}

std::vector<std::pair<Eigen::Matrix3d,Eigen::Vector3d>> compute_P_from_Essential(Eigen::Matrix3d E)
{
     /*Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    returns: list of 4 possible camera matrices. */
    std::vector<std::pair<Eigen::Matrix3d,Eigen::Vector3d>>result_vector;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV(); 
    Eigen::VectorXd S = svd.singularValues();


    

    //create 4 possible camera matrices (Hartley p 258)
    Eigen::Matrix3d W;
    W << 0, -1, 0,
    1, 0, 0,
    0, 0, 1;

    Eigen::Matrix3d R1= U*(W*V.transpose());
    //Ensure rotation matrix are right-handed with positive determinant
    // Ensure U and V are right-handed coordinate systems
    // This is crucial for a valid rotation matrix.
    double R1_SIGN= R1.determinant();
    Eigen::Vector3d T1=U.col(2);

    Eigen::Matrix3d R2= U*(W*V.transpose());
    double R2_SIGN= R2.determinant();
    Eigen::Vector3d T2= -U.col(2);


    Eigen::Matrix3d R3= U*(W.transpose()*V.transpose());
    double R3_SIGN= R3.determinant();
    Eigen::Vector3d T3= U.col(2);

    Eigen::Matrix3d R4= U*(W.transpose()*V.transpose());
    double R4_SIGN= R4.determinant();
    Eigen::Vector3d T4= -U.col(2);

    
    result_vector.push_back(std::make_pair(R1*R1_SIGN, T1));
    result_vector.push_back(std::make_pair(R2*R2_SIGN, T2));
    result_vector.push_back(std::make_pair(R3*R3_SIGN, T3));
    result_vector.push_back(std::make_pair(R4*R4_SIGN, T4));
    return result_vector;

}


Eigen::Vector4d reconstruct_one_3dpoint(Eigen::Vector3d pt1,Eigen::Vector3d pt2,Eigen::Matrix<double, 3, 4> m1,Eigen::Matrix<double, 3, 4> m2)
{
    /*Linear Triangulation: 
     pt1 and m1 * P are parallel and cross product = 0
        pt1 x m1 * P  =  pt2 x m2 * P  =  0  */
    Eigen::Matrix3d pt1_skew;
    pt1_skew << 0, -pt1(2), pt1(1),
              pt1(2), 0, -pt1(0),
              -pt1(1), pt1(0), 0;

    Eigen::Matrix3d pt2_skew;
    pt2_skew << 0, -pt2(2), pt2(1),
              pt2(2), 0, -pt2(0),
              -pt2(1), pt2(0), 0;

    Eigen::Matrix<double, 3, 4> pt1_m1 = pt1_skew * m1;
    Eigen::Matrix<double, 3, 4> pt2_m2 = pt2_skew * m2;

    Eigen::MatrixXd P(6, 4);

    P.block(0, 0, 3, 4) = pt1_m1;
    P.block(3, 0, 3, 4) = pt2_m2;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV(); 

    Eigen::Vector4d P_hat= V.col(V.cols() - 1);

    return P_hat / P_hat(3);
    
}

Eigen::Vector4d linear_triangulation(Eigen::Vector3d p1,Eigen::Vector3d p2,Eigen::Matrix<double, 3, 4> m1,Eigen::Matrix<double, 3, 4> m2)
{
    /*
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * P and p2 = m2 * P. Solve AP = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3, )
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4, homogenous 3d triangulated points
    */
    Eigen::Matrix4d A;
    A.row(0)=p1.row(0)*m1.row(2)-m1.row(0);
    A.row(1)=p1.row(1)*m1.row(2)-m1.row(1);
    A.row(2)=p2.row(0)*m2.row(2)-m2.row(0);
    A.row(3)=p2.row(1)*m2.row(2)-m2.row(1);


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV(); 

    Eigen::Vector4d X_hat= V.col(V.cols() - 1);

    return X_hat / X_hat(3);
    

}
