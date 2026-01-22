#include "extract_features.h"




std::vector<std::vector<cv::Point2f>> find_correspondence_points(cv::Mat image1,cv::Mat image2, bool visualize = true)
{
  
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // RGB image to GRAY image
    cv::Mat gray_image1, gray_image2;
    cv::cvtColor(image1, gray_image1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray_image2, cv::COLOR_BGR2GRAY);

    //find the keypoints and descriptors with SIFT
    std::vector<cv::KeyPoint> keypoints1,keypoints2;
    cv::Mat descriptors1,descriptors2;

    detector->detectAndCompute(gray_image1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(gray_image2, cv::noArray(), keypoints2, descriptors2);

    std::cout << "Keypoints1:" << keypoints1.size() << std::endl;
    std::cout << "Keypoints2:" << keypoints2.size() << std::endl;
    
    //Find matching points
    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::KDTreeIndexParams(5); // 5 trees
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams(50);
    cv::FlannBasedMatcher matcher(indexParams, searchParams);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    int k = 2; // Number of nearest neighbors to find for each query descriptor
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, k);
    //Apply Lowe's SIFT matching ratio test
    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.8f; // Common threshold for Lowe's ratio test

    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) 
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }

    std::vector<Point2f> src_pts;
    std::vector<Point2f> dst_pts;

    for(int j=0; j<good_matches.size(); j++)
    {
         //-- Get the keypoints from the good matches
        src_pts.push_back( keypoints1[ good_matches[j].queryIdx ].pt );
        dst_pts.push_back( keypoints2[ good_matches[j].trainIdx ].pt );
    }
    //Constrain matches to fit homography
    // Find homography
    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 100.0, mask);

    // Select only inlier points
    std::vector<cv::Point2f> pts1, pts2;
    for (int ii = 0; ii < mask.rows; ++ii) {
        if (mask.at<uchar>(ii)) {
            pts1.push_back(src_pts[ii]);
            pts2.push_back(dst_pts[ii]);
        }
    }
    std::vector<std::vector<cv::Point2f>> final_points_group;
    final_points_group.push_back(pts1);
    final_points_group.push_back(pts2);

     // Visualization
    if (visualize) {
        // Draw all matches
        Mat img_matches;
        drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches,
                   Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //imshow("All Good Matches", img_matches);
        
        // Draw only inlier matches
        vector<DMatch> inlier_matches;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<uchar>(i)) {
                inlier_matches.push_back(good_matches[i]);
            }
        }
        
        Mat img_inlier_matches;
        drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, img_inlier_matches,
                   Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //imshow("Inlier Matches", img_inlier_matches);
        
        // Draw matching points on original images
        Mat img1_pts = image1.clone();
        Mat img2_pts = image2.clone();
        
        for (size_t i = 0; i < pts1.size(); i++) {
            circle(img1_pts, pts1[i], 5, Scalar(0, 255, 0), 2);
        }
        for (size_t i = 0; i < pts2.size(); i++) {
            circle(img2_pts, pts2[i], 5, Scalar(0, 255, 0), 2);
        }
        
        //imshow("Image 1 Points", img1_pts);
        //imshow("Image 2 Points", img2_pts);
        cv::imwrite("Inlier_Matches.png", img_inlier_matches);
        cv::imwrite("All_Good_Matches.png", img_matches);
        cv::imwrite("Image1_Points.png",img1_pts);
        cv::imwrite("Image2_Points.png",img2_pts);
        
       
    }

    return final_points_group;


}

