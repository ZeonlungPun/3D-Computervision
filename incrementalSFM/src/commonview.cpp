#include "commonview.h"


CommonView::CommonView(std::vector<std::string> images_dirs,std::string base_path) 
{
    this->_image_nums = images_dirs.size();
    this->_images.resize(this->_image_nums); 
    this->_graph.resize(this->_image_nums );
    //read all the images in order to construre scene graph
    for (int i = 0; i < images_dirs.size(); i++) {
        fs::path full_path = fs::path(base_path) / images_dirs[i];
        cv::Mat img = cv::imread(full_path.string());
        if (img.empty()) {
            std::cerr << "image reading error" << std::endl;
            exit(1);
        }

        this->_images[i] = img;
    }
    //assume all the images have the same size
    this->img_width = this->_images[0].cols;
    this->img_height = this->_images[0].rows;

    std::cout<<" all images read finished! image number is " <<this->_images.size() <<std::endl;
    this->create_graph();
    this->create_tracks();
}



void CommonView::create_graph() {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    std::vector<cv::Mat> &images = CommonView::_images;
    
    
    //  特徵提取 (Node 建立)
    for (int i = 0; i < this->_image_nums; i++) {
        Node &node = this->_graph[i];
        sift->detectAndCompute(this->_images[i], cv::noArray(), node.keyPoints, node.descriptors);
        node.track_id.assign(node.keyPoints.size(), -1);
        node.edges.resize(this->_image_nums); // 剛好對應所有影像索引
    }


    //建立邊 edge[i][j] = edge[j][i]
    const float ratio_thresh = 0.7f;
    for (int i = 0; i < this->_image_nums; i++) {
        for (int j = i + 1; j < this->_image_nums; j++) 
        { 
            std::vector<std::vector<cv::DMatch>> knn_matches;
            
            matcher->knnMatch(this->_graph[i].descriptors, this->_graph[j].descriptors, knn_matches, 2);
            
            //  Lowe's Ratio Test 初選
            std::vector<cv::DMatch> ratio_matches;
            std::vector<cv::Point2d> pts1, pts2;
            for (const auto &m : knn_matches) {
                if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
                    ratio_matches.push_back(m[0]);
                    pts1.push_back(this->_graph[i].keyPoints[m[0].queryIdx].pt);
                    pts2.push_back(this->_graph[j].keyPoints[m[0].trainIdx].pt);
                }
            }

            // RANSAC 幾何驗證去除噪音點 
            std::vector<cv::DMatch> final_matches;
            if (pts1.size() >= 8) { // 計算基礎矩陣至少需要 8 個點
                std::vector<uchar> mask;
                // 使用 RANSAC 尋找符合極線幾何 (Epipolar Geometry) 的點
                cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask);

                for (size_t k = 0; k < mask.size(); k++) {
                    if (mask[k]) { // 只有 mask 為 1 的才是 Inliers
                        final_matches.push_back(ratio_matches[k]);
                    }
                }
            }

            // 只有當匹配數量足夠時才建立邊
            if (final_matches.size() > 5) { 
                this->_graph[i].edges[j] = {true, final_matches};
            } else {
                this->_graph[i].edges[j] = {false, {}};
            }
        }
            
    }
    
}


void CommonView::track_insert(std::pair<int, int> predecessor, std::pair<int, int> successor) {
    int id1 = predecessor.first;
    int kp_id1 = predecessor.second;
    int id2 = successor.first;
    int kp_id2 = successor.second;

    // 代碼遍歷所有已存在的 _tracks，尋找這兩個點是否出現過
    int t1 = this->_graph[id1].track_id[kp_id1]; 
    int t2 = this->_graph[id2].track_id[kp_id2];
    // 後繼點已有歸屬，跳過
    if (t2 != -1) return; 
    // 延伸舊軌跡，前驅點在舊軌跡中，但後繼點不在。
    if (t1 != -1) {
        
        this->_tracks[t1].push_back(successor);
        this->_graph[id2].track_id[kp_id2] = t1;
    } else {
        // 新建軌跡
        std::list<std::pair<int, int>> new_list = {predecessor, successor};
        this->_tracks.push_back(new_list);
        int new_id = this->_tracks.size() - 1;
        this->_graph[id1].track_id[kp_id1] = new_id;
        this->_graph[id2].track_id[kp_id2] = new_id;
    }
}



void CommonView::create_tracks() {
    
    for (int i = 0; i < this->_image_nums; i++) {
        for (int j = i + 1; j <this->_image_nums; j++) {
            //this->_graph[i].track_id.resize(this->_graph[i].keyPoints.size());
            //this->_graph[j].track_id.resize(this->_graph[j].keyPoints.size());
            Edge &edge = this->_graph[i].edges[j];
            const std::vector<cv::DMatch> &matches = edge.matches;
            for (int k = 0; k < matches.size(); k++) {
                const cv::DMatch &match = matches[k];
                int queryIdx = match.queryIdx; // 第一幅圖的特徵點index
                int trainIdx = match.trainIdx; // 第二幅圖的特徵點index
                std::pair<int, int> predecessor = {i, queryIdx}; 
                std::pair<int, int> successor = {j, trainIdx}; 
                track_insert(predecessor, successor);
            }
        }
    }
}

