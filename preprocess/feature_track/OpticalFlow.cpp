
template <typename T>
void TrackKLT<T>::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  // mask_klt stores Output status vector indicating whether the flow for the corresponding feature point was found.
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);


  if (this->use_flow_back_) {
    // rapid::common::TicToc timer_flow_back;
    std::vector<uchar> mask_klt_reverse;
    std::vector<cv::Point2f> reverse_pts = pts0;
    cv::calcOpticalFlowPyrLK(img1pyr, img0pyr, pts1, reverse_pts, mask_klt_reverse, error, win_size, 1, 
                            term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    for(size_t i = 0; i < mask_klt.size(); i++) {
      if(mask_klt[i] && mask_klt_reverse[i] && distance(pts0[i], reverse_pts[i]) <= 0.5) {
        mask_klt[i] = 1;
      } else
        mask_klt[i] = 0;
    }
    // LOG(INFO) << "Flow back time: " << timer_flow_back.toc() << " ms";
  }


  std::vector<uchar> mask_rsc;
  // rapid::common::TicToc timer_rsc;
  if (this->use_ransac_) {
    if (pts0.size() < 10) {
      for (size_t i = 0; i < pts0.size(); i++)
        mask_rsc.push_back((uchar)1); // we simply consider all points as inliers
    } else { // RANSAC requires at least 10 points
      // Normalize these points, so we can then do ransac
      // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
      std::vector<cv::Point2f> pts0_n, pts1_n;
      for (size_t i = 0; i < pts0.size(); i++) {
        pts0_n.push_back(this->camera_calib.at(id0)->undistort_cv_norm(pts0.at(i)));
        pts1_n.push_back(this->camera_calib.at(id1)->undistort_cv_norm(pts1.at(i)));
      }

  
      T max_focallength_img0 = std::max(this->camera_calib.at(id0)->get_K()(0, 0), this->camera_calib.at(id0)->get_K()(1, 1));
      T max_focallength_img1 = std::max(this->camera_calib.at(id1)->get_K()(0, 0), this->camera_calib.at(id1)->get_K()(1, 1));
      T max_focallength = std::max(max_focallength_img0, max_focallength_img1);
      
      // Images captured with different focal lengths will have different scales.
      // Dividing the threshold by max_focallength normalizes the threshold, making it independent of the image scale.
      // This will fail in degenerated scenarios (planar scenes)
      cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc); // mask_rsc stores inlier or not
    }
  }
}