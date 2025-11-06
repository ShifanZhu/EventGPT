template <typename T>
void TrackBase<T>::ProjectFromEventToRgb(const std::vector<cv::KeyPoint>& pts_event,
                                                        const cv::Mat& depth_img,
                                                        std::vector<cv::KeyPoint>& pts_rgb, const bool already_undistort_events) {
  pts_rgb.reserve(pts_event.size());

  // rgb -> event extrinsics
  const auto event_to_rgb = this->camera_calib.at(RGBDCam)->GetRGBtoEvent().inverse();

  // helpers
  auto inBounds = [&](float u, float v) {
    return u >= 0.f && v >= 0.f &&
           u < static_cast<float>(depth_img.cols) &&
           v < static_cast<float>(depth_img.rows);
  };
  auto depthBilinear = [&](float u, float v) -> T {
    // assumes depth_img already undistorted/rectified to RGB new_K space
    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= depth_img.cols || y1 >= depth_img.rows) return T(0);

    float dx = u - x0, dy = v - y0;
    float w00 = (1 - dx) * (1 - dy);
    float w10 = dx * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w11 = dx * dy;

    // support CV_16U (mm) or CV_32F (m)
    float d00 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y0, x0)) * 1e-3f
                                           : depth_img.at<float>(y0, x0);
    float d10 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y0, x1)) * 1e-3f
                                           : depth_img.at<float>(y0, x1);
    float d01 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y1, x0)) * 1e-3f
                                           : depth_img.at<float>(y1, x0);
    float d11 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y1, x1)) * 1e-3f
                                           : depth_img.at<float>(y1, x1);

    float d = w00*d00 + w10*d10 + w01*d01 + w11*d11;
    return static_cast<T>(d);
  };

  // stats (optional)
  size_t skipped_oob_depth = 0, skipped_invalid_depth = 0, skipped_behind_cam = 0, skipped_oob_event = 0;

  for (size_t i = 0; i < pts_event.size(); ++i) {
    const auto& pt = pts_event[i];

    // 1) Undistort RGB pixel to its new_K space (to match undistorted depth)
    const Vec2<T> pt_event(pt.pt.x, pt.pt.y);
    Vec2<T> uv_event_u;
    if (!already_undistort_events) {
      uv_event_u = this->camera_calib.at(EventCam)->undistort_pxl_frame_new_K(pt_event);
    } else {
      uv_event_u = pt_event;
    }
    // std::cout << "etorgb-uv_event_u: " << uv_event_u.transpose() << std::endl; 

    // 2) Depth lookup in same undistorted grid — bilinear on float coords
    const float u = static_cast<float>(uv_event_u[0]);
    const float v = static_cast<float>(uv_event_u[1]);
    if (!inBounds(u, v)) { ++skipped_oob_depth; continue; }

    T depth = (this->data_source_ == perception::DataSource::SIMULATION)
                  ? this->camera_calib.at(RGBDCam)->get_depth_mj(depth_img, static_cast<int>(std::round(u)), static_cast<int>(std::round(v)))
                  #ifdef DO_NOT_USE_INTERPOLATION
                  : this->camera_calib.at(RGBDCam)->get_depth(depth_img, static_cast<int>(u), static_cast<int>(v));
                  #else
                  : depthBilinear(u, v);
                  #endif
    if (depth <= T(0)) { ++skipped_invalid_depth; continue; }

    // 3) Back-project in event cam (using new_K)
    Vec3<T> X_event = this->camera_calib.at(EventCam)->pixel2camera_new_K(uv_event_u, depth);
    // std::cout << "etorgb-X_event: " << X_event.transpose() << std::endl; 

    // 4) Transform to RGB cam
    Vec3<T> X_rgb = this->camera_calib.at(RGBDCam)->camera2camera(X_event, event_to_rgb);
    if (X_rgb[2] <= T(0)) { ++skipped_behind_cam; continue; }
    // std::cout << "etorgb-X_rgb: " << X_rgb.transpose() << std::endl; 

    // 5) Project with RGB intrinsics (undistorted pixel)…
    Vec2<T> uv_rgb_u = this->camera_calib.at(RGBDCam)->camera2pixel(X_rgb);
    // std::cout << "etorgb-uv_rgb_u: " << uv_rgb_u.transpose() << std::endl; 

    //    …then APPLY EVENT distortion to get raw event pixel coords
    Vec2<T> uv_rgb_d = this->camera_calib.at(RGBDCam)->distort_pxl(uv_rgb_u); // <-- use at(1), not at(0)
    // std::cout << "etorgb-uv_rgb_d: " << uv_rgb_d.transpose() << std::endl; 

    cv::KeyPoint kp_rgb;
    kp_rgb.pt.x = static_cast<float>(uv_rgb_d[0]);
    kp_rgb.pt.y = static_cast<float>(uv_rgb_d[1]);

    // 6) Bounds check on the event image
    if (uv_rgb_d[0] < T(0) || uv_rgb_d[1] < T(0) ||
        uv_rgb_d[0] >= static_cast<T>(this->camera_calib.at(RGBDCam)->w()) ||
        uv_rgb_d[1] >= static_cast<T>(this->camera_calib.at(RGBDCam)->h())) {
      ++skipped_oob_event; continue;
    }

    // 7) Keep result
    pts_rgb.push_back(kp_rgb);
  }

  return;
}

template <typename T>
void TrackBase<T>::ProjectFromRgbToEvent(const std::vector<cv::KeyPoint>& pts_rgb,
                                                        const std::vector<size_t>& ids_left_old,
                                                        const cv::Mat& depth_img,
                                                        std::vector<cv::KeyPoint>& pts_event,
                                                        std::vector<size_t>& ids_in_event) {
  ids_in_event.clear();
  pts_event.reserve(pts_rgb.size());
  ids_in_event.reserve(pts_rgb.size());

  // rgb -> event extrinsics
  const auto rgb_to_event = this->camera_calib.at(RGBDCam)->GetRGBtoEvent();

  // helpers
  auto inBounds = [&](float u, float v) {
    return u >= 0.f && v >= 0.f &&
           u < static_cast<float>(depth_img.cols) &&
           v < static_cast<float>(depth_img.rows);
  };
  auto depthBilinear = [&](float u, float v) -> T {
    // assumes depth_img already undistorted/rectified to RGB new_K space
    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= depth_img.cols || y1 >= depth_img.rows) return T(0);

    float dx = u - x0, dy = v - y0;
    float w00 = (1 - dx) * (1 - dy);
    float w10 = dx * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w11 = dx * dy;

    // support CV_16U (mm) or CV_32F (m)
    float d00 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y0, x0)) * 1e-3f
                                           : depth_img.at<float>(y0, x0);
    float d10 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y0, x1)) * 1e-3f
                                           : depth_img.at<float>(y0, x1);
    float d01 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y1, x0)) * 1e-3f
                                           : depth_img.at<float>(y1, x0);
    float d11 = depth_img.type() == CV_16U ? static_cast<float>(depth_img.at<uint16_t>(y1, x1)) * 1e-3f
                                           : depth_img.at<float>(y1, x1);

    float d = w00*d00 + w10*d10 + w01*d01 + w11*d11;
    return static_cast<T>(d);
  };

  // stats (optional)
  size_t skipped_oob_depth = 0, skipped_invalid_depth = 0, skipped_behind_cam = 0, skipped_oob_event = 0;
  for (size_t i = 0; i < pts_rgb.size(); ++i) {
    const auto& pt = pts_rgb[i];
    const size_t id = ids_left_old[i];

    // 1) Undistort RGB pixel to its new_K space (to match undistorted depth)
    const Vec2<T> pt_rgb(pt.pt.x, pt.pt.y);
    const Vec2<T> uv_rgb_u = this->camera_calib.at(RGBDCam)->undistort_pxl_frame_new_K(pt_rgb);
    // std::cout << "rgb2e-uv_rgb_u: " << uv_rgb_u.transpose() << std::endl; // TODO: DEBUG

    // //TODO: sanity check, distort back and check if they are the same
    // Vec2<T> tmp_uv_rgb_d = this->camera_calib.at(RGBDCam)->distort_pxl_new_K(uv_rgb_u); // <-- use at(1), not at(0)
    // // Vec2<T> tmp_uv_rgb_d = this->camera_calib.at(RGBDCam)->distort_pxl(uv_rgb_u); // <-- use at(1), not at(0)
    // std::cout << "!!!tmp_etorgb-uv_rgb_d: " << tmp_uv_rgb_d.transpose() << std::endl; 


    // 2) Depth lookup in same undistorted grid — bilinear on float coords
    const size_t u = std::round(static_cast<float>(uv_rgb_u[0]));
    const size_t v = std::round(static_cast<float>(uv_rgb_u[1]));
    if (!inBounds(u, v)) { ++skipped_oob_depth; continue; }

    T depth = (this->data_source_ == perception::DataSource::SIMULATION)
                  ? this->camera_calib.at(RGBDCam)->get_depth_mj(depth_img, static_cast<int>(u), static_cast<int>(v))
                  #ifdef DO_NOT_USE_INTERPOLATION
                  : this->camera_calib.at(RGBDCam)->get_depth(depth_img, static_cast<int>(u), static_cast<int>(v));
                  #else
                  : depthBilinear(u, v);
                  #endif
    if (depth <= T(0)) { ++skipped_invalid_depth; continue; }

    // 3) Back-project in RGB cam (using new_K)
    Vec3<T> X_rgb = this->camera_calib.at(RGBDCam)->pixel2camera_new_K(uv_rgb_u, depth);
    // std::cout << "rgb2e-X_rgb: " << X_rgb.transpose() << std::endl; // TODO: DEBUG

    // 4) Transform to Event cam
    Vec3<T> X_event = this->camera_calib.at(RGBDCam)->camera2camera(X_rgb, rgb_to_event);
    if (X_event[2] <= T(0)) { ++skipped_behind_cam; continue; }
    // std::cout << "rgb2e-X_event: " << X_event.transpose() << std::endl; // TODO: DEBUG

    // 5) Project with EVENT intrinsics (undistorted pixel)…
    // Vec2<T> uv_event_u = this->camera_calib.at(EventCam)->camera2pixel(X_event);
    Vec2<T> uv_event_u = this->camera_calib.at(EventCam)->camera2pixel_new_K(X_event);
    // std::cout << "rgb2e-uv_event_u: " << uv_event_u.transpose() << std::endl; // TODO: DEBUG
    // 6) Bounds check on the event image
    // This check might remove some points that could be valid after distortion
    if (uv_event_u[0] < T(0) || uv_event_u[1] < T(0) ||
        uv_event_u[0] >= static_cast<T>(this->camera_calib.at(EventCam)->w()) ||
        uv_event_u[1] >= static_cast<T>(this->camera_calib.at(EventCam)->h())) {
      ++skipped_oob_event; continue;
    }
    // 7) Keep result
    cv::KeyPoint kp_event_undist;
    kp_event_undist.pt.x = static_cast<float>(uv_event_u[0]);
    kp_event_undist.pt.y = static_cast<float>(uv_event_u[1]);
    pts_event.push_back(kp_event_undist);
    ids_in_event.push_back(id);
  }

  return;
}
