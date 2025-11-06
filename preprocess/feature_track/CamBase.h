#ifndef SE_CORE_CAM_BASE_H
#define SE_CORE_CAM_BASE_H

#include <Eigen/Eigen>
#include <unordered_map>
#include "Utils/cppTypes.h"
#include <opencv2/opencv.hpp>
#include "so3.hpp"
#include "se3.hpp"

namespace se_core {

/**
 * @brief Base pinhole camera model class
 *
 * This is the base class for all our camera models.
 * All these models are pinhole cameras, thus just have standard reprojection logic.
 * See each derived class for detailed examples of each model.
 */
template <typename T>
class CamBase {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamBase(size_t width, size_t height) : _width(width), _height(height) {

    // TODO: test which one is better
    // size_before_ = cv::Size(width-1, height-1);
    size_before_ = cv::Size(width, height);
    size_after_ = cv::Size(width, height);
  }

  virtual ~CamBase() {}

  /// @brief Controls output FOV vs black borders for undistortion.
  enum class UndistortEdges {
    KeepFullSize,     ///< alpha=1: keep FOV; black edges may remain.
    RemoveBlackEdges  ///< alpha=0: minimize black edges; same output size.
    // CropValidROI   ///< Typical third option if you later support cropping.
  };

  /**
   * @brief This will set and update the camera calibration values.
   * This should be called on startup for each camera and after update!
   * @param calib Camera calibration information (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
   */
  virtual void set_value(const DMat<T> &calib) {

    // Assert we are of size eight
    assert(calib.rows() == 9);
    camera_values = calib;

    // Camera matrix
    cv::Matx<T, 3, 3> tempK;
    tempK(0, 0) = calib(0);
    tempK(0, 1) = 0;
    tempK(0, 2) = calib(2);
    tempK(1, 0) = 0;
    tempK(1, 1) = calib(1);
    tempK(1, 2) = calib(3);
    tempK(2, 0) = 0;
    tempK(2, 1) = 0;
    tempK(2, 2) = 1;
    camera_k_OPENCV = tempK;

    // RGB/event camera intrinsics
    fx_ = calib(0);
    fy_ = calib(1);
    cx_ = calib(2);
    cy_ = calib(3);
    fx_inv_ = 1.0 / fx_;
    fy_inv_ = 1.0 / fy_;

    // Distortion parameters (k1,k2,p1,p2,k3)
    cv::Vec<T, 5> tempD;
    tempD(0) = calib(4);
    tempD(1) = calib(5);
    tempD(2) = calib(6);
    tempD(3) = calib(7);
    tempD(4) = calib(8);
    camera_d_OPENCV = tempD;

    // Pick alpha from mode and compute new K
    const double alpha = (undist_edge_mode_ == UndistortEdges::KeepFullSize) ? 1.0 : 0.0;
    camera_k_OPENCV_new = cv::getOptimalNewCameraMatrix(
        get_K(), camera_d_OPENCV, size_before_, alpha, size_after_, nullptr);

    // Cache scalar new-K and inverses
    fx_new_ = camera_k_OPENCV_new(0, 0);
    fy_new_ = camera_k_OPENCV_new(1, 1);
    cx_new_ = camera_k_OPENCV_new(0, 2);
    cy_new_ = camera_k_OPENCV_new(1, 2);
    fx_inv_new_ = 1.0 / fx_new_;
    fy_inv_new_ = 1.0 / fy_new_;

    // Build undistort/rectify maps
    precalculate_undistort_rectify_map();
    precalc_distort_map();
  }

  /**
   * @brief Set K for the depth sensor (no distortion here).
   * @param K Vector [fx, fy, cx, cy] for the depth camera.
   * @post Populates camera_k_depth_OPENCV and cached inverses.
   */
  void set_depth_K(const Vec4<T> &K) {
    camera_k_depth_OPENCV(0, 0) = K(0);
    camera_k_depth_OPENCV(1, 1) = K(1);
    camera_k_depth_OPENCV(0, 2) = K(2);
    camera_k_depth_OPENCV(1, 2) = K(3);
    camera_k_depth_OPENCV(0, 1) = 0;
    camera_k_depth_OPENCV(1, 0) = 0;
    camera_k_depth_OPENCV(2, 0) = 0;
    camera_k_depth_OPENCV(2, 1) = 0;
    camera_k_depth_OPENCV(2, 2) = 1;
    fx_depth_ = K(0);
    fy_depth_ = K(1);
    cx_depth_ = K(2);
    cy_depth_ = K(3);
    fx_depth_inv_ = 1.0 / fx_depth_;
    fy_depth_inv_ = 1.0 / fy_depth_;
  }

  /**
   * @brief Set Mujoco (or similar) depth-unprojection parameters.
   * @param near   Near plane (meters).
   * @param far    Far plane (meters).
   * @param extent Optional extra param stored (not used here).
   * @note Recomputes K_new with alpha=0; does not build maps.
   */
  void set_mj_depth_parameters(const T& near, const T& far, const T& extent) {
    this->near = near;
    this->far = far;
    this->extent = extent;
    size_before_ = cv::Size(_width-1, _height-1);
    double alpha = 0; // 0 will remove black edges
    camera_k_OPENCV_new = cv::getOptimalNewCameraMatrix(get_K(), 
            camera_d_OPENCV, size_before_, alpha, size_after_, nullptr);
  }


  /**
   * @brief Build undistort/rectify maps for current K, D, and K_new.
   * @pre set_value() must have been called to initialize K/D/K_new.
   * @post Fills map_dist2undist_x_ (x-map) and map_dist2undist_y_ (y-map) as CV_32FC1.
   */
  void precalculate_undistort_rectify_map() {
    size_before_ = cv::Size(static_cast<int>(_width), static_cast<int>(_height));
    if (size_after_.area() == 0) size_after_ = size_before_;

    cv::initUndistortRectifyMap(get_K(), camera_d_OPENCV, cv::Matx33d::eye(), camera_k_OPENCV_new,
                                size_after_, CV_32FC1, map_dist2undist_x_, map_dist2undist_y_);
  }

  // Build maps that take an undistorted (K_new, D=0) image to the original distorted (K, D) image
  void precalc_distort_map() {
    // size_before_ is the size of the *distorted/original* image (your dst size)
    cv::initInverseRectificationMap(
      camera_k_OPENCV,            // original K  (distorted model)
      camera_d_OPENCV,            // original D  (distorted model)
      cv::Matx33d::eye(),         // R = I
      camera_k_OPENCV_new,        // new rectified K (the undistorted model)
      size_before_,               // output (distorted) map size
      CV_32FC1,
      map_undist2dist_x_, map_undist2dist_y_);
  }

  /**
   * @brief Undistort an image using cached maps.
   * @param src Input image (must match width/height given in ctor).
   * @return Undistorted image (same size as size_after_).
   * @note Uses INTER_NEAREST for depth types; INTER_LINEAR otherwise.
   */
  cv::Mat undistort_image(const cv::Mat& src) {
    CV_Assert(!src.empty());
    CV_Assert(src.cols == (int)_width && src.rows == (int)_height);

    int interp = cv::INTER_LINEAR;
    const int type = src.type();
    if (type == CV_16U || type == CV_32F || type == CV_32FC1 || type == CV_16UC1) {
      interp = cv::INTER_NEAREST;
    }

    cv::Mat dst;
    cv::remap(src, dst, map_dist2undist_x_, map_dist2undist_y_, interp, cv::BORDER_CONSTANT);

    return dst;
  }

  /**
   * @brief Distort an image using the camera's distortion parameters.
   *
   * @param src Input image (must match width/height given in ctor).
   * @return Distorted image (same size as input).
   */

  cv::Mat distort_image(const cv::Mat& src_undistorted) {
    CV_Assert(!src_undistorted.empty());
    CV_Assert(src_undistorted.size() == size_before_ /*or size_after_, depending on your pipeline*/);

    int interp = cv::INTER_LINEAR;
    const int depth = CV_MAT_DEPTH(src_undistorted.type());
    if (depth == CV_16U || depth == CV_32F) interp = cv::INTER_NEAREST;

    cv::Mat dst_distorted;
    cv::remap(src_undistorted, dst_distorted,
              map_undist2dist_x_, map_undist2dist_y_,
              interp, cv::BORDER_CONSTANT);
    return dst_distorted;
  }

  // -------- Abstract distortion/undistortion (implemented in derived classes)

  /**
   * @brief Undistort a distorted pixel to normalized camera coords (K^-1).
   * @param uv_dist Distorted pixel coordinates.
   * @return Normalized coordinates (z=1 ray on normalized plane).
   */
  virtual Vec2<T> undistort_norm(const Vec2<T> &uv_dist) = 0;

  /**
   * @brief Undistort to pixel coordinates (original K).
   * @param uv_dist Distorted pixel coordinates.
   * @return Undistorted pixel coordinates under original K.
   */
  virtual Vec2<T> undistort_pxl_frame(const Vec2<T> &uv_dist) = 0;

  /**
   * @brief Undistort to pixel coordinates using the "new" K (K_new).
   * @param uv_dist Distorted pixel coordinates.
   * @return Undistorted pixel coordinates under K_new.
   */
  virtual Vec2<T> undistort_pxl_frame_new_K(const Vec2<T> &uv_dist) = 0;

  /**
   * @brief Convenience: undistort to normalized coords, return as cv::Point_.
   */
  cv::Point_<T> undistort_cv_norm(const cv::Point_<T> &uv_dist) {
    // Eigen::Vector2f ept1, ept2;
    Vec2<T> ept1, ept2;
    ept1 << uv_dist.x, uv_dist.y;
    ept2 = undistort_norm(ept1);
    cv::Point_<T> pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Convenience: undistort to pixel coords under K_new (cv::Point_).
   */
  cv::Point_<T> undistort_cv_pxl_frame_new_K(const cv::Point_<T> &uv_dist) {
    Vec2<T> ept1, ept2;
    ept1 << uv_dist.x, uv_dist.y;
    ept2 = undistort_pxl_frame_new_K(ept1);
    cv::Point_<T> pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Convenience: undistort to pixel coords under original K (cv::Point_).
   */
  cv::Point_<T> undistort_cv_pxl_frame(const cv::Point_<T> &uv_dist) {
    Vec2<T> ept1, ept2;
    ept1 << uv_dist.x, uv_dist.y;
    ept2 = undistort_pxl_frame(ept1);
    cv::Point_<T> pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  virtual Vec2<T> distort(const Vec2<T> &uv_norm) = 0;
  /// @brief Distort using K_new (override in derived class if needed).
  virtual Vec2<T> distort_new_K(const Vec2<T> &uv_norm) {};
  /// @brief Distort a pixel (K->distorted) under K_new (override as needed).
  virtual Vec2<T> distort_pxl_new_K(const Vec2<T> &uv_pxl) {};
  /// @brief Distort a pixel (K->distorted) under original K (override as needed).
  virtual Vec2<T> distort_pxl(const Vec2<T> &uv_pxl) {};

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  cv::Point_<T> distort_cv(const cv::Point_<T> &uv_norm) {
    Eigen::Vector2f ept1, ept2;
    ept1 << uv_norm.x, uv_norm.y;
    ept2 = distort(ept1);
    cv::Point_<T> pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Computes the derivative of raw distorted to normalized coordinate.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
   */
  virtual void compute_distort_jacobian(const Vec2<T> &uv_norm, DMat<T> &H_dz_dzn, DMat<T> &H_dz_dzeta) = 0;
  /**
   * @brief Computes the derivative of undistorted to normalized coordinate.
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   */
  virtual void compute_dz_dzn_new_K(DMat<T> &H_dz_dzn) {
    // Jacobian of undistorted pixel to normalized pixel
    H_dz_dzn = DMat<T>::Zero(2, 2);
    H_dz_dzn(0, 0) = this->get_new_K()(0, 0);
    H_dz_dzn(1, 1) = this->get_new_K()(1, 1);
  }

  /**
   * @brief Read depth (meters) at (u,v), fallback to 4-neighborhood min.
   * @param depth_img CV_16U depth (millimeters).
   * @param u,v Pixel indices.
   * @return Best depth in meters, or -1 if invalid.
   */
  T get_depth(const cv::Mat& depth_img, size_t u, size_t v) {
    constexpr T scale = 0.001;  // Convert depth values to meters

    // Bounds check
    if (u >= depth_img.cols || v >= depth_img.rows) {
      return -1.0;
    }

    // Direct pointer access (SIMD optimized)
    const ushort* row_ptr = depth_img.ptr<ushort>(v);
    T best_depth = row_ptr[u] * scale;

    // // If depth is valid, return immediately
    // if (d > 0.1) {
    //     return d;
    // }

    // Define search pattern (left, up, right, down)
    static const int dx[4] = {-2, 0, 2, 0};
    static const int dy[4] = {0, -2, 0, 2};

    // T best_depth = std::numeric_limits<T>::max();  // Initialize to max value

    // Search in 4-neighborhood
    for (int i = 0; i < 4; i++) {
      int new_u = u + dx[i];
      int new_v = v + dy[i];

      // Ensure within bounds
      if (new_u >= 0 && new_u < depth_img.cols && new_v >= 0 && new_v < depth_img.rows) {
        const ushort* neighbor_ptr = depth_img.ptr<ushort>(new_v);
        T neighbor_d = neighbor_ptr[new_u] * scale;

        // Select the smallest valid depth in range (0.4m - 50m)
        if (neighbor_d > 0.05 && neighbor_d < 50) {
          best_depth = std::min(best_depth, neighbor_d);
        }
      }
    }

    // If no valid depth was found, return -1.0
    return (best_depth <= 0.0) ? -1.0 : best_depth;
  }


  /**
   * @brief Minimum valid depth (meters) in a (2*range+1)^2 window.
   * @return Min depth in meters, or -1 if none valid.
   */
  T get_min_depth_in_range(const cv::Mat& depth_img, size_t u, size_t v, size_t range) {
    constexpr T scale = 0.001;  // Convert depth values to meters

    // Bounds check
    if (u >= (size_t)depth_img.cols || v >= (size_t)depth_img.rows) {
      return -1.0;
    }

    T best_depth = std::numeric_limits<T>::max();

    // Loop over window [u-range, u+range], [v-range, v+range]
    for (int dv = -(int)range; dv <= (int)range; ++dv) {
      int new_v = (int)v + dv;
      if (new_v < 0 || new_v >= depth_img.rows) continue;

      const ushort* row_ptr = depth_img.ptr<ushort>(new_v);

      for (int du = -(int)range; du <= (int)range; ++du) {
        int new_u = (int)u + du;
        if (new_u < 0 || new_u >= depth_img.cols) continue;

        T d = row_ptr[new_u] * scale;

        // Select the smallest valid depth in range (0.05m - 50m)
        if (d > 0.05 && d < 50.0) {
          best_depth = std::min(best_depth, d);
        }
      }
    }

    // If no valid depth was found, return -1.0
    return (best_depth == std::numeric_limits<T>::max()) ? -1.0 : best_depth;
  }

  /**
   * @brief Convert normalized depth buffer value to metric depth (m), with
   *        8-neighborhood min check (for Mujoco-like depth).
   * @param depth_img CV_32F depth in normalized space.
   * @param x,y Pixel indices.
   * @return Min valid depth (m) in local neighborhood or -1 if invalid.
   */
  T get_depth_mj(const cv::Mat& depth_img, size_t x, size_t y) {
    // Bounds check
    if (x >= depth_img.cols || y >= depth_img.rows) {
      return -1.0;
    }

    // Compute depth at (x, y)
    // T best_depth = static_cast<T>((this->near * this->far) / 
    //     (this->far + this->near - depth_img.at<float>(y, x) * (this->far - this->near)));
    T best_depth = static_cast<T>((this->near) / 
        (1.0 - depth_img.at<float>(y, x) * (1.0 - this->near/this->far)));
    // Define search pattern (left, up, right, down)
    static const int dx[8] = {-2, 0, 2, 0, -4, 0, 4, 0};
    static const int dy[8] = {0, -2, 0, 2, 0, -4, 0, 4};

    // Search for the smallest valid depth in the 4-neighborhood
    for (int i = 0; i < 8; i++) {
      int new_x = x + dx[i];
      int new_y = y + dy[i];

      // Ensure within bounds
      if (new_x >= 0 && new_x < depth_img.cols && new_y >= 0 && new_y < depth_img.rows) {
        T neighbor_depth = static_cast<T>((this->near) / 
              (1.0 - depth_img.at<float>(new_y, new_x) * (1.0 - this->near/this->far)));

        // Select the smallest valid depth in range (0.4m - 50m)
        if (neighbor_depth > 0.05 && neighbor_depth < 50) {
          best_depth = std::min(best_depth, neighbor_depth);
        }
      }
    }

    // If best_depth is outside the valid range, return -1.0
    return (best_depth > 0.05 && best_depth < 50) ? best_depth : -1.0;
  }


  /// @brief Raw calibration vector (as set by set_value()).
  DMat<T> get_value() { return camera_values; }

  /// @brief Get original intrinsics K (OpenCV 3x3 double).
  cv::Matx33d get_K() { return camera_k_OPENCV; }

  /// @brief Get depth camera intrinsics K_depth (OpenCV 3x3 double).
  cv::Matx33d get_K_depth() { return camera_k_depth_OPENCV; }

  /// @brief Convenience: inverse of depth K (computed on the fly).
  cv::Matx33d get_K_inv_depth() { return get_K_depth().inv(); }

  /// @brief Get undistorted intrinsics K_new chosen by alpha/mode.
  cv::Matx33d get_new_K() { return camera_k_OPENCV_new; }

  /// @brief Get K as Eigen::Matrix (double) for math routines.
  Mat3<T> get_K_mat() { 
    Mat3<T> K;
    K << camera_k_OPENCV(0, 0), camera_k_OPENCV(0, 1), camera_k_OPENCV(0, 2),
         camera_k_OPENCV(1, 0), camera_k_OPENCV(1, 1), camera_k_OPENCV(1, 2),
         camera_k_OPENCV(2, 0), camera_k_OPENCV(2, 1), camera_k_OPENCV(2, 2);
    return K;
  }

  /// @brief Get K_new as Eigen::Matrix (double) for math routines.
  Mat3<T> get_new_K_mat() { 
    Mat3<T> K_new;
    K_new << camera_k_OPENCV_new(0, 0), camera_k_OPENCV_new(0, 1), camera_k_OPENCV_new(0, 2),
             camera_k_OPENCV_new(1, 0), camera_k_OPENCV_new(1, 1), camera_k_OPENCV_new(1, 2),
             camera_k_OPENCV_new(2, 0), camera_k_OPENCV_new(2, 1), camera_k_OPENCV_new(2, 2);
    return K_new;
  }

  /// @brief Get depth K as Eigen::Matrix (double).
  Mat3<T> get_K_depth_mat() { 
    Mat3<T> K_depth;
    K_depth << camera_k_depth_OPENCV(0, 0), camera_k_depth_OPENCV(0, 1), camera_k_depth_OPENCV(0, 2),
               camera_k_depth_OPENCV(1, 0), camera_k_depth_OPENCV(1, 1), camera_k_depth_OPENCV(1, 2),
               camera_k_depth_OPENCV(2, 0), camera_k_depth_OPENCV(2, 1), camera_k_depth_OPENCV(2, 2);
    return K_depth;
  }

  /// @brief Get inverse(depth K) as Eigen::Matrix (double).
  Mat3<T> get_K_inv_depth_mat() { 
    return get_K_depth_mat().inverse();
  }

  /// @brief Get distortion vector D (OpenCV order: k1,k2,p1,p2,k3).
  cv::Vec<double, 5> get_D() { return camera_d_OPENCV; }

  /// @brief Get distortion vector D (OpenCV order: k1,k2,p1,p2,k3) as Eigen::Vector.
  Vec5<T> get_D_mat() {
    Vec5<T> D;
    D << camera_d_OPENCV[0], camera_d_OPENCV[1], camera_d_OPENCV[2], camera_d_OPENCV[3], camera_d_OPENCV[4];
    return D;
  }

  /// @brief Image width in pixels.
  size_t w() { return _width; }

  /// @brief Image height in pixels.
  size_t h() { return _height; }

  // -------- Extrinsics getters/setters

  /// @brief Depth → Event transform.
  const Sophus::SE3<T> GetDepthtoEvent() { return depth_to_event_; }

  /// @brief Depth → RGB transform.
  const Sophus::SE3<T> GetDepthtoRGB() { return depth_to_rgb_; }

  /// @brief IMU → RGB transform.
  const Sophus::SE3<T> GetIMUtoRGB() { return imu_to_rgb_; }

  /// @brief RGB → Event transform.
  const Sophus::SE3<T> GetRGBtoEvent() { return rgb_to_event_; }

  /// @brief Set Depth → Event extrinsic.
  void SetDepthtoEvent(Sophus::SE3<T> ex_pose) { depth_to_event_ = ex_pose; }

  /// @brief Set RGB → Event extrinsic.
  void SetRGBtoEvent(Sophus::SE3<T> ex_pose) { rgb_to_event_ = ex_pose; }

  /// @brief Set Depth → RGB extrinsic.
  void SetDepthtoRGB(Sophus::SE3<T> ex_pose) { depth_to_rgb_ = ex_pose; }

  /// @brief Set IMU → RGB extrinsic.
  void SetIMUtoRGB(Sophus::SE3<T> ex_pose) { imu_to_rgb_ = ex_pose; }

  /// @brief Set IMU → Event extrinsic (note: currently assigns imu_to_rgb_).
  void SetIMUtoEvent(Sophus::SE3<T> ex_pose) { imu_to_rgb_ = ex_pose; }

  /// -------- Coordinate transforms

  /**
   * @brief Transform a 3D point between camera frames.
   * @param p_c   Point in source camera frame.
   * @param T_c_c SE3 from source to target camera.
   * @return Point in target camera frame.
   */
  inline Vec3<T> camera2camera(const Vec3<T>& p_c, const Sophus::SE3<T>& T_c_c) {
    return T_c_c * p_c;
  }

  /**
   * @brief Project camera-frame point to pixel (original K).
   * @param p_c Point in camera frame.
   * @return Pixel coordinates (u,v).
   */
  inline Vec2<T> camera2pixel(Vec3<T> &p_c) {
    T zi = 1.0 / p_c(2, 0);
    return Vec2<T>(fx_ * p_c(0, 0) * zi + cx_, fy_ * p_c(1, 0) * zi + cy_);
  }

  /**
   * @brief Project camera-frame point to pixel (new K).
   */
  inline Vec2<T> camera2pixel_new_K(Vec3<T> &p_c) {
    T zi = 1.0 / p_c(2, 0);
    return Vec2<T>(fx_new_ * p_c(0, 0) * zi + cx_new_, fy_new_ * p_c(1, 0) * zi + cy_new_);
  }

  /**
   * @brief Back-project pixel to camera frame with known depth (original K).
   * @param p_p   Pixel (u,v).
   * @param depth Depth in meters (defaults to 1).
   */
  inline Vec3<T> pixel2camera(const Vec2<T>& p_p, const T& depth = 1.) {
    return Vec3<T>(
      (p_p(0, 0) - cx_) * depth * fx_inv_,
      (p_p(1, 0) - cy_) * depth * fy_inv_,
      depth
    );
  }

  /**
   * @brief Back-project pixel to unit-depth ray (original K).
   */
  inline Vec3<T> pixel2camera(const Vec2<T>& p_p) {
    return Vec3<T>(
      (p_p(0, 0) - cx_) * fx_inv_,
      (p_p(1, 0) - cy_) * fy_inv_,
      1.
    );
  }

  /**
   * @brief Back-project pixel to depth camera frame with known depth (depth K).
   */
  inline Vec3<T> pixel2depth_camera(const Vec2<T>& p_p, const T& depth = 1.) {
    return Vec3<T>(
      (p_p(0, 0) - cx_depth_) * depth * fx_depth_inv_,
      (p_p(1, 0) - cy_depth_) * depth * fy_depth_inv_,
      depth
    );
  }

  /**
   * @brief Back-project pixel to unit-depth ray (depth K).
   */
  inline Vec3<T> pixel2depth_camera(const Vec2<T>& p_p) {
    return Vec3<T>(
      (p_p(0, 0) - cx_depth_) * fx_depth_inv_,
      (p_p(1, 0) - cy_depth_) * fy_depth_inv_,
      1.
    );
  }

  /**
   * @brief Back-project pixel to camera frame with known depth (new K).
   */
  inline Vec3<T> pixel2camera_new_K(const Vec2<T>& p_p, const T& depth = 1.) {
    return Vec3<T>(
      (p_p(0, 0) - cx_new_) * depth * fx_inv_new_,
      (p_p(1, 0) - cy_new_) * depth * fy_inv_new_,
      depth
    );
  }

  /**
   * @brief Back-project pixel to unit-depth ray (new K).
   */
  inline Vec3<T> pixel2camera_new_K(const Vec2<T>& p_p) {
    return Vec3<T>(
      (p_p(0, 0) - cx_new_) * fx_inv_new_,
      (p_p(1, 0) - cy_new_) * fy_inv_new_,
      1.
    );
  }

  /**
   * @brief Warp a homogeneous pixel vector by KRKi and Kt with inv-depth.
   * @param pxl_3d   Homogeneous pixel (u,v,1).
   * @param KRKi     K2 * R * K1^{-1}.
   * @param Kt       K2 * t.
   * @param depth_inv Inverse depth (1/z) in source camera.
   * @return Target pixel coordinates.
   */
  inline Vec2<T> pixel2pixel(const Vec3<T>& pxl_3d, const Mat3<T>& KRKi, const Vec3<T>& Kt, const T& depth_inv) {
    Vec3<T> ptp2 = KRKi * pxl_3d + Kt * depth_inv;
    T id = 1.0/ptp2[2];
    return Vec2<T>(ptp2[0]*id, ptp2[1]*id);
  }

protected:
  /// @brief Default-protected ctor for derived classes that set dims later.
  CamBase() = default;

  // ---- Stored calibration and state

  DMat<T>   camera_values;             ///< Raw calibration vector.

  cv::Matx33d camera_k_OPENCV;         ///< Original K (OpenCV).
  cv::Matx33d camera_k_depth_OPENCV;   ///< Depth K (OpenCV).
  cv::Matx33d camera_k_inv_depth_OPENCV; ///< (reserved) inverse depth K.
  cv::Matx33d camera_k_OPENCV_new;     ///< New K after undistortion (alpha).

  cv::Vec<double, 5> camera_d_OPENCV;  ///< Distortion (k1,k2,p1,p2,k3).

  size_t _width  = 0;                  ///< Image width.
  size_t _height = 0;                  ///< Image height.

  UndistortEdges undist_edge_mode_ = UndistortEdges::KeepFullSize; ///< Edge policy for K_new.
  cv::Mat map_dist2undist_x_, map_dist2undist_y_;    ///< Remap maps: x-map, y-map (CV_32FC1).
  cv::Mat map_undist2dist_x_, map_undist2dist_y_;
  cv::Size size_before_, size_after_; ///< Input/output sizes for undistortion.

  // Extrinsics between sensors (as needed by your system)
  Sophus::SE3<T> robot_to_imu_, imu_to_marker_, imu_to_lidar_;
  Sophus::SE3<T> depth_to_event_, depth_to_rgb_, rgb_to_event_, imu_to_rgb_;

  // Depth unprojection parameters (for simulator formats)
  T near = 0, far = 0, extent = 0;

  // Cached intrinsics (original/new/depth) and inverses
  T fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  T fx_depth_ = 0, fy_depth_ = 0, cx_depth_ = 0, cy_depth_ = 0;
  T fx_depth_inv_ = 0, fy_depth_inv_ = 0;
  T fx_new_ = 0, fy_new_ = 0, cx_new_ = 0, cy_new_ = 0;
  T fx_inv_ = 0, fy_inv_ = 0;
  T fx_inv_new_ = 0, fy_inv_new_ = 0;
};

} // namespace se_core

#endif /* SE_CORE_CAM_BASE_H */