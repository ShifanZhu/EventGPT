#include "RgbdDataIO.h"
#include <ParamHandler/ParamHandler.hpp>
#include "VisualGeomManager.hpp"
#include "engine/engine_vis_visualize.h"
#include "render/render_gl2.h"
#include "render/render_gl3.h"
#include <Timer.h>
#include "glog/logging.h"
#include <filesystem>
namespace fs = std::filesystem;

template <typename T>
RgbdDataIO<T>::RgbdDataIO(const std::string& config_file)
              : BaseIO<T>() {
  ReadConfig(config_file);  
}

template <typename T>
RgbdDataIO<T>::~RgbdDataIO() {
  Stop();
}

template <typename T>
void RgbdDataIO<T>::Stop() {
  set_running(false);
  if (rgbd_thread_.joinable()) {
    rgbd_thread_.join();
  }
  std::cout << "[RgbdDataIO] Stopped Running.\n";
}

template <typename T>
void RgbdDataIO<T>::ReadConfig(const std::string &config_file) {
  ParamHandler param_handler(config_file);
  param_handler.getString("data_path", data_path_);
  param_handler.getValue("rs_width", rs_width);
  param_handler.getValue("rs_height", rs_height);
  param_handler.getValue("rs_fps", rs_fps);
  param_handler.getBoolean("rs_enable_autoexposure", rs_enable_autoexposure);
  param_handler.getValue("rs_exposure", rs_exposure);
  param_handler.getValue("rs_gain", rs_gain);
  param_handler.getBoolean("use_raw_depth", use_raw_depth_);
}

template <typename T>
// void RgbdDataIO<T>::PushData(DataPoint<T>&& data) {
void RgbdDataIO<T>::PushData(std::shared_ptr<DataPoint<T>> data) {
  std::lock_guard<std::mutex> lock(rgbd_io_mutex_);

  if (data->type == DataType::RGBD) {
    rgbd_data_queue_.emplace(std::move(data));
  }
}

template <typename T>
void RgbdDataIO<T>::PopDataUntil(const T& time, std::vector<std::shared_ptr<DataPoint<T>>>& result) {
  std::lock_guard<std::mutex> lock(rgbd_io_mutex_);

  while (!rgbd_data_queue_.empty() && rgbd_data_queue_.front()->timestamp <= time) {
    result.emplace_back(rgbd_data_queue_.front());  // Move shared_ptr
    rgbd_data_queue_.pop();
    // LOG(INFO) << "rgbd_data_queue_ size: " << rgbd_data_queue_.size();
  }
}

template <typename T>
void RgbdDataIO<T>::RemoveFolder(std::string path) {
  try {
    std::filesystem::remove_all(path);
    std::cout << "Directory removed successfully." << std::endl;
  } catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Error removing directory: " << e.what() << std::endl;
  }
  return;
}

template <typename T>
void RgbdDataIO<T>::CreateFolder(std::string base_path, std::string folder_name) {
  // Check if the "base" folder exists
  if (!std::filesystem::exists(base_path)) {
    std::filesystem::create_directory(base_path);
    std::cout << "Created folder: " << base_path << std::endl;
  } else {
    std::cout << "Base folder " << base_path << " already exists." << std::endl;
  }

  // Check if the "folder_name" folder exists
  std::string full_path = base_path + folder_name;
  if (!std::filesystem::exists(full_path)) {
    std::filesystem::create_directory(full_path);
    std::cout << "Created folder: " << full_path << std::endl;
  } else {
    std::cout << "Folder " << full_path << " already exists." << std::endl;
  }
  return;
}

template <typename T>
void RgbdDataIO<T>::RemoveAndCreateFolders() {
  RemoveFolder(data_path_);
  CreateFolder(data_path_, "/raw_depth");
  CreateFolder(data_path_, "/depth");
  CreateFolder(data_path_, "/rgb");
  return;
}

template <typename T>
void RgbdDataIO<T>::GetLocalFilePath() {
  if (data_recording_mode_) {
    RemoveAndCreateFolders(); // only remove in data recording mode
  }
  try {
    const std::string timestamp_file = data_path_ + "/realsense_timestamp.txt";
    LOG(INFO) << "timetamp file: " << timestamp_file;

    if (data_recording_mode_) {
      std::cout << "[RgbdDataIO] Running in Data Recording mode\n";
      std::cout << "Trying to save data to: " << data_path_ << std::endl;

      // Check if file already exists, we do not want to accidently overwrite it
      if (std::filesystem::exists(timestamp_file)) {
        std::cout << "[Warning] File already exists: " << timestamp_file << std::endl;
        std::cout << "Press Enter to continue (it will append to the file)...";
        // std::cin.get();  // Wait for any key to continue  //todo if not commenting this out, two sensors cannot synchronousely work due to the wait. need to find a better way to handle this...
      }

      fout_.open(timestamp_file, std::ios::app);

    } else {
      std::cout << "[RgbdDataIO] Running in Offline mode\n";
      std::cout << "Trying to read RGBD data from: " << data_path_ << std::endl;
      fin_.open(timestamp_file);

      if (!fin_.is_open()) {
        throw std::runtime_error("[RgbdDataIO]: Failed to open the file. Check above data_path_.");
      }
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Error in GetLocalFilePath: ") + e.what());
  }
}

// Convert rs2::frame to cv::Mat
template <typename T>
cv::Mat RgbdDataIO<T>::Frame2Mat(const rs2::frame& f) {
  using namespace cv;
  using namespace rs2;

  auto vf = f.as<video_frame>();
  const int w = vf.get_width();
  const int h = vf.get_height();

  if (f.get_profile().format() == RS2_FORMAT_BGR8) {
    return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
  } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
    auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    Mat r_bgr;
    cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
    return r_bgr;
  } else if (f.get_profile().format() == RS2_FORMAT_Z16) { // current depth image format
    return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
  } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
    return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
  } else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) {
    return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
  }

  throw std::runtime_error("Frame format is not supported yet!");
}

template <typename T>
bool RgbdDataIO<T>::ProjectDepthToRgbAndEvent(const cv::Mat& depth_depth_img, cv::Mat& depth_rgb_img, cv::Mat& depth_event_img) {
  if (cam_intrinsics_cameras_.find(0) == cam_intrinsics_cameras_.end() ||
      cam_intrinsics_cameras_.find(1) == cam_intrinsics_cameras_.end()) {
    // LOG(WARNING) << "Camera intrinsics for RGB or Event camera not set!";
    return false;
  }
  size_t depth_h = depth_depth_img.rows;
  size_t depth_w = depth_depth_img.cols;
  depth_rgb_img = cv::Mat::zeros(cam_intrinsics_cameras_.at(0)->h(), cam_intrinsics_cameras_.at(0)->w(), CV_16UC1); // ushort type
  depth_event_img = cv::Mat::zeros(cam_intrinsics_cameras_.at(1)->h(), cam_intrinsics_cameras_.at(1)->w(), CV_16UC1); // ushort type
  auto rgbd_cam = cam_intrinsics_cameras_.at(0);
  auto event_cam = cam_intrinsics_cameras_.at(1);
  Mat3<T> Kdi = rgbd_cam->get_K_inv_depth_mat();
  Mat3<T> depth_to_event = rgbd_cam->GetDepthtoEvent().rotationMatrix();
  Mat3<T> depth_to_rgb = rgbd_cam->GetDepthtoRGB().rotationMatrix();
  Mat3<T> Krgb = rgbd_cam->get_new_K_mat();
  Mat3<T> Kevent = event_cam->get_new_K_mat();
  Mat3<T> KRKi_event = Kevent * depth_to_event * Kdi;
  Mat3<T> KRKi_rgb = Krgb * depth_to_rgb * Kdi;
  Vec3<T> Kt_event = Kevent * rgbd_cam->GetDepthtoEvent().translation();
  Vec3<T> Kt_rgb = Krgb * rgbd_cam->GetDepthtoRGB().translation();
  // LOG(INFO) << "Krgb: \n" << Krgb;
  // LOG(INFO) << "Kevent: \n" << Kevent;
  // LOG(INFO) << "depth_to_event: \n" << depth_to_event.matrix();
  // LOG(INFO) << "Kdi: \n" << Kdi;
  for (int y = 0; y < static_cast<int>(depth_h); ++y) {
    for (int x = 0; x < static_cast<int>(depth_w); ++x) {
      const ushort d_raw = depth_depth_img.at<ushort>(y, x);
      if (d_raw == 0) continue;                       // no depth
      const double d  = static_cast<double>(d_raw) * 0.001;  // meters
      if (d <= 0.1 || d >= 150.0) continue;            // sanity range
      const double di = 1.0 / d;

      // Map TL and BR corners of the depth pixel through the warp
      Vec3<T> pt_tl_3(x - 0.5, y - 0.5, 1.0);
      Vec3<T> pt_br_3(x + 0.5, y + 0.5, 1.0);

      // Project to RGB image
      Vec2<T> rgb_tl = rgbd_cam->pixel2pixel(pt_tl_3, KRKi_rgb,   Kt_rgb,   di);
      Vec2<T> rgb_br = rgbd_cam->pixel2pixel(pt_br_3, KRKi_rgb,   Kt_rgb,   di);

      // Project to Event image
      Vec2<T> evt_tl = rgbd_cam->pixel2pixel(pt_tl_3, KRKi_event, Kt_event, di);
      Vec2<T> evt_br = rgbd_cam->pixel2pixel(pt_br_3, KRKi_event, Kt_event, di);

      // Convert to integer pixel bounds (round to nearest)
      int u_rgb_tl = static_cast<int>(rgb_tl[0] + 0.5);
      int v_rgb_tl = static_cast<int>(rgb_tl[1] + 0.5);
      int u_rgb_br = static_cast<int>(rgb_br[0] + 0.5);
      int v_rgb_br = static_cast<int>(rgb_br[1] + 0.5);

      int u_evt_tl = static_cast<int>(evt_tl[0] + 0.5);
      int v_evt_tl = static_cast<int>(evt_tl[1] + 0.5);
      int u_evt_br = static_cast<int>(evt_br[0] + 0.5);
      int v_evt_br = static_cast<int>(evt_br[1] + 0.5);

      // Ensure TL <= BR (warps can flip)
      if (u_rgb_tl > u_rgb_br) std::swap(u_rgb_tl, u_rgb_br);
      if (v_rgb_tl > v_rgb_br) std::swap(v_rgb_tl, v_rgb_br);
      if (u_evt_tl > u_evt_br) std::swap(u_evt_tl, u_evt_br);
      if (v_evt_tl > v_evt_br) std::swap(v_evt_tl, v_evt_br);

      // Clamp to bounds
      const int W_rgb = static_cast<int>(cam_intrinsics_cameras_.at(0)->w());
      const int H_rgb = static_cast<int>(cam_intrinsics_cameras_.at(0)->h());
      const int W_evt = static_cast<int>(cam_intrinsics_cameras_.at(1)->w());
      const int H_evt = static_cast<int>(cam_intrinsics_cameras_.at(1)->h());

      // If rectangle is completely out of bounds, skip
      if (u_rgb_br < 0 || v_rgb_br < 0 || u_rgb_tl >= W_rgb || v_rgb_tl >= H_rgb) {
        // nothing to write in RGB
      } else {
        // clamp inside
        u_rgb_tl = std::max(0, std::min(u_rgb_tl, W_rgb - 1));
        v_rgb_tl = std::max(0, std::min(v_rgb_tl, H_rgb - 1));
        u_rgb_br = std::max(0, std::min(u_rgb_br, W_rgb - 1));
        v_rgb_br = std::max(0, std::min(v_rgb_br, H_rgb - 1));

        for (int py = v_rgb_tl; py <= v_rgb_br; ++py) {
          for (int px = u_rgb_tl; px <= u_rgb_br; ++px) {
            ushort &dst = depth_rgb_img.at<ushort>(py, px);
            // keep the closest depth (smallest value in mm)
            dst = (dst == 0) ? d_raw : static_cast<ushort>(std::min<int>(dst, d_raw));
          }
        }
      }

      if (u_evt_br < 0 || v_evt_br < 0 || u_evt_tl >= W_evt || v_evt_tl >= H_evt) {
        // nothing to write in Event
      } else {
        u_evt_tl = std::max(0, std::min(u_evt_tl, W_evt - 1));
        v_evt_tl = std::max(0, std::min(v_evt_tl, H_evt - 1));
        u_evt_br = std::max(0, std::min(u_evt_br, W_evt - 1));
        v_evt_br = std::max(0, std::min(v_evt_br, H_evt - 1));

        for (int py = v_evt_tl; py <= v_evt_br; ++py) {
          for (int px = u_evt_tl; px <= u_evt_br; ++px) {
            ushort &dst = depth_event_img.at<ushort>(py, px);
            dst = (dst == 0) ? d_raw : static_cast<ushort>(std::min<int>(dst, d_raw));
          }
        }
      }
    }
  }
  return true;
}

// use inline to be safe and avoid linker error
inline cv::Mat try_read(const std::string& p) {
  return std::filesystem::exists(p) ? cv::imread(p, cv::IMREAD_UNCHANGED) : cv::Mat();
}

// In data_path_ folder, there should be three folders: rgb, depth, realsense_timestamp.txt.
template <typename T>
void RgbdDataIO<T>::GoOffline()
{
  std::cout << "[RgbdDataIO] Entering Offline mode\n";
  set_running(true);
  data_recording_mode_ = false;
  GetLocalFilePath();

  rgbd_thread_ = std::thread([this]() {
    std::string line;
    std::vector<std::string> lines;
    lines.reserve(3);

    while (running_ && std::getline(fin_, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      lines.emplace_back(std::move(line));

      if (lines.size() == 3) {
        double depth_depth_time = std::stod(lines[0].substr(0, 16)) * 1e-6;
        if (depth_depth_time < timer_->CurrentTime() - 1) { // too far behind
          lines.clear();
          continue;
        }
        cv::Mat depth_depth_img, depth_rgb_img_orig, depth_rgb_img, depth_event_img, rgb_img;
        double rgb_time, depth_rgb_time, depth_event_time;
        bool valid_depth_projection = true;
        if (use_raw_depth_) {
          std::string line0 = lines[0];
          std::size_t pos = line0.find("rgb");
          if (pos != std::string::npos) {
            line0.replace(pos, 3, "depth");
          }

          std::string path1 = data_path_ + "/raw_depth/" + line0;
          depth_depth_img = try_read(path1);

          if (depth_depth_img.empty()) {
            if (auto pos = line0.find("_depth"); pos != std::string::npos) {
              std::string alt = line0; 
              alt.erase(pos, 6);  // remove "_depth"
              std::string path2 = data_path_ + "/raw_depth/" + alt;
              depth_depth_img = try_read(path2);
            }
          }

          rgb_img = cv::imread(data_path_ + "/rgb/" + lines[2], cv::IMREAD_UNCHANGED);
          valid_depth_projection = ProjectDepthToRgbAndEvent(depth_depth_img, depth_rgb_img, depth_event_img);
          depth_event_time = depth_depth_time;
          depth_rgb_time = depth_event_time;
          rgb_time = std::stod(lines[2].substr(0, 16)) * 1e-6;

          // # define TEST_DEPTH_RGB_ALIGNMENT
          # ifdef TEST_DEPTH_RGB_ALIGNMENT
          cv::Mat depth_jet;
          cv::Mat depth_jet_orig;
          cv::Mat depth_rgb_jet;
          cv::Mat depth_event_jet;
          if (!depth_rgb_img.empty() && !depth_event_img.empty() && cam_intrinsics_cameras_.find(0) != cam_intrinsics_cameras_.end()) {
            depth_rgb_img_orig = cv::imread(data_path_ + "/depth/" + lines[0], cv::IMREAD_UNCHANGED);
            cv::Mat depth16_orig = depth_rgb_img_orig;                     // CV_16U;
            cv::Mat valid_orig   = depth16_orig > 0;
            cv::Mat depth8_orig;
            depth16_orig.convertTo(depth8_orig, CV_8U, 1/55.0);
            depth8_orig.setTo(0, ~valid_orig);
            cv::applyColorMap(depth8_orig, depth_jet_orig, cv::COLORMAP_JET);
            std::string depth_file_orig = data_path_ + "/debug/" + std::to_string(depth_depth_time) + "_depth_orig.png";
            cv::imwrite(depth_file_orig, depth_jet_orig);

            cv::Mat depth16 = depth_rgb_img;                     // CV_16U
            cv::Mat valid   = depth16 > 0;
            cv::Mat depth8;
            depth16.convertTo(depth8, CV_8U, 1/55.0);
            depth8.setTo(0, ~valid);
            cv::applyColorMap(depth8, depth_jet, cv::COLORMAP_JET);
            std::string depth_file = data_path_ + "/debug/" + std::to_string(depth_depth_time) + "_depth_new.png";
            cv::imwrite(depth_file, depth_jet);

  
            cv::Mat depth_rgb16 = depth_rgb_img;                     // CV_16U
            cv::Mat valid_rgb   = depth_rgb16 > 0;
            cv::Mat depth_rgb8;
            depth_rgb16.convertTo(depth_rgb8, CV_8U, 1/55.0);
            depth_rgb8.setTo(0, ~valid_rgb);
            cv::applyColorMap(depth_rgb8, depth_rgb_jet, cv::COLORMAP_JET);
  
            cv::Mat depth_event16 = depth_event_img;                     // CV_16U
            cv::Mat valid_event   = depth_event16 > 0;
            cv::Mat depth_event8;
            depth_event16.convertTo(depth_event8, CV_8U, 1/55.0);
            depth_event8.setTo(0, ~valid_event);
            cv::applyColorMap(depth_event8, depth_event_jet, cv::COLORMAP_JET);
            cv::imshow("Depth Image", depth_jet);
            cv::imshow("Depth to RGB", depth_rgb_jet);
            cv::imshow("Depth to Event", depth_event_jet);

            cv::Mat rgb_img_undist = cam_intrinsics_cameras_.at(0)->undistort_image(rgb_img);
            std::string rgb_file_orig = data_path_ + "/debug/" + std::to_string(depth_depth_time) + "_rgb_orig.png";
            cv::imwrite(rgb_file_orig, rgb_img);
            std::string rgb_file = data_path_ + "/debug/" + std::to_string(depth_depth_time) + "_rgb_undist.png";
            cv::imwrite(rgb_file, rgb_img_undist);
            cv::Mat alignment_result;
            double alpha = 0.6, beta = 0.4;
            cv::addWeighted(rgb_img_undist, alpha, depth_jet, beta, 0.0, alignment_result);
            std::string alignment_file = data_path_ + "/debug/" + std::to_string(depth_depth_time) + "_alignment.png";
            cv::imwrite(alignment_file, alignment_result);
            cv::imshow("Alignment Result", alignment_result);
            cv::imshow("RGB Image", rgb_img);
            cv::waitKey(1);
          }
          # endif

        } else { // use the pre-projected depth images
          depth_rgb_img = cv::imread(data_path_ + "/depth/" + lines[0], cv::IMREAD_UNCHANGED);
          depth_event_img = cv::imread(data_path_ + "/depth/" + lines[1], cv::IMREAD_UNCHANGED);
          depth_event_time = depth_depth_time;
          depth_rgb_time = depth_event_time;
          rgb_img = cv::imread(data_path_ + "/rgb/" + lines[2], cv::IMREAD_UNCHANGED);
          rgb_time = std::stod(lines[2].substr(0, 16)) * 1e-6;
        }


        if (valid_depth_projection) {
          auto rgbd_point = std::make_shared<DataPoint<T>>(
            rgb_time,
            DataType::RGBD,
            std::make_shared<RgbdData<T>>(
              TimedImage<T>(rgb_time, std::move(rgb_img)),       
              TimedImage<T>(depth_rgb_time, std::move(depth_rgb_img)),  
              TimedImage<T>(depth_event_time, std::move(depth_event_img))
            )
          );
          PushData(rgbd_point);
        }

        while (depth_depth_time > timer_->CurrentTime() + 1) { // 1 sec ahead
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        lines.clear();
      }
    }
    set_running(false);
    std::cout << "[RgbdDataIO] Exiting Offline mode\n";
  });
}

// todo Zhipeng
// Check https://m3ed.io/sequences/#spot  (Spot legged robot)
// Parse downloaded RGBD data and push to queue
// template <typename T>
// void RgbdDataIO<T>::GoOfflineM3ED() {

// }

// Enable Rgbd camera and config some parameters
template <typename T>
bool RgbdDataIO<T>::SetupRgbdCameraHardware() {
  if (!serial_.empty()) {
    cfg_.enable_device(serial_);
  }
  cfg_.enable_stream(RS2_STREAM_DEPTH, rs_width, rs_height, RS2_FORMAT_Z16, rs_fps);
  cfg_.enable_stream(RS2_STREAM_COLOR, rs_width, rs_height, RS2_FORMAT_BGR8, rs_fps);
  LOG(INFO) << "Enable RGBD Camera Hardware";
  // cfg_.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F); // we do not need acce
  // cfg_.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400); // we do not need gyro
  pipe_.start(cfg_);
  senser_ = pipe_.get_active_profile().get_device().query_sensors()[1]; // RGB
  if (rs_enable_autoexposure) {
    senser_.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
    senser_.set_option(RS2_OPTION_BRIGHTNESS, 0);
    senser_.set_option(RS2_OPTION_CONTRAST, 50);
    senser_.set_option(RS2_OPTION_GAMMA, 295);
    senser_.set_option(RS2_OPTION_SATURATION, 35);
    senser_.set_option(RS2_OPTION_SHARPNESS, 51);
  } else {
    senser_.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
    senser_.set_option(RS2_OPTION_EXPOSURE, rs_exposure);
    senser_.set_option(RS2_OPTION_GAIN, rs_gain);
  }
  LOG(INFO) << "RealSense Camera is set up successfully.";
  return true;
}

template <typename T>
inline void RgbdDataIO<T>::SaveImage(const std::string& path, const cv::Mat& img) {
  cv::imwrite(path, img);
}

template <typename T>
void RgbdDataIO<T>::GoOnline() {
  std::cout << "[RgbdDataIO] Reading RGBD data from RealSense Camera.\n";
  if(SetupRgbdCameraHardware()) {
    std::cout << "[RgbdDataIO] RealSense Camera is set up successfully.\n";
  } else {
    std::cerr << "[RgbdDataIO] Failed to set up RealSense Camera.\n";
    std::exit(EXIT_FAILURE);
  }
  set_running(true);
  rgbd_thread_ = std::thread([this]() {
    while (running_) {
      rs2::frameset frameset_ = pipe_.wait_for_frames(2000);
      rs2::depth_frame depth_frame = frameset_.get_depth_frame();
      rs2::video_frame color_frame = frameset_.get_color_frame();
      int64_t curr_depth_time_ = color_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)*1e3;
      int64_t curr_rgb_time_ = color_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)*1e3;

      cv::Mat rgb_img = Frame2Mat(color_frame);
      std::string str_rgb = std::to_string(curr_rgb_time_) + "_rgb.png";
      cv::Mat depth_img = Frame2Mat(depth_frame);
      cv::Mat depth_event_img = depth_img.clone(); // todo: replace with event depth image
      std::string str_depth_depth = std::to_string(curr_depth_time_) + "_depth_depth.png";
      std::string str_depth_rgb = std::to_string(curr_depth_time_) + "_depth_rgb.png";

      auto rgbd_point = std::make_shared<DataPoint<T>>(
        curr_rgb_time_,
        DataType::RGBD,
        std::make_shared<RgbdData<T>>(
          TimedImage<T>(curr_rgb_time_, std::move(rgb_img)),
          TimedImage<T>(curr_depth_time_, std::move(depth_img)),
          TimedImage<T>(curr_depth_time_, std::move(depth_event_img))
        )
      );

      PushData(rgbd_point);
    }
    set_running(false);
  });
}

// Save Rgbd data to local file
template <typename T>
void RgbdDataIO<T>::GoRecording() {
  std::cout << "[RgbdDataIO] Running in Data Recording mode\n";
  data_recording_mode_ = true;
  GetLocalFilePath();
  if(SetupRgbdCameraHardware()) {
    std::cout << "[RgbdDataIO] RealSense Camera is set up successfully.\n";
  } else {
    std::cerr << "[RgbdDataIO] Failed to set up RealSense Camera.\n";
    std::exit(EXIT_FAILURE);
  }
  set_running(true);

  rgbd_thread_ = std::thread([this]() {
    while (running_) {
      rs2::frameset frameset_ = pipe_.wait_for_frames(2000);
      rs2::depth_frame depth_frame = frameset_.get_depth_frame();
      rs2::video_frame color_frame = frameset_.get_color_frame();
      int64_t curr_depth_time_ = color_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)*1e3;
      int64_t curr_rgb_time_ = color_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)*1e3;
      // LOG(INFO) << "[RgbdDataIO] Current RGBD timestamp: " << std::setprecision(17) << curr_rgb_time_*1e-6 << std::endl;

      cv::Mat rgb_img = Frame2Mat(color_frame);
      std::string str_rgb = std::to_string(curr_rgb_time_) + "_rgb.png";
      cv::Mat depth_img = Frame2Mat(depth_frame);
      std::string str_depth_depth = std::to_string(curr_depth_time_) + "_depth_depth.png";
      std::string str_depth_rgb = std::to_string(curr_depth_time_) + "_depth_rgb.png";

      // std::cout << "[RgbdDataIO] Saving data: " << str_rgb << std::endl;

      // std::thread depthThread(RgbdDataIO<T>::SaveImage, data_path_+"/raw_depth/"+str_depth_depth, depth_img);
      // std::thread rgbThread(RgbdDataIO<T>::SaveImage, data_path_+"/rgb/"+str_rgb, rgb_img);
      std::thread depthThread(&RgbdDataIO<T>::SaveImage, this, data_path_ + "/raw_depth/" + str_depth_depth, depth_img);
      std::thread rgbThread(&RgbdDataIO<T>::SaveImage, this, data_path_ + "/rgb/" + str_rgb, rgb_img);
      rgbThread.join();
      depthThread.join();

      std::string str_depth_event = std::to_string(curr_depth_time_) + "_depth_event.png";
      fout_ << str_depth_rgb << std::endl << str_depth_event << std::endl << str_rgb << std::endl;
    }
    fout_.close();
    set_running(false);
    std::cout << "[RgbdDataIO] Exiting from Data Recording mode\n";
  });
}

template <typename T>
void RgbdDataIO<T>::set_running(bool status) {
  running_ = status;
  // std::cout << "[RgbdDataIO] running_ is set to " << status << std::endl;
}

template <typename T>
std::future<void> RgbdDataIO<T>::NotifyUpdate() {
  // Reset the promise for a new future
  completion_promise_ = std::promise<void>();

  RenderOnce();
  // Mark completion
  completion_promise_.set_value();

  return completion_promise_.get_future();
}

template <typename T>
void RgbdDataIO<T>::GoSimulation() {
  std::cout << "[RgbdDataIO] Reading RGBD data from simulation.\n";
  SetRenderParams();
  // running_ = true;
  // rgbd_thread_ = std::thread([this]() {
  //   while (running_) {
  //     std::unique_lock<std::mutex> lock(rgbd_io_mutex_);
  //     data_cond_.wait(lock, [this] { return update_requested_; });
  //     update_requested_ = false;  // Reset flag
  //     lock.unlock();

  //     int rgbd_sensor_id = mj_name2id(_mjModel, mjOBJ_CAMERA, "D455");
  //     if (rgbd_sensor_id == -1) {
  //       std::cerr << "Camera 'tracking' not found." << std::endl;
  //       return;
  //     }

  //     // Set up the viewport for rendering
  //     mjrRect viewport = {0, 0, rs_width, rs_height};

  //     mjvCamera d455;
  //     d455.type = mjCAMERA_FIXED;
  //     d455.fixedcamid = rgbd_sensor_id;  // Use the robot-attached camera
  //     d455.trackbodyid = -1;           // Not tracking any body
  //     mjv_updateScene(_mjModel, _mjData, _mj_viz_opt, NULL, &d455, mjCAT_ALL, _mj_scene);
  //     // LOG(INFO) << "near: " << _mjModel->vis.global << ", far: " << _mjModel->vis.global.far;
  //     mjr_render(viewport, _mj_scene, _mj_context);
  //     {
  //       // lock is not necessary if finished within mujoco render loop (16.6ms)
  //       // std::lock_guard<std::mutex> lck(mj_camera_mtx);
  //       mjr_readPixels(rgb_data, depth_data, viewport, _mj_context);
  //       // _mjData->get_new_camera_data = true;
  //     }

  //     {
  //       // std::lock_guard<std::mutex> lck(mj_camera_mtx);
  //       std::memcpy(this->mj_rgb_image_.data, rgb_data, rs_height * rs_width * 3);
  //       std::memcpy(this->mj_depth_image_.data, depth_data, rs_height * rs_width * sizeof(float));
  //     }

  //     cv::cvtColor(this->mj_rgb_image_, this->mj_gray_image_, cv::COLOR_RGB2GRAY);
  //     cv::flip(this->mj_gray_image_, this->mj_gray_image_, 1); // Flip horizontally
  //     cv::flip(this->mj_depth_image_, this->mj_depth_image_, 1); // Flip horizontally
  //     // float norm_depth = mj_depth_image_.at<float>(rs_height/2+95, rs_width/2-25);
  //     float extent = _mjModel->stat.extent;
  //     float near = _mjModel->vis.map.znear * extent;
  //     float far = _mjModel->vis.map.zfar * extent;
  //     // std::cout << "depth: " << (near * far) / (far + near - norm_depth * (far - near)) << std::endl;

  //     cv::imshow("Robot Camera View", mj_gray_image_);
  //     cv::imshow("Robot Camera Depth", mj_depth_image_);
  //     cv::waitKey(1);
  //     T time = _mjData->time;
  //     auto depth_shared = std::make_shared<cv::Mat>(this->mj_depth_image_);
  //     // cv::Mat depth_tmp = mj_depth_image_.clone();
  //     // cv::Mat temp_dum_mat = cv::Mat::zeros(mj_depth_image_.size(), CV_32F);
  //     auto rgbd_point = std::make_shared<DataPoint<T>>(
  //       time,
  //       DataType::RGBD,
  //       std::make_shared<RgbdData<T>>(
  //         TimedImage<T>(time, std::move(this->mj_gray_image_)),
  //         TimedImage<T>(time, *depth_shared),
  //         TimedImage<T>(time, *depth_shared)
  //       )
  //     );
  //     PushData(std::move(*rgbd_point));

  //     // Signal completion
  //     completion_promise_.set_value();
  //     completion_promise_ = std::promise<void>();  // Reset for next use
  //   }
  // });
  std::cout << "[RgbdDataIO] Exiting Simulation mode.\n";
}

template <typename T>
void RgbdDataIO<T>::SetRenderParams() {
  rgbd_sensor_id = mj_name2id(_mjModel, mjOBJ_CAMERA, "D455");
  if (rgbd_sensor_id == -1) {
    std::cerr << "Camera 'tracking' not found." << std::endl;
    return;
  }

  // Set up the viewport for rendering
  viewport = {0, 0, int(rs_width), int(rs_height)};

  d455.type = mjCAMERA_FIXED;
  d455.fixedcamid = rgbd_sensor_id;  // Use the robot-attached camera
  d455.trackbodyid = -1;           // Not tracking any body

  mj_rgb_image_ = cv::Mat(rs_height, rs_width, CV_8UC3);  // Allocates memory for RGB
  mj_depth_image_ = cv::Mat(rs_height, rs_width, CV_32F);  // Allocates memory for Depth
  // rgb_data = (unsigned char*)malloc(3 * rs_width * rs_height);
  // depth_data = (float*)malloc(rs_width * rs_height * sizeof(float));
  rgb_data.resize(3 * rs_width * rs_height);
  depth_data.resize(rs_width * rs_height);
}

template <typename T>
void RgbdDataIO<T>::RenderOnce() {
  mjv_updateScene(_mjModel, _mjData, _mj_viz_opt, NULL, &d455, mjCAT_ALL, _mj_scene);
  VisualGeomManager::GetInstance()->updateGeoms(*_mj_scene);
  mjr_render(viewport, _mj_scene, _mj_context);
  mjr_readPixels(rgb_data.data(), depth_data.data(), viewport, _mj_context);

  // std::memcpy(mj_rgb_image_.data, rgb_data, rs_height * rs_width * 3);
  // std::memcpy(mj_depth_image_.data, depth_data, rs_height * rs_width * sizeof(float));
  cv::Mat raw_rgb_image(rs_height, rs_width, CV_8UC3, rgb_data.data());
  cv::Mat raw_depth_image(rs_height, rs_width, CV_32F, depth_data.data());
  cv::cvtColor(raw_rgb_image, raw_rgb_image, cv::COLOR_RGB2BGR);

  cv::flip(raw_rgb_image, raw_rgb_image, 1); // Flip horizontally
  cv::flip(raw_depth_image, raw_depth_image, 1); // Flip horizontally

  // cv::imshow("Robot Camera View", mj_rgb_image_);
  // cv::imshow("Robot Camera Depth", mj_depth_image_);
  // cv::waitKey(1);
  T time = _mjData->time;
  auto rgbd_point = std::make_shared<DataPoint<T>>(
    time,
    DataType::RGBD,
    std::make_shared<RgbdData<T>>(
      TimedImage<T>(time, std::move(raw_rgb_image)),
      TimedImage<T>(time, std::move(raw_depth_image)),
      TimedImage<T>(time, raw_depth_image) // todo: Using depth image as a placeholder for event image
    )
  );
  PushData(rgbd_point);
}

template <typename T>
void RgbdDataIO<T>::GoSeSimulation() {
  std::cout << "[RgbdDataIO] Reading RGBD data from State Estimation simulator.\n";
}

// Explicit template instantiations
template class RgbdDataIO<double>;
template class RgbdDataIO<float>;

// float norm_depth = mj_depth_image_.at<float>(rs_height/2+95, rs_width/2-25);
// float extent = _mjModel->stat.extent;
// float near = _mjModel->vis.map.znear * extent;
// float far = _mjModel->vis.map.zfar * extent;
// std::cout << "depth: " << (near * far) / (far + near - norm_depth * (far - near)) << std::endl;