#include <glog/logging.h>
#include "EventsDataIO.h"
#include <ParamHandler/ParamHandler.hpp>
#include "VisualGeomManager.hpp"
#include "engine/engine_vis_visualize.h"
#include "render/render_gl2.h"
#include "render/render_gl3.h"
#include <Timer.h>
#include <metavision/sdk/stream/camera.h>
// #include <dv-processing/io/mono_camera_recording.hpp>
// #include <dv-processing/core/event.hpp>
// #include <dv-processing/core/core.hpp>
// #include <dv-processing/core/utils.hpp>

template <typename T>
EventsDataIO<T>::EventsDataIO(const std::string& config_file)
              : BaseIO<T>() {
  ReadConfig(config_file);
}

template <typename T>
EventsDataIO<T>::~EventsDataIO() {
  // std::cout << "[EventsDataIO] DEBUGINFO called ~EventsDataIO\n";
  Stop();
}

template <typename T>
void EventsDataIO<T>::Stop() {
  std::cout << "[EventsDataIO] Stopping Running...\n";

  {
    std::lock_guard<std::mutex> lock(event_running_mutex_);
    set_running(false);
  }

  event_running_cv_.notify_one();

  if (event_thread_.joinable()) {
    event_thread_.join();
  }

  std::cout << "[EventsDataIO] Stopped Running.\n";
}

template <typename T>
void EventsDataIO<T>::ReadConfig(const std::string &config_file) {
  std::cout << "[EventsDataIO] Reading config file: " << config_file << std::endl;
  ParamHandler param_handler(config_file);
  param_handler.getString("data_path", data_path_);
  LOG(INFO) << "data path: " << data_path_;
}

template <typename T>
// void EventsDataIO<T>::PushData(DataPoint<T>&& data) {
void EventsDataIO<T>::PushData(std::shared_ptr<DataPoint<T>> data) {
  std::lock_guard<std::mutex> lock(event_io_mutex_);

  if (data->type == DataType::EVENTS) {
    event_data_queue_.emplace(std::move(data));
    // std::cout << "[DEBUG] last event_data_queue_ item's timestamp " << (long long ) dynamic_pointer_cast<EventsData<T>>(event_data_queue_.back()->data)->timestamp_ << std::endl;
    // std::cout << "[DEBUG] first pushed event's timestamp " << (long long ) (dynamic_pointer_cast<EventsData<T>>(event_data_queue_.back()->data)->events_).front().timestamp_ << std::endl;
    // std::cout << "[DEBUG] last pushed event's timestamp " << (long long ) (dynamic_pointer_cast<EventsData<T>>(event_data_queue_.back()->data)->events_).back().timestamp_ << std::endl;
  }
}

template <typename T>
int64_t EventsDataIO<T>::get_record_start_timestamp()
{
  std::ifstream file(data_path_ + "/record_start_timestamp_us.txt");
  int64_t timestamp_us;
  if (file >> timestamp_us)
  {
    return timestamp_us;
  } else {
    return -1;
  }
}

template <typename T>
void EventsDataIO<T>::PopDataUntil(const T& time, std::vector<std::shared_ptr<DataPoint<T>>>& result) {
  std::lock_guard<std::mutex> lock(event_io_mutex_);
  std::vector<EventData<T>> res_events;
  T time_us = time * 1e6;
  // std::cout << "[EventsDataIO] DEBUGINFO: Running PopDataUtil time=" << time_us << std::endl;
  // std::cout << "[EventsDataIO] event_queue before pop " << event_data_queue_.size() << std::endl;
  if (time < 0)
  {
    while (!event_data_queue_.empty()) {
      auto events_sensor_data = event_data_queue_.front();
      auto events_data = std::dynamic_pointer_cast<EventsData<T>>(events_sensor_data->data);
      for (auto& item : events_data->events_)  // * std::vector<EventData<T>>
      {
        res_events.push_back(item);
      }
      event_data_queue_.pop();
    }

    auto events_point = std::make_shared<DataPoint<T>>(
      res_events.front().timestamp_,
      DataType::EVENTS,
      std::make_shared<EventsData<T>>(res_events.front().timestamp_, 0, res_events)
    );
    result.push_back(events_point);
  } else {
    while (!event_data_queue_.empty())
    {
      auto events_sensor_data = event_data_queue_.front();
      auto events_data = std::dynamic_pointer_cast<EventsData<T>>(events_sensor_data->data);

      int popped_items = 0;
      for (auto& item : events_data->events_)  // * std::vector<EventData<T>>
      {
        // std::cout << "time comparison " << (long long)item.timestamp_ << " " << (long long)time <<std::endl;
        if (item.timestamp_ < time_us)
        {
          res_events.push_back(item);
          popped_items++;
        } else {
          break;
        }
      }
      // std::cout << "popped_items in this round " << popped_items << " ";
      // std::cout << (events_data->events_).size() << std::endl;
      if (popped_items == (events_data->events_).size())
      {
        event_data_queue_.pop();
      } else {
        auto& events_list = (events_data->events_);
        events_list.erase(events_list.begin(), events_list.begin() + popped_items);
        break;
      }
    }
    // std::cout << "[DEBUG] res_events in total " << res_events.size() << "\n";

    if (res_events.size() > 0)
    {
      auto events_point = std::make_shared<DataPoint<T>>(
        res_events.front().timestamp_,
        DataType::EVENTS,
        std::make_shared<EventsData<T>>(res_events.front().timestamp_, 0, res_events)
      );
      result.push_back(events_point);
    }
 }
}

// template <typename T>
// void EventsDataIO<T>::PopDataUntil(const T& time, std::vector<std::shared_ptr<DataPoint<T>>>& result) {
//   std::lock_guard<std::mutex> lock(event_io_mutex_);

//   while (!event_data_queue_.empty() && event_data_queue_.front()->timestamp <= time) {
//     result.emplace_back(event_data_queue_.front());  // Move shared_ptr
//     event_data_queue_.pop();
//   }
// }

// * changed to new camera's api
template <typename T>
void EventsDataIO<T>::GetLocalFilePath() {
  try {
    if (live_mode_) {
      std::cout << "data_path_ = " << data_path_ << std::endl;
      event_record_file_path_ = data_path_ + "/event_prophesee.hdf5";
      std::cout << "[EventsDataIO] Running in live mode\n";
      std::cout << "Trying to save data to: " << event_record_file_path_ << std::endl;
      // Check if file already exists, we do not want to accidently overwrite it
      if (std::filesystem::exists(event_record_file_path_)) {
        std::cout << "[Warning] File already exists: " << event_record_file_path_ << std::endl;
        std::cout << "Press Enter to continue (it will append to the file)...";
        // std::cin.get();  // Wait for any key to continue
      }
    } else {
      // cam_read_file_path_ = data_path_ + "/event_prophesee_denoised.hdf5";
      cam_read_file_path_ = data_path_ + "/event_prophesee.hdf5";
      std::cout << "[EventsDataIO] Read from local dataset.\n";
      std::cout << "Trying to read events from: " << cam_read_file_path_ << std::endl;
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Error in GetLocalFilePath: ") + e.what());
  }
}

template <typename T>
void EventsDataIO<T>::GoOffline() {
  GoOfflineTxt();  // * Redirect to txt file for debugging!!! TODO
  // std::cout << "[EventsDataIO_prophesee] Running in Offline mode\n";
  // running_ = true;
  // live_mode_ = false; // offline mode
  // GetLocalFilePath();
  // if(SetupEventCamera()) {
  //   std::cout << "[EventsDataIO] Event Camera is set up successfully.\n";
  // } else {
  //   std::cerr << "[EventsDataIO] Failed to set up EventCamera Camera.\n";
  //   std::exit(EXIT_FAILURE);
  // }

  // record_start_timestamp_us_ = get_record_start_timestamp();

  // event_thread_ = std::thread([this]() {
  //   std::unique_lock<std::mutex> lock(event_running_mutex_);
  //   event_running_cv_.wait(lock, [this]() { return !cam.is_running() || !running_; });
  //   cam.stop();
  //   set_running(false);
  //   std::cout << "[EventsDataIO] Exiting Offline mode\n";
  // });
}

// todo Zhipeng
// Check https://m3ed.io/sequences/#spot  (Spot legged robot)
// Parse downloaded event data and push to queue
// template <typename T>
// void EventsDataIO<T>::GoOfflineM3ED() {

// }

// template <typename T>
// void EventsDataIO<T>::GoOfflineAedat() {
//   std::cout << "[EventsDataIO_prophesee] Running in Offline mode\n";
//   running_ = true;
//   live_mode_ = false; // offline mode
//   GetLocalFilePathAedat();
//   if(SetupEventCameraAedat()) {
//     std::cout << "[EventsDataIO] Event Camera is set up successfully.\n";
//   } else {
//     std::cerr << "[EventsDataIO] Failed to set up EventCamera Camera.\n";
//     std::exit(EXIT_FAILURE);
//   }

//   event_thread_ = std::thread([this]() {
//     std::unique_lock<std::mutex> lock(event_running_mutex_);
//     set_running(true);
//     while(monoCameraRecordingPtr.isRunning())
//     {
//       if (const std::optional<dv::EventStore> events = monoCameraRecordingPtr->getNextEventBatch(); events.has_value()) {
//         // Print received event packet information
//         // std::cout << *events << std::endl;
//         push_events_aedat(events);
//       }      
//     }
//     set_running(false);
//     std::cout << "[EventsDataIO] Exiting Offline mode\n";
//   });
// }

// template <typename T>
// void EventsAedatDataIO<T>::GetLocalFilePathAedat() {
//   try {
//     if (live_mode_) {
//       throw std::runtime_error("Function not implemented");
//     } else {
//       cam_read_file_path_ = data_path_ + "/event_dvxplorerlite320.aedat4";
//       std::cout << "[EventsDataIO] Read from local dataset.\n";
//       std::cout << "Trying to read events from: " << cam_read_file_path_ << std::endl;
//     }
//   } catch (const std::exception& e) {
//     throw std::runtime_error(std::string("Error in GetLocalFilePath: ") + e.what());
//   }
// }

// template <typename T>
// bool EventsAedatDataIO<T>::SetupEventCameraAedat() {
//   try {
//     if (live_mode_) {
//       // TODO: not implemented
//       throw std::runtime_error("Function not implemented");
//     } else {
//       monoCameraRecordingPtr = std::make_unique<dv::io::MonoCameraRecording>(cam_read_file_path_);
//     }

//     if (recording_mode_) {
//       throw std::runtime_error("Function not implemented");
//       return false;
//     }

//   } catch (...) {
//     return false;
//   }
//   return true;
// }

// template <typename T>
// void EventsAedatDataIO<T>::push_events_aedat(const std::optional<dv::EventStore>& events) {
//   for (auto &ev : *events) {
//       int64_t timestamp = ev.timestamp(); // * record the timestamp at the hdf5 file

//       events_buffer.push_back(EventData<T>(timestamp, ev.x(), ev.y(), ev.polarity()));
//       if (events_buffer.back().timestamp_ - events_buffer.front().timestamp_ > 1000)
//       {
//         std::cout << "[DEBUG] events_buffer size before pushing " << events_buffer.size() << std::endl;
//         auto event_point = std::make_shared<DataPoint<T>>(
//           events_buffer.front().timestamp_,
//           DataType::EVENTS,
//           std::make_shared<EventsData<T>>(timestamp, 0, events_buffer)
//         );
//         PushData(event_point);
//         events_buffer.clear();
//       }
//   }
// }

template <typename T>
void EventsDataIO<T>::GoOfflineTxt() {
  std::cout << "[EventsDataIO_prophesee] Running in Offline mode\n";
  running_ = true;
  live_mode_ = false; // offline mode
  GetLocalFilePathTxt();
  if(SetupEventCameraTxt()) {
    std::cout << "[EventsDataIO] Event Camera is set up successfully.\n";
  } else {
    std::cerr << "[EventsDataIO] Failed to set up EventCamera Camera.\n";
    std::exit(EXIT_FAILURE);
  }

  event_thread_ = std::thread([this]() {
    std::unique_lock<std::mutex> lock(event_running_mutex_);
    set_running(true);
    std::string line;
    T t; int x, y, p;

    int prev_time = -1;
    std::chrono::_V2::system_clock::time_point last_push_us = std::chrono::high_resolution_clock::now();
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        if (iss >> t >> x >> y >> p) {
          if (prev_time < 0)
          {
            prev_time = t;
          } else {
            int duration_us = t - prev_time;
            auto real_elapsed_us = std::chrono::high_resolution_clock::now() - last_push_us;
            if (duration_us > real_elapsed_us.count())
              std::this_thread::sleep_for(std::chrono::microseconds(duration_us-real_elapsed_us.count()));

            prev_time = t;
          }
          push_events_txt(t, x, y, p);
          last_push_us = std::chrono::high_resolution_clock::now();
        } else {
            throw std::runtime_error("Invalid txt format");
        }
    }
    std::cout << "[DEBUG] finished reading txt file!" << std::endl;
    set_running(false);
    std::cout << "[EventsDataIO] Exiting Offline mode\n";
  });
}

template <typename T>
void EventsDataIO<T>::GetLocalFilePathTxt() {
  try {
    if (live_mode_) {
      throw std::runtime_error("Function not implemented");
    } else {
      cam_read_file_path_ = data_path_ + "/events.txt";
      std::cout << "[EventsDataIO] Read from local dataset(txt).\n";
      std::cout << "Trying to read events from: " << cam_read_file_path_ << std::endl;
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Error in GetLocalFilePath: ") + e.what());
  }
}

template <typename T>
bool EventsDataIO<T>::SetupEventCameraTxt() {
  try {
    if (live_mode_) {
      // TODO: not implemented
      throw std::runtime_error("Function not implemented");
    } else {
      infile.open(cam_read_file_path_);
      if (!infile.is_open()) return false;
    }

    if (recording_mode_) {
      throw std::runtime_error("Function not implemented");
      return false;
    }

  } catch (...) {
    return false;
  }
  return true;
}

template <typename T>
void EventsDataIO<T>::push_events_txt(T t, int x, int y, int p) {
  events_buffer.push_back(EventData<T>(t*1e-6, x, y, p));
  if (events_buffer.back().timestamp_ - events_buffer.front().timestamp_ > 1e-3) {
    // std::cout << "[DEBUG] events_buffer size before pushing " << events_buffer.size() << std::endl;
    auto event_point = std::make_shared<DataPoint<T>>(
      events_buffer.front().timestamp_,
      DataType::EVENTS,
      std::make_shared<EventsData<T>>(t, 0, std::vector<EventData<T>>(events_buffer))
    );
    PushData(event_point);
    events_buffer.clear();

    while (t*1e-6 > timer_->CurrentTime()+1) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }
}


template <typename T>
void EventsDataIO<T>::push_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
  // static int counter = 0;

  // this loop allows us to get access to each event received in this callback
  for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
      // ++counter; // count each event

      // std::cout << "Event received: coordinates (" << ev->x << ", " << ev->y << "), t: " << ev->t
      //           << ", polarity: " << ev->p << std::endl;


      int64_t timestamp = ev->t; // * record the timestamp at the hdf5 file

      events_buffer.push_back(EventData<T>(timestamp, ev->x, ev->y, ev->p));
      if (events_buffer.back().timestamp_ - events_buffer.front().timestamp_ > 1000)
      {
        std::cout << "[DEBUG] events_buffer size before pushing " << events_buffer.size() << std::endl;
        auto event_point = std::make_shared<DataPoint<T>>(
          events_buffer.front().timestamp_,
          DataType::EVENTS,
          std::make_shared<EventsData<T>>(timestamp, 0, events_buffer)
        );
        PushData(event_point);
        events_buffer.clear();
      }
  }
  // std::cout << "counter " << counter << std::endl;
}

template <typename T>
bool EventsDataIO<T>::SetupEventCamera() {
  try {
    if (live_mode_) {
      cam = Metavision::Camera::from_first_available();
    } else {
      cam = Metavision::Camera::from_file(cam_read_file_path_);
    }

    cam.cd().add_callback(
      [this](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
        this->push_events(begin, end);
      }
    );
    
    cam.start();

    if (recording_mode_) {
      cam.start_recording(event_record_file_path_);
      return true; // we do not need callback for record mode
    }

  } catch (const Metavision::CameraException &e) {
    return false;
  }
  return true;
}

// Save events data to local file
template <typename T>
void EventsDataIO<T>::GoRecording() {
  live_mode_ = true;
  recording_mode_ = true;
  GetLocalFilePath();
  if(SetupEventCamera()) {
    std::cout << "[EventsDataIO] EventCamera Camera is set up successfully.\n";
    std::cout << "[EventsDataIO_prophesee] Running in Recording mode\n";
  } else {
    std::cerr << "[EventsDataIO] Failed to set up EventCamera Camera.\n";
    std::exit(EXIT_FAILURE);
  }

  // Get current time point
  auto now = std::chrono::system_clock::now();
  // Get duration since epoch in microseconds
  record_start_timestamp_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                                                              now.time_since_epoch()).count();
  // Open a text file to write the record timestamp
  std::ofstream outfile(data_path_ + "/record_start_timestamp_us.txt");

  if (outfile.is_open()) {
      outfile << record_start_timestamp_us_ << std::endl;
      outfile.close();
  } else {
      std::cerr << "[EventsDataIO] Unable to open file for writing record_start_timestamp_us." << std::endl;
  }
  
  running_ = true;

  event_thread_ = std::thread([this]() {
    std::unique_lock<std::mutex> lock(event_running_mutex_);
    event_running_cv_.wait(lock, [this]() { return !cam.is_running() || !running_; });
    cam.stop_recording(event_record_file_path_);
    cam.stop();
    set_running(false);
    std::cout << "[EventsDataIO] Exiting Recording mode\n";
  });
}

template <typename T>
void EventsDataIO<T>::set_running(bool status)
{
  running_ = status;
  std::cout << "[EventsDataIO] running_ is set to " << status << std::endl;
}

template <typename T>
void EventsDataIO<T>::GoOnline() { //TODO
  // std::cout << "[EventsDataIO] Reading EVENTS data from real event camera.\n";
  // live_mode_ = true;
  // running_ = true;

  // std::cout << "[EventsDataIO_prophesee] Running in Online mode\n";
  // // cam = Metavision::Camera::from_first_available();

  // if(SetupEventCamera()) {
  //   std::cout << "[EventsDataIO] EventCamera Camera is set up successfully.\n";
  // } else {
  //   std::cerr << "[EventsDataIO] Failed to set up EventCamera Camera.\n";
  //   std::exit(EXIT_FAILURE);
  // }

  // event_thread_ = std::thread([this]() {
  //   std::unique_lock<std::mutex> lock(event_running_mutex_);
  //   event_running_cv_.wait(lock, [this]() { return !cam.is_running() || !running_; });
  //   cam.stop();
  //   set_running(false);
  //   std::cout << "[EventsDataIO] Exiting Offline mode\n";
  // });
}

template <typename T>
void EventsDataIO<T>::GoSimulation() {
  std::cout << "[EventsDataIO] Reading EVENTS data from simulation.\n";
  // SetRenderParams();
  std::cout << "[EventsDataIO] Exiting Simulation mode.\n";
  //todo: under contruction
}

template <typename T>
void EventsDataIO<T>::GoSeSimulation() {
  std::cout << "[EventsDataIO] Reading Event data from State Estimation simulator.\n";
}

// template <typename T>
template class EventsDataIO<double>;
template class EventsDataIO<float>;
template class EventsDataIO<long long int>;