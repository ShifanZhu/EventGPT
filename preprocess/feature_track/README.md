The high-level processing steps are 1. detect features on RGB images, 2. track them with optical flow, filter out outliers 3. project them from RGB pixel frame to event pixel frame, 4. save previous/current pixel positions and event data within time window.
# Steps
1. Implement two data readers: EventDataReader, RgbdDataReader, refer to EventsDataIO.cpp and RgbdDataIO.cpp.
2. Detect features on RGB images, assign each feature with a unique ID
3. Track each feature on following images, and save its position at each timestamp.
4. Project all features from RGB pixel frame to event pixel frame. (refer to FeatureTransform.cpp and CamBase.h CamRadtan.h)
5. Save all tracked feature position on event pixel frame. The saved info should include (a) feature id, (b) feature time window, (c) feature positions at previous timestamp and current timestamp, (d) event data within the 11*11 pixel window, note that the window should move to latest position.