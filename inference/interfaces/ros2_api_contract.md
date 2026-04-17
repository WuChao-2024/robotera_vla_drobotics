# ROS2 API Contract (v0.1)

## 0. M7 Compatibility Profile

- ROS2 distro: Humble
- ROS domain: `211`
- RMW: `rmw_cyclonedds_cpp`
- Robot fixed management IP: `192.168.8.100`

## 1. Topics

### 1.1 `/robotera/vla/observation`

- Direction: robot/app -> inference node
- Recommended msg type: custom `robotera_vla_msgs/Observation`
- Frequency: 10-30 Hz (task dependent)
- Required fields:
  - `header.stamp`
  - `instruction_text` (string)
  - `joint_state` (float array)
  - `ee_pose` (float array: x,y,z,qx,qy,qz,qw)
  - `camera_front` (image)

### 1.2 `/robotera/vla/action_chunk`

- Direction: inference node -> robot/app
- Recommended msg type: custom `robotera_vla_msgs/ActionChunk`
- Frequency: follow control loop
- Required fields:
  - `header.stamp`
  - `chunk_id` (uint32)
  - `actions` (float array)
  - `horizon_steps` (uint32)

### 1.3 `/robotera/vla/inference_status`

- Direction: inference node -> robot/app
- Recommended msg type: custom `robotera_vla_msgs/InferenceStatus`
- Required fields:
  - `state` (`idle` | `running` | `error`)
  - `error_code` (int32)
  - `error_message` (string)
  - `model_version` (string)

## 2. Services

### 2.1 `/robotera/vla/control`

- Recommended srv type: `robotera_vla_msgs/Control`
- Supported command:
  - `load_model`
  - `start`
  - `stop`
  - `reset`
- Response:
  - `accepted` (bool)
  - `error_code` (int32)
  - `message` (string)

### 2.2 `/robotera/vla/ping`

- Recommended srv type: `std_srvs/Trigger` or custom ping
- Response:
  - `ok` (bool)
  - `service_version` (string)

## 3. M7 Sensor-Topic Compatibility (Reference)

These topics are available in the M7 manual and can feed/validate observation-side integration:

- `/camera/camera/color/image_raw` (`sensor_msgs/msg/Image`)
- `/camera/camera/color/metadata` (`realsense2_camera_msgs/msg/Metadata`)
- `/camera/camera/depth/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/camera/camera/depth/image_rect_raw` (`sensor_msgs/msg/Image`)
- `/camera/camera/depth/metadata` (`realsense2_camera_msgs/msg/Metadata`)
- `/camera/camera/extrinsics/depth_to_color` (`realsense2_camera_msgs/msg/Extrinsics`)
- `/tf_static` (`tf2_msgs/msg/TFMessage`)

## 4. Error Code Baseline

- `0`: OK
- `1001`: MODEL_NOT_LOADED
- `1002`: INVALID_OBSERVATION
- `1003`: INFERENCE_TIMEOUT
- `1004`: ACTION_PUBLISH_FAILED
- `1005`: INTERNAL_RUNTIME_ERROR

## 5. Timing and Reliability

- Observation timestamps must be monotonic.
- `inference_status` should publish at least 1 Hz heartbeat.
- Control service should be idempotent for `stop` and `reset`.
