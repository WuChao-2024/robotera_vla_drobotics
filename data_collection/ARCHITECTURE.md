# Data Collection Architecture

## Responsibility Boundary

- In scope:
  - Usage documentation
  - Dataset schema definition
  - Validation and helper examples
- Out of scope:
  - Teleoperation runtime implementation
  - Native recorder implementation

## Teleoperation-to-Dataset Flow

1. Operator controls robot via existing Robotera tool.
2. Existing recorder produces raw session data.
3. Exporter aligns data to Robotera VLA custom schema.
4. Validator checks required fields and directory layout.
5. Approved dataset version is prepared for training input.

## M7 Collection-Relevant Runtime Baseline

- ROS2: Humble
- ROS domain: `211`
- RMW: `rmw_cyclonedds_cpp`
- Fixed robot address: `192.168.8.100`
- Developer container SSH: `ssh developer@192.168.8.100 -p 2222`

Example depth-camera related topics from manual (used for collection-side alignment):

- `/camera/camera/color/image_raw`
- `/camera/camera/color/metadata`
- `/camera/camera/depth/camera_info`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/depth/metadata`
- `/camera/camera/extrinsics/depth_to_color`
- `/tf_static`

Reference figures:

![M7 Wi-Fi location hint](../docs/assets/m7_manual/image5.jpeg)
![M7 realsense viewer example](../docs/assets/m7_manual/image10.png)

## Offline Collection and Local Export (Xingdong Platform)

Collection-side workflow now explicitly includes:

1. XOS App offline task creation
2. VR teleoperation collection
3. Local download of offline package
4. Platform upload of offline package
5. Task-level local JSON export for downstream training use

This workflow depends on teleoperation stack readiness from Xingyao App:

1. XOS teleoperation app license upload and activation
2. XOS teleoperation service startup
3. Meta Quest + Xingyao app network connection and robot binding
4. XR control actions to drive data collection (start/stop and reset origin)

Reference figures:

![offline mode and task list](../docs/assets/data_platform/offline_task_list.png)
![download offline data](../docs/assets/data_platform/download_offline_data_to_local.png)
![export collected data json](../docs/assets/data_platform/offline_export_task_json_download.png)
![xingyao intranet connect](../docs/assets/xingyao_app/xingyao_select_intranet_connect.png)
![xingyao joystick controls](../docs/assets/xingyao_app/xingyao_joystick_controls.jpeg)

## Dataset Directory Standard

Top-level:

- `dataset_info.json`
- `episodes/episode_00001/...`

Per episode:

- `metadata.json`
- `observations/`
- `actions/`

See details in `interfaces/dataset_schema.md`.

## Hugging Face Publishing Convention

- Dataset repo naming: `robotera/<task_or_suite_name>`
- Versioning: semantic dataset tag, for example `v0.1.0`
- Required dataset card sections:
  - Task summary
  - Robot/sensor setup
  - Data fields and units
  - Safety and usage limitations
  - License and attribution

## Risks and Control

- Risk: custom schema drift across tasks.
- Control: lock schema required fields and validate before publishing.
