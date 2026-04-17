# Robotera Dataset Schema (v1.0)

> Based on LeRobot v2.0 format. 

## 1. Directory Layout

```text
<dataset_root>/
  meta/
    info.json
    episodes.jsonl
    task.jsonl
    joint_indexes.json        # recommended
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
    chunk-001/
      ...
  videos/
    chunk-000/
      observation.images.cam_high/
        episode_000000.mp4
        episode_000001.mp4
      observation.images.cam_left/
        episode_000000.mp4
        ...
      observation.images.cam_right/
        episode_000000.mp4
        ...
    chunk-001/
      ...
```

### Naming Conventions

- Chunk directories: `chunk-{chunk_index:03d}` (zero-padded 3 digits)
- Episode data files: `episode_{episode_index:06d}.parquet` (zero-padded 6 digits)
- Episode video files: `episode_{episode_index:06d}.mp4`
- Video subdirectories: named by feature key (e.g. `observation.images.cam_high`)

## 2. Required Files

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/task.jsonl`
- `data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet` (at least one)
- `videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4` (per video feature)

### Recommended Files

- `meta/joint_indexes.json` — maps action/state dimension indices to joint names

## 3. `meta/info.json` Required Fields

| Field | Type | Description |
|---|---|---|
| `codebase_version` | string | Format version, e.g. `"v2.0"` |
| `robot_type` | string | Robot model identifier, e.g. `"m7"` |
| `fps` | integer | Data capture frequency in Hz |
| `chunks_size` | integer | Max episodes per chunk |
| `total_chunks` | integer | Number of chunk directories |
| `total_episodes` | integer | Total episode count |
| `total_frames` | integer | Total frame count across all episodes |
| `total_tasks` | integer | Number of distinct tasks |
| `total_videos` | integer | Total video file count |
| `data_path` | string | Path template, e.g. `"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"` |
| `video_path` | string | Path template, e.g. `"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"` |
| `features` | object | Feature definitions (see §3.1) |
| `splits` | object | Train/val/test split ranges, e.g. `{"train": "0:20"}` |

### 3.1 `features` Object

Each key in `features` describes a data column or video stream. Required feature keys for M7:

| Feature Key | dtype | shape | Description |
|---|---|---|---|
| `action` | `float32` | `[38]` | Robot action commands (38 DOF) |
| `observation.state` | `float32` | `[57]` | Robot state readings (57 DOF) |
| `observation.images.cam_high` | `video` | `[480, 848, 3]` | Front high camera |
| `observation.images.cam_left` | `video` | `[480, 640, 3]` | Left camera |
| `observation.images.cam_right` | `video` | `[480, 640, 3]` | Right camera |
| `episode_index` | `int64` | `[1]` | Episode identifier |
| `frame_index` | `int64` | `[1]` | Frame index within episode |
| `index` | `int64` | `[1]` | Global frame index |
| `task_index` | `int64` | `[1]` | Task identifier |
| `timestamp` | `float32` | `[1]` | Frame timestamp |

Video features must include `video_info`:

```json
{
  "video.codec": "h264",
  "video.fps": 30,
  "video.pix_fmt": "yuv420p",
  "video.is_depth_map": false,
  "has_audio": false
}
```

## 4. `meta/episodes.jsonl` Format

One JSON object per line, one line per episode:

```json
{"episode_index": 0, "length": 621, "tasks": ["capture test"]}
{"episode_index": 1, "length": 451, "tasks": ["capture test"]}
```

| Field | Type | Description |
|---|---|---|
| `episode_index` | integer | Zero-based episode ID |
| `length` | integer | Number of frames in this episode |
| `tasks` | array of strings | Task descriptions for this episode |

## 5. `meta/task.jsonl` Format

One JSON object per line, one line per unique task:

```json
{"task": "capture test", "task_index": 0}
```

| Field | Type | Description |
|---|---|---|
| `task` | string | Human-readable task description |
| `task_index` | integer | Zero-based task ID |

## 6. `meta/joint_indexes.json` Format (Recommended)

Maps dimension index → joint name for `action` and `observation.state` features:

```json
{
  "action": {
    "0": "right_end_x",
    "1": "right_end_y",
    ...
  },
  "observation.state": {
    "0": "right_shoulder_pitch_joint",
    "1": "right_shoulder_roll_joint",
    ...
  }
}
```

### M7 Action Space (38 DOF)

| Index | Joint Name | Group |
|---|---|---|
| 0–2 | `right_end_{x,y,z}` | Right end-effector position |
| 3–6 | `right_end_{qx,qy,qz,qw}` | Right end-effector quaternion |
| 7–9 | `right_hand_thumb_{bend,rota1,rota2}_joint` | Right thumb |
| 10–18 | Right hand finger joints | Right fingers (index/mid/ring/pinky) |
| 19–21 | `left_end_{x,y,z}` | Left end-effector position |
| 22–25 | `left_end_{qx,qy,qz,qw}` | Left end-effector quaternion |
| 26–28 | `left_hand_thumb_{bend,rota1,rota2}_joint` | Left thumb |
| 29–37 | Left hand finger joints | Left fingers (index/mid/ring/pinky) |

### M7 Observation State (57 DOF)

| Index | Joint Name | Group |
|---|---|---|
| 0–6 | Right arm joints | Right shoulder/arm/elbow/wrist |
| 7–13 | Left arm joints | Left shoulder/arm/elbow/wrist |
| 14–20 | `right_end_{x,y,z,qx,qy,qz,qw}` | Right end-effector pose |
| 21–27 | `left_end_{x,y,z,qx,qy,qz,qw}` | Left end-effector pose |
| 28–39 | Right hand joints | Right hand fingers |
| 40–51 | Left hand joints | Left hand fingers |
| 52–54 | `waist_{roll,pitch,yaw}_joint` | Torso |
| 55–56 | `neck_{yaw,pitch}_joint` | Head |

## 7. Parquet Data File

Each `episode_{NNNNNN}.parquet` contains one row per frame. Columns correspond to non-video feature keys in `info.json`:

- `action` — float32 array of length `action_dim`
- `observation.state` — float32 array of length `state_dim`
- `episode_index` — int64
- `frame_index` — int64
- `index` — int64
- `task_index` — int64
- `timestamp` — float32

## 8. Video Files

- Codec: H.264
- Pixel format: YUV420p → decoded to RGB
- Frame rate: synchronized with `fps` in `info.json`
- No audio track
- Resolution per camera defined in `features[camera_key].shape`

## 9. Versioning Rule

- This schema follows semantic versioning.
- Backward-incompatible field changes require major version bump.
- Current version: **1.0.0**
