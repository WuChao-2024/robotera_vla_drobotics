# Training Data Contract

## 1. Purpose

Define how Robotera custom dataset fields map into training loader inputs.

## 2. Mapping Table (v0.1)

| Robotera Source | Loader Field | Type | Notes |
|---|---|---|---|
| `metadata.instruction` | `task.text` | string | Natural language instruction |
| `observations/camera_front/*.jpg` | `obs.image.front` | image sequence | Timestamp-aligned |
| `observations/joint_state.csv` | `obs.proprio.joint` | float array | Unit declared in schema |
| `observations/ee_state.csv` | `obs.proprio.ee_pose` | float array | Position in meters |
| `actions/action.csv` | `action.chunk` | float array | Action dim from dataset_info |
| `metadata.success` | `episode.label.success` | bool | For optional filtering |

## 3. Required Normalization Metadata

- `action_normalization.type`
- `action_normalization.stats`
- `proprio_normalization.type`
- `proprio_normalization.stats`

## 4. Adapter Interface Placeholders

Future adapters (interface-only in stage-1):

- `RoboteraToRLDSAdapter`
- `RoboteraToLeRobotAdapter`

Expected interface behavior:

1. Validate required fields before conversion.
2. Emit deterministic ordering and timestamps.
3. Save conversion metadata for traceability.

## 5. Validation Rules

- Missing required fields must fail fast.
- Timestamp non-monotonic episodes should be rejected by default.
