# Robotera VLA

[![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange.svg)](https://ubuntu.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.4-green.svg)](https://developer.nvidia.com/cuda-zone)

Robotera VLA provides the `release_1.0` baseline for Robotera M7 data collection, training, and inference workflows. The repository focuses on the public-facing pieces around that stack: collection instructions, dataset and interface contracts, training baseline code, and an inference deployment example.

The repository assumes existing Robotera teleoperation and recorder software on the robot side. Those runtime services are not implemented here.

## M7 Baseline

This project currently targets Robotera M7 as the default robot baseline.

![M7 robot overview](docs/assets/m7_manual/image1.jpg)

| Item | Value |
|---|---|
| Model | M7 |
| DOF | 43 |
| Battery | 57.6V 15Ah 864Wh |
| Compute | 80 TOPS (x86) + 275 TOPS (Orin AGX) |
| Interfaces | Ethernet / USB 4.0 / Wi-Fi 6 |
| ROS2 Baseline | Humble, `ROS_DOMAIN_ID=211`, `rmw_cyclonedds_cpp` |

Detailed hardware and runtime constraints are documented in `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`.

## Repository Layout

- `data_collection/`: offline acquisition instructions, operation manuals, dataset schema, and lightweight validation examples
- `training/`: training baseline, configs, model wrappers, and example fine-tuning workflow
- `inference/`: inference deployment example, ROS 2 integration code, and interface definitions
- `docs/`: retained project-level hardware and software baseline

## Start Here

1. Review `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`.
2. Follow `data_collection/README.md` for offline data acquisition and export.
3. Review `data_collection/interfaces/dataset_schema.md` for dataset layout and field requirements.
4. Review `training/README.md` for the current training baseline and example checkpoint references.
5. Review `inference/README.md` for inference deployment and ROS 2 integration.
6. Review `inference/interfaces/ros2_api_contract.md` for observation and action message contracts.

## Current Surface

- Data collection is documented around the existing Robotera XOS and Meta Quest teleoperation workflow.
- Training includes the current M7-oriented baseline and related configs.
- Inference includes a Docker-based example and Robotera interface definitions for integration on an inference PC.

## License

This repository is released under the MIT License. See `LICENSE`.
