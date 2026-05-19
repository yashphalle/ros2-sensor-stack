# ros2-sensor-stack

A collection of ROS2 sensor driver packages along with sensor-fusion and computer-vision experiments.

## Packages

| Folder | Description |
|---|---|
| [`GPS/`](./GPS) | ROS2 driver for a single GPS receiver. Publishes NMEA-derived fixes on a custom message. Includes analysis scripts and a written report. |
| [`RTK-GPS/`](./RTK-GPS) | ROS2 driver for an RTK-capable GPS receiver, with separate `gps_driver` / `rtk_driver` packages and matching custom message definitions. |
| [`IMU/`](./IMU) | ROS2 driver for the VectorNav VN-100 IMU. Parses `$VNYMR` strings into orientation, angular velocity, linear acceleration, and magnetic field readings. Includes Allan-variance analysis and report. |
| [`GPS IMU Fusion/`](./GPS%20IMU%20Fusion) | Combined GPS + IMU launch and analysis. A single `ros2 launch` brings up both drivers; analysis covers dead reckoning and sensor fusion. |
| [`Camera Calibration/`](./Camera%20Calibration) | Intrinsic camera calibration (`calibration.py`) and panoramic image stitching (`stich.py`) with sample scenes and calibration images. |

## Build (per-package)

Each driver folder is a self-contained ROS2 workspace. From inside a driver folder:

```bash
colcon build
source install/setup.bash
```

Then launch the relevant node — see each package's `launch/` directory or the per-folder readme (e.g. [`GPS IMU Fusion/readme.md`](./GPS%20IMU%20Fusion/readme.md)).
