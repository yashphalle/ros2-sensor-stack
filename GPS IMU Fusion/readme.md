# LAB4 - GPS + IMU Combined Driver


## Build

From the `GPS IMU Fusion` directory:

```bash
colcon build
source install/setup.bash
```

## Launch both nodes

```bash
ros2 launch gps_driver combined_launch.py
```

This single command starts both the GPS driver node and the IMU driver node.

## Port arguments

If devices are on different ports, override the defaults:

```bash
ros2 launch gps_driver combined_launch.py gps_port:=/dev/ttyUSB0 imu_port:=/dev/ttyUSB1
```

| Argument   | Default        | Description             |
|------------|----------------|-------------------------|
| `gps_port` | `/dev/ttyUSB0` | Serial port for GPS     |
| `imu_port` | `/dev/ttyUSB1` | Serial port for IMU     |
