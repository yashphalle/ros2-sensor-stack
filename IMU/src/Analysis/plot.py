import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from imu_msg.msg import IMUmsg
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import os

bag_path = '/home/yash/Github/EECE5554/LAB3/src/Data/imu_team_rosbags/imu_team_combined'

save_dir = 'plots_team_combined'
os.makedirs(save_dir, exist_ok=True)

storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = SequentialReader()
reader.open(storage_options, converter_options)

timestamps = []
gyro_x, gyro_y, gyro_z = [], [], []
acc_x, acc_y, acc_z = [], [], []
mag_x, mag_y, mag_z = [], [], []
euler_x, euler_y, euler_z = [], [], []  

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/imu':
        msg = deserialize_message(data, IMUmsg)

        timestamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        gyro_x.append(msg.imu.angular_velocity.x)
        gyro_y.append(msg.imu.angular_velocity.y)
        gyro_z.append(msg.imu.angular_velocity.z)

        acc_x.append(msg.imu.linear_acceleration.x)
        acc_y.append(msg.imu.linear_acceleration.y)
        acc_z.append(msg.imu.linear_acceleration.z)

        mag_x.append(msg.mag_field.magnetic_field.x)
        mag_y.append(msg.mag_field.magnetic_field.y)
        mag_z.append(msg.mag_field.magnetic_field.z)

        q = [
            msg.imu.orientation.x,
            msg.imu.orientation.y,
            msg.imu.orientation.z,
            msg.imu.orientation.w
        ]
        r = Rotation.from_quat(q)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        euler_x.append(roll)
        euler_y.append(pitch)
        euler_z.append(yaw)

timestamps = np.array(timestamps)
t = timestamps - timestamps[0]  

gyro_x  = np.array(gyro_x);  gyro_y  = np.array(gyro_y);  gyro_z  = np.array(gyro_z)
acc_x   = np.array(acc_x);   acc_y   = np.array(acc_y);   acc_z   = np.array(acc_z)
mag_x   = np.array(mag_x);   mag_y   = np.array(mag_y);   mag_z   = np.array(mag_z)
euler_x = np.array(euler_x); euler_y = np.array(euler_y); euler_z = np.array(euler_z)

print(f"Total messages: {len(t)}")
print(f"Duration: {t[-1]:.2f} s")



def time_series_plot(t, xs, ys, zs, title, ylabel, labels=('X', 'Y', 'Z'), fname=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, xs, label=labels[0], linewidth=0.8)
    ax.plot(t, ys, label=labels[1], linewidth=0.8)
    ax.plot(t, zs, label=labels[2], linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if fname:
        plt.savefig(f'{save_dir}/{fname}', dpi=150, bbox_inches='tight')
        print(f"Saved {save_dir}/{fname}")
    plt.show()

# a. Gyroscope
time_series_plot(t, gyro_x, gyro_y, gyro_z,
                 'Gyroscope — Angular Velocity combined motion',
                 'Angular Velocity (rad/s)',
                 fname='gyro.png')

# b. Accelerometer
time_series_plot(t, acc_x, acc_y, acc_z,
                 'Accelerometer — Linear Acceleration combined motion',
                 'Acceleration (m/s²)',
                 fname='accelerometer.png')

# c. Magnetometer
time_series_plot(t, mag_x, mag_y, mag_z,
                 'Magnetometer — Magnetic Field combined motion',
                 'Magnetic Field (Gauss)',
                 fname='magnetometer.png')

# d. Orientation (Euler angles)
time_series_plot(t, euler_x, euler_y, euler_z,
                 'Orientation — Euler Angles combined motion',
                 'Angle (degrees)',
                 labels=('Roll (X)', 'Pitch (Y)', 'Yaw (Z)'),
                 fname='euler_orientation.png')

print("\n── Orientation Statistics ──────────────────────────────────────────")
for name, data in [('Roll (X)', euler_x), ('Pitch (Y)', euler_y), ('Yaw (Z)', euler_z)]:
    print(f"  {name}:  mean = {np.mean(data):.4f}°   median = {np.median(data):.4f}°"
          f"   std = {np.std(data):.4f}°")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
hist_data = [
    (euler_x, 'Roll (X)','red'),
    (euler_y, 'Pitch (Y)', 'green'),
    (euler_z, 'Yaw (Z)', 'blue'),
]

for ax, (data, label, color) in zip(axes, hist_data):
    median_val = np.median(data)
    mean_val   = np.mean(data)
    centered   = data - median_val          # distribution around median

    ax.hist(centered, bins=60, color=color, edgecolor='white', linewidth=0.4, density=True)
    ax.axvline(0,            color='red',    linestyle='--', linewidth=1.5, label=f'Median = {median_val:.3f}°')
    ax.axvline(mean_val - median_val, color='black', linestyle=':',  linewidth=1.5, label=f'Mean offset = {mean_val - median_val:.3f}°')
    ax.set_xlabel(f'{label} - Median (degrees)')
    ax.set_ylabel('Density')
    ax.set_title(f'{label} Distribution\n(mean={mean_val:.3f}°, median={median_val:.3f}°, St.Dev ={np.std(data):.3f}°)', fontsize = 8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.suptitle('Orientation Distribution', fontsize=13)
plt.tight_layout()
plt.savefig(f'{save_dir}/orientation_histograms.png', dpi=150, bbox_inches='tight')
print("Plots Done")
plt.show()
