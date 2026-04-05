from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from imu_msg.msg import IMUmsg
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'plots_driving'
os.makedirs(save_dir, exist_ok=True)


def read_imu_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    ts, gx, gy, gz, ax, ay, az, mx, my, mz, ex, ey, ez = ([] for _ in range(13))

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/imu':
            msg = deserialize_message(data, IMUmsg)
            ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            gx.append(msg.imu.angular_velocity.x)
            gy.append(msg.imu.angular_velocity.y)
            gz.append(msg.imu.angular_velocity.z)
            ax.append(msg.imu.linear_acceleration.x)
            ay.append(msg.imu.linear_acceleration.y)
            az.append(msg.imu.linear_acceleration.z)
            mx.append(msg.mag_field.magnetic_field.x)
            my.append(msg.mag_field.magnetic_field.y)
            mz.append(msg.mag_field.magnetic_field.z)
            q = [msg.imu.orientation.x, msg.imu.orientation.y,
                 msg.imu.orientation.z, msg.imu.orientation.w]
            r = Rotation.from_quat(q)
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)
            ex.append(roll); ey.append(pitch); ez.append(yaw)

    ts = np.array(ts)
    return {
        't':      ts - ts[0],
        'gyro_x': np.array(gx), 'gyro_y': np.array(gy), 'gyro_z': np.array(gz),
        'acc_x':  np.array(ax), 'acc_y':  np.array(ay), 'acc_z':  np.array(az),
        'mag_x':  np.array(mx), 'mag_y':  np.array(my), 'mag_z':  np.array(mz),
        'euler_x': np.array(ex), 'euler_y': np.array(ey), 'euler_z': np.array(ez),
    }


def fit_ellipse(x, y):
    D = np.column_stack([x**2, x*y, y**2, x, y])
    coeffs, _, _, _ = np.linalg.lstsq(D, np.ones(len(x)), rcond=None)
    return coeffs  # [A, B, C, D, E]

def get_center(coeffs):
    A, B, C, D, E = coeffs
    M = np.array([[2*A, B], [B, 2*C]])
    return np.linalg.solve(M, np.array([-D, -E]))

def get_soft_iron_matrix(coeffs):
    A, B, C, *_ = coeffs
    M = np.array([[A, B/2], [B/2, C]])
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.abs(eigenvalues)
    scale = np.diag( np.sqrt(eigenvalues))
    W = eigenvectors @ scale @ eigenvectors.T
    return W / np.mean(np.diag(W))

def correct(x, y, W, center):
    shifted = np.column_stack([x, y]) - center
    return (W @ shifted.T).T

def butter_lowpass(data, fc, fs, order=2):
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype='low')
    return filtfilt(b, a, data)

def butter_highpass(data, fc, fs, order=2):
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype='high')
    return filtfilt(b, a, data)


#calibration 
cal = read_imu_bag('/home/yash/Github/EECE5554/LAB4/data/data_going_in_circles/')

coeffs = fit_ellipse(cal['mag_x'], cal['mag_y'])
center = get_center(coeffs)
W      = get_soft_iron_matrix(coeffs)

print(f"Hard-iron offset: {center}")
print(f"Soft-iron matrix W:\n{W}")

d = read_imu_bag('/home/yash/Github/EECE5554/LAB4/data/data_driving/')

imu_t   = d['t']
mag_x   = d['mag_x']
mag_y   = d['mag_y']

# raw yaw from magnetometer 
heading = np.arctan2(mag_y, mag_x)
yaw_mag_raw = np.degrees(np.unwrap(heading))

# corrected yaw = hard-iron + soft-iron calibration from circles run
corrected  = correct(mag_x, mag_y, W, center)
yaw_mag_corrected = np.degrees(np.unwrap(np.arctan2(corrected[:, 1], corrected[:, 0])))


# Plot:
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imu_t, yaw_mag_raw,       label='Raw magnetometer yaw',       alpha=0.8, linewidth=0.8)
ax.plot(imu_t, yaw_mag_corrected, label='Corrected magnetometer yaw', alpha=0.8, linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Yaw (degrees)')
ax.set_title('Magnetometer Yaw: Raw vs Calibration-Corrected (data_driving)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mag_yaw_raw_vs_corrected.png'), dpi=150)
plt.show()
print("Saved mag_yaw_raw_vs_corrected.png")
gyro_z = -d['gyro_z']  
gyro_yaw_integrated = cumulative_trapezoid(gyro_z, imu_t, initial=0)
gyro_yaw_deg = np.degrees(gyro_yaw_integrated)
gyro_yaw_deg += yaw_mag_corrected[0] - gyro_yaw_deg[0]


# compare magnetometer yaw vs integrated gyro yaw 
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imu_t, yaw_mag_corrected, label='Magnetometer yaw (corrected)', alpha=0.8, linewidth=0.8)
ax.plot(imu_t, gyro_yaw_deg,      label='Integrated gyro yaw',          alpha=0.8, linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Yaw (degrees)')
ax.set_title('Magnetometer vs Integrated Gyro Yaw (data_driving)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mag_vs_gyro_yaw.png'), dpi=150)
plt.show()
print("Saved mag_vs_gyro_yaw.png")



cutoff_freq = 0.01  
sample_rate = 1.0 / np.mean(np.diff(imu_t))

yaw_mag_smooth   = butter_lowpass(yaw_mag_corrected, cutoff_freq, sample_rate)
yaw_gyro_dynamic = butter_highpass(gyro_yaw_deg,     cutoff_freq, sample_rate)
yaw_fused        = yaw_mag_smooth + yaw_gyro_dynamic

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imu_t, yaw_mag_smooth,   label=f'LP magnetometer (fc={cutoff_freq} Hz)',   alpha=0.8, linewidth=0.8)
ax.plot(imu_t, yaw_gyro_dynamic, label=f'HP integrated gyro (fc={cutoff_freq} Hz)', alpha=0.8, linewidth=0.8)
ax.plot(imu_t, yaw_fused,        label='Complementary filter',                      alpha=0.9, linewidth=1.2, color='green')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Yaw (degrees)')
ax.set_title(f'Complementary Filter Components (fc={cutoff_freq} Hz)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complementary_filter.png'), dpi=150)
plt.show()
print("Saved complementary_filter.png")


# All combined
euler_z = d['euler_z']
imu_yaw = -np.degrees(np.unwrap(np.radians(euler_z)))  
imu_yaw += yaw_mag_corrected[0] - imu_yaw[0]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(imu_t, yaw_mag_corrected, label='Magnetometer yaw (corrected)',  alpha=0.6, linewidth=0.7)
ax.plot(imu_t, gyro_yaw_deg,      label='Integrated gyro yaw',           alpha=0.6, linewidth=0.7)
ax.plot(imu_t, yaw_fused,          label='Complementary filter',          alpha=0.9, linewidth=1.2, color='green')
ax.plot(imu_t, imu_yaw,           label='IMU onboard yaw (euler_z)',     alpha=0.8, linewidth=1.0, color='red')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Yaw (degrees)')
ax.set_title('Yaw Comparison: Magnetometer / Gyro / Complementary Filter / IMU (data_driving)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_yaw_comparison.png'), dpi=150)
plt.show()
print("Saved all_yaw_comparison.png")
