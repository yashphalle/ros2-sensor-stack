from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from imu_msg.msg import IMUmsg
from gps_msgs.msg import GpsMsg
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'plots_dead_reckoning'
os.makedirs(save_dir, exist_ok=True)

CAL_BAG     = '/home/yash/Github/EECE5554/LAB4/data/data_going_in_circles/'
DRIVING_BAG = '/home/yash/Github/EECE5554/LAB4/data/data_driving/'



def fit_ellipse(x, y):
    D = np.column_stack([x**2, x*y, y**2, x, y])
    coeffs, _, _, _ = np.linalg.lstsq(D, np.ones(len(x)), rcond=None)
    return coeffs

def get_center(coeffs):
    A, B, C, D, E = coeffs
    M = np.array([[2*A, B], [B, 2*C]])
    return np.linalg.solve(M, np.array([-D, -E]))

def get_soft_iron_matrix(coeffs):
    A, B, C, *_ = coeffs
    M = np.array([[A, B/2], [B/2, C]])
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.abs(eigenvalues)
    scale = np.diag(np.sqrt(eigenvalues))
    W = eigenvectors @ scale @ eigenvectors.T
    return W / np.mean(np.diag(W))

def correct_mag(x, y, W, center):
    shifted = np.column_stack([x, y]) - center
    return (W @ shifted.T).T

def butter_lowpass(data, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2.0), btype='low')
    return filtfilt(b, a, data)

def butter_highpass(data, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2.0), btype='high')
    return filtfilt(b, a, data)


def read_imu_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    ts, gx, gy, gz, ax, ay, mx, my = ([] for _ in range(8))

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
            mx.append(msg.mag_field.magnetic_field.x)
            my.append(msg.mag_field.magnetic_field.y)

    ts = np.array(ts)
    return {
        't':      ts - ts[0],
        'gyro_z': np.array(gz),
        'acc_x':  np.array(ax),
        'acc_y':  np.array(ay),
        'mag_x':  np.array(mx),
        'mag_y':  np.array(my),
    }


def read_driving_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    imu_ts, gx, gy, gz, ax, ay, mx, my = ([] for _ in range(8))
    gps_ts, utm_e, utm_n = [], [], []

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/imu':
            msg = deserialize_message(data, IMUmsg)
            imu_ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            gx.append(msg.imu.angular_velocity.x)
            gy.append(msg.imu.angular_velocity.y)
            gz.append(msg.imu.angular_velocity.z)
            ax.append(msg.imu.linear_acceleration.x)
            ay.append(msg.imu.linear_acceleration.y)
            mx.append(msg.mag_field.magnetic_field.x)
            my.append(msg.mag_field.magnetic_field.y)
        elif topic == '/gps':
            msg = deserialize_message(data, GpsMsg)
            gps_ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            utm_e.append(msg.utm_easting)
            utm_n.append(msg.utm_northing)

    imu_ts = np.array(imu_ts)
    gps_ts = np.array(gps_ts)

    return {
        'imu_t':  imu_ts - imu_ts[0],
        'gyro_z': np.array(gz),
        'acc_x':  np.array(ax),
        'acc_y':  np.array(ay),
        'mag_x':  np.array(mx),
        'mag_y':  np.array(my),
        'gps_t':  gps_ts - gps_ts[0],
        'utm_e':  np.array(utm_e),
        'utm_n':  np.array(utm_n),
    }



print("Reading calibration bag (circles)...")
cal = read_imu_bag(CAL_BAG)

coeffs = fit_ellipse(cal['mag_x'], cal['mag_y'])
center = get_center(coeffs)
W      = get_soft_iron_matrix(coeffs)
print(f"Hard-iron offset: {center}")
print(f"Soft-iron matrix W:\n{W}")


print("Reading driving bag...")
d = read_driving_bag(DRIVING_BAG)

imu_t  = d['imu_t']
acc_x  = d['acc_x']
acc_y  = d['acc_y']
gyro_z = d['gyro_z']
mag_x  = d['mag_x']
mag_y  = d['mag_y']
gps_t  = d['gps_t']
utm_e  = d['utm_e']
utm_n  = d['utm_n']

print(f"IMU samples: {len(imu_t)}  duration: {imu_t[-1]:.1f} s")
print(f"GPS samples: {len(gps_t)}  duration: {gps_t[-1]:.1f} s")


de = np.diff(utm_e)
dn = np.diff(utm_n)
dt_gps = np.diff(gps_t)

valid = dt_gps > 0
de, dn, dt_gps = de[valid], dn[valid], dt_gps[valid]
gps_t_mid = ((gps_t[:-1] + gps_t[1:]) / 2)[valid]  

v_gps = np.sqrt((de / dt_gps)**2 + (dn / dt_gps)**2)

ds_gps     = np.sqrt(de**2 + dn**2)
gps_disp   = np.concatenate([[0], np.cumsum(ds_gps)])
gps_t_disp = np.concatenate([[gps_t[0]], gps_t_mid])


STATIONARY_WINDOW_SEC = 25
bias_x = np.mean(acc_x[imu_t < STATIONARY_WINDOW_SEC])
print(f"acc_x bias (stationary): {bias_x:.4f} m/s²")

# Before adjustment: integrate raw
v_imu_raw = cumulative_trapezoid(acc_x, imu_t, initial=0)

# adjustment  1 -  bias removed
acc_x_corrected = acc_x - bias_x
v_imu_corrected = cumulative_trapezoid(acc_x_corrected, imu_t, initial=0)

# Adjustment 2 -  linear drift removed + negatives clipped
drift_line = np.linspace(v_imu_raw[0], v_imu_raw[-1], len(v_imu_raw))
v_fwd = np.clip(v_imu_raw - drift_line, 0, None)
v_unclipped = v_imu_raw - drift_line

# IMU displacement (

imu_disp_unscaled = cumulative_trapezoid(v_fwd, imu_t, initial=0)
gps_disp_imu      = np.interp(imu_t, gps_t_disp, gps_disp)

# Plot before scaling
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(imu_t, imu_disp_unscaled, label='IMU displacement (before scaling)')
ax.plot(imu_t, gps_disp_imu,      label='GPS displacement')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.set_title('Forward Displacement: IMU vs GPS (Before Scaling)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'displacement_imu_vs_gps_unscaled.png'), dpi=150)
plt.show()
print("Saved displacement_imu_vs_gps_unscaled.png")

# Scaling factor: GPS total distance / IMU total distance
scale_factor = gps_disp[-1] / imu_disp_unscaled[-1]
print(f"Scaling factor (GPS/IMU): {scale_factor:.3f}")
v_fwd = v_fwd * scale_factor

# Recompute displacement with scaled velocity
imu_disp = cumulative_trapezoid(v_fwd, imu_t, initial=0)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(imu_t, imu_disp,     label=f'IMU displacement (scaled by {scale_factor:.3f})')
ax.plot(imu_t, gps_disp_imu, label='GPS displacement')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.set_title('Forward Displacement: IMU vs GPS (After Scaling)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'displacement_imu_vs_gps.png'), dpi=150)
plt.show()
print("Saved displacement_imu_vs_gps.png")


# heading: complementary filter (mag + gyro) 

fs = 1.0 / np.mean(np.diff(imu_t))

# Calibrated magnetometer heading
corrected_mag  = correct_mag(mag_x, mag_y, W, center)
yaw_mag_rad    = np.unwrap(np.arctan2(corrected_mag[:, 1], corrected_mag[:, 0]))
yaw_mag_deg    = np.degrees(yaw_mag_rad)

# Integrated gyro yaw 
gyro_yaw_rad = cumulative_trapezoid(-gyro_z, imu_t, initial=0)
gyro_yaw_deg = np.degrees(gyro_yaw_rad)
gyro_yaw_deg += yaw_mag_deg[0] - gyro_yaw_deg[0]

# Complementary filter
CUTOFF_HZ    = 0.01
yaw_mag_lp   = butter_lowpass(yaw_mag_deg, CUTOFF_HZ, fs)
yaw_gyro_hp  = butter_highpass(gyro_yaw_deg, CUTOFF_HZ, fs)
yaw_fused_deg = yaw_mag_lp + yaw_gyro_hp
yaw_fused_rad = np.radians(yaw_fused_deg)


# dead reckoning: rotate forward velocity to East/North 

ve = v_fwd * np.cos(yaw_fused_rad)
vn = v_fwd * np.sin(yaw_fused_rad)

xe = cumulative_trapezoid(ve, imu_t, initial=0)
xn = cumulative_trapezoid(vn, imu_t, initial=0)


# align IMU trajectory with GPS 

# Translation: start both tracks at the same GPS origin
xe_aligned = xe + utm_e[0]
xn_aligned = xn + utm_n[0]

# Heading alignment: rotate IMU track so its initial direction matches GPS
# GPS initial heading from first valid displacement segment
gps_heading_init = np.arctan2(dn[0], de[0])

# IMU initial heading from first non-zero velocity sample
imu_heading_init = yaw_fused_rad[np.argmax(v_fwd > 0.5)]

dtheta = gps_heading_init - imu_heading_init
print(f"Heading alignment offset: {np.degrees(dtheta):.2f} deg")

cos_dt, sin_dt = np.cos(dtheta), np.sin(dtheta)
xe_rot = cos_dt * (xe_aligned - utm_e[0]) - sin_dt * (xn_aligned - utm_n[0]) + utm_e[0]
xn_rot = sin_dt * (xe_aligned - utm_e[0]) + cos_dt * (xn_aligned - utm_n[0]) + utm_n[0]


# trajectory plot: IMU dead reckoning vs GPS

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(utm_e - utm_e[0], utm_n - utm_n[0],
        label='GPS track', linewidth=1.2, color='tab:green')
ax.plot(xe_rot - utm_e[0], xn_rot - utm_n[0],
        label='IMU dead reckoning', linewidth=1.0, color='tab:orange', alpha=0.85)
ax.plot(0, 0, 'ko', markersize=6, label='Start')
ax.set_xlabel('Easting relative to start (m)')
ax.set_ylabel('Northing relative to start (m)')
ax.set_title('Dead Reckoning Trajectory vs GPS Track')
ax.legend()
ax.axis('equal')
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'trajectory_imu_vs_gps.png'), dpi=150)
plt.show()
print("Saved trajectory_imu_vs_gps.png")

scale_factor = v_gps.mean() / v_fwd[v_fwd > 0.1].mean()
print(f"Scaling factor (GPS mean / IMU mean): {scale_factor:.3f}")


# omega * X_dot and compare to acc_y
omega_x_dot = gyro_z * v_unclipped   # ω·Ẋ

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imu_t, omega_x_dot, label='ω·Ẋ (gyro_z × v_fwd)', linewidth=0.8)
ax.plot(imu_t, acc_y,       label='ÿ_obs (acc_y)',          linewidth=0.8, alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration (m/s²)')
ax.set_title('ω·Ẋ vs observed lateral acceleration (ÿ_obs)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'omega_xdot_vs_acc_y.png'), dpi=150)
plt.show()