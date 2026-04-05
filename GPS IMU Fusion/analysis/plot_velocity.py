from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from scipy.integrate import cumulative_trapezoid
from imu_msg.msg import IMUmsg
from gps_msgs.msg import GpsMsg
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt


save_dir = 'plots_velocity'
os.makedirs(save_dir, exist_ok=True)

DRIVING_BAG = '/home/yash/Github/EECE5554/LAB4/data/data_driving/'

def read_driving_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    imu_ts, ax, ay = [], [], []
    gps_ts, utm_e, utm_n = [], [], []

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == '/imu':
            msg = deserialize_message(data, IMUmsg)
            imu_ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            ax.append(msg.imu.linear_acceleration.x)
            ay.append(msg.imu.linear_acceleration.y)

        elif topic == '/gps':
            msg = deserialize_message(data, GpsMsg)
            gps_ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            utm_e.append(msg.utm_easting)
            utm_n.append(msg.utm_northing)

    imu_ts = np.array(imu_ts)
    gps_ts = np.array(gps_ts)

    return {
        'imu_t':  imu_ts - imu_ts[0],
        'acc_x':  np.array(ax),
        'acc_y':  np.array(ay),
        'gps_t':  gps_ts - gps_ts[0],
        'utm_e':  np.array(utm_e),
        'utm_n':  np.array(utm_n),
    }

print("Reading driving bag...")
d = read_driving_bag(DRIVING_BAG)

imu_t = d['imu_t']
acc_x = d['acc_x']
acc_y = d['acc_y']
gps_t = d['gps_t']
utm_e = d['utm_e']
utm_n = d['utm_n']

print(f"IMU samples : {len(imu_t)}  duration: {imu_t[-1]:.1f} s")
print(f"GPS samples : {len(gps_t)}  duration: {gps_t[-1]:.1f} s")
print(f"acc_x range: {acc_x.min():.3f} to {acc_x.max():.3f} m/s²")
print(f"acc_y range: {acc_y.min():.3f} to {acc_y.max():.3f} m/s²")


#  GPS Velocity 

# velocity from UTM positions differece
de = np.diff(utm_e)   # delta easting (m)
dn = np.diff(utm_n)   # delta northing (m)
dt = np.diff(gps_t)   # delta time (s)

valid = dt > 0
de, dn, dt = de[valid], dn[valid], dt[valid]
gps_t_mid = ((gps_t[:-1] + gps_t[1:]) / 2)[valid]   # midpoint times

v_gps_e = de / dt        # eastward velocity component
v_gps_n = dn / dt        # northward velocity component
v_gps   = np.sqrt(v_gps_e**2 + v_gps_n**2)   # speed magnitude

# Plot GPS velocity
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(gps_t_mid, v_gps, linewidth=1.0, color='tab:green', label='GPS speed')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Velocity Estimate from GPS')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gps_velocity.png'), dpi=150)
plt.show()
print("Saved gps_velocity.png")

# IMU Velocity 

# stationary carr acc was not zero so caluculating avg bias and substracing it from raw acc_x to make it zero
STATIONARY_WINDOW_SEC = 20
bias_x = np.mean(acc_x[imu_t < STATIONARY_WINDOW_SEC])
print(f"Estimated acc_x bias: {bias_x:.4f} m/s²")

#integrate raw acc_x  velocity -  velocity before adjustments
v_imu_raw = cumulative_trapezoid(acc_x, imu_t, initial=0)

#subtract bias, then integrate  - velocity AFTER adjustment 1
acc_x_corrected = acc_x - bias_x
v_imu_corrected = cumulative_trapezoid(acc_x_corrected, imu_t, initial=0)


# IMU Velocity - Adjustment 2 -  drift reference drawn on raw, subtracted from raw
drift_line = np.linspace(v_imu_raw[0], v_imu_raw[-1], len(v_imu_raw))
v_imu_adjusted = np.clip(v_imu_raw - drift_line, 0, None)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(imu_t, v_imu_raw,
        label='No correction', linewidth=0.8, alpha=0.6, color='tab:blue')
ax.plot(imu_t, v_imu_corrected,
        label=f'Correction 1: bias removed ({bias_x:.4f} m/s²)', linewidth=1.0, color='tab:orange')
ax.plot(imu_t, drift_line,
        label='Drift reference line (start → end)', linewidth=1.2,
        color='red', linestyle='--')
ax.plot(imu_t, v_imu_adjusted,
        label='Correction 2: drift removed + clipped', linewidth=1.2, color='tab:green')
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('IMU Forward Velocity: No Correction vs Correction 1 vs Correction 2')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'imu_velocity_before_after_combined.png'), dpi=150)
plt.show()
print("Saved imu_velocity_before_after_combined.png")

v_gps_interp = np.interp(imu_t, gps_t_mid, v_gps)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(imu_t, v_imu_adjusted,
        label='IMU Correction 2 (drift removed + clipped)', linewidth=1.0, color='tab:orange')
ax.plot(imu_t, v_gps_interp,
        label='GPS speed', linewidth=1.0, color='tab:green', alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('IMU Correction 2 vs GPS Speed')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'imu_correction2_vs_gps.png'), dpi=150)
plt.show()
print("Saved imu_correction2_vs_gps.png")
