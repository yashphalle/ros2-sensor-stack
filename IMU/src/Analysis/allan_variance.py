import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from imu_msg.msg import IMUmsg
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import os
import allantools

bag_path = '/home/yash/Downloads/5_hour_imu_data'

os.makedirs('plots allan', exist_ok=True)

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
t0 = None
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/imu':
        msg = deserialize_message(data, IMUmsg)
        if t0 is None:
            t0 = t
        timestamps.append((t - t0) * 1e-9)

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
# t = timestamps - timestamps[0]  
# rate = 1.0/np.mean(np.diff(t))

gyro_x  = np.array(gyro_x);  gyro_y  = np.array(gyro_y);  gyro_z  = np.array(gyro_z)
acc_x   = np.array(acc_x);   acc_y   = np.array(acc_y);   acc_z   = np.array(acc_z)
mag_x   = np.array(mag_x);   mag_y   = np.array(mag_y);   mag_z   = np.array(mag_z)
euler_x = np.array(euler_x); euler_y = np.array(euler_y); euler_z = np.array(euler_z)

# valid = timestamps > 1e9
# timestamps = timestamps[valid]
# gyro_x = gyro_x[valid]; gyro_y = gyro_y[valid]; gyro_z = gyro_z[valid]
# acc_x  = acc_x[valid];  acc_y  = acc_y[valid];  acc_z  = acc_z[valid]
# mag_x  = mag_x[valid];  mag_y  = mag_y[valid];  mag_z  = mag_z[valid]
# euler_x = euler_x[valid]; euler_y = euler_y[valid]; euler_z = euler_z[valid]

t = (timestamps - timestamps[0]) 
dt = np.mean(np.diff(t))
rate = 1.0 / dt

angle_x = np.cumsum(gyro_x) * dt
angle_y = np.cumsum(gyro_y) * dt
angle_z = np.cumsum(gyro_z) * dt

vel_x = np.cumsum(acc_x) * dt
vel_y = np.cumsum(acc_y) * dt
vel_z = np.cumsum(acc_z) * dt




print(f"Total messages: {len(t)}")
print(f"Duration: {t[-1]:.2f} s")
print(f"Sampling Rate: {rate:.2f} Hz")

taus_gyro_x, adev_gyro_x, _, _ = allantools.oadev(angle_x, rate=rate, data_type='phase', taus='all')
print(f"Gyro X: {len(taus_gyro_x)} taus, {len(adev_gyro_x)} adevs")
taus_gyro_y, adev_gyro_y, _, _ = allantools.oadev(angle_y, rate=rate, data_type='phase', taus='all')
print(f"Gyro Y: {len(taus_gyro_y)} taus, {len(adev_gyro_y)} adevs")
taus_gyro_z, adev_gyro_z, _, _ = allantools.oadev(angle_z, rate=rate, data_type='phase', taus='all')
print(f"Gyro Z: {len(taus_gyro_z)} taus, {len(adev_gyro_z)} adevs")


taus_acc_x, adev_acc_x, _, _ = allantools.oadev(vel_x, rate=rate, data_type='phase', taus='all')
print(f"Accel X: {len(taus_acc_x)} taus, {len(adev_acc_x)} adevs")
taus_acc_y, adev_acc_y, _, _ = allantools.oadev(vel_y, rate=rate, data_type='phase', taus='all')
print(f"Accel Y: {len(taus_acc_y)} taus, {len(adev_acc_y)} adevs")
taus_acc_z, adev_acc_z, _, _ = allantools.oadev(vel_z, rate=rate, data_type='phase', taus='all')
print(f"Accel Z: {len(taus_acc_z)} taus, {len(adev_acc_z)} adevs")

def detailed_allan_variance(tau, adev, axis_name, units, save_dir='plots'):
    logtau   = np.log10(tau)
    logadev  = np.log10(adev)
    dlogadev = np.diff(logadev) / np.diff(logtau)

    # Angle Random Walk (slope = -0.5)
    slope = -0.5
    i = np.argmin(np.abs(dlogadev - slope))
    b = logadev[i] - slope * logtau[i]
    N = 10 ** (slope * np.log10(1) + b)
    tauN  = 1.0
    lineN = N / np.sqrt(tau)

    # Rate Random Walk (slope = +0.5) 
    slope = 0.5
    i = np.argmin(np.abs(dlogadev - slope))
    b = logadev[i] - slope * logtau[i]
    K = 10 ** (slope * np.log10(3) + b)
    tauK  = 3.0
    lineK = K * np.sqrt(tau / 3)

    # Bias Instability (slope = 0) 
    slope = 0.0
    i = np.argmin(np.abs(dlogadev - slope))
    b = logadev[i] - slope * logtau[i]
    scfB  = np.sqrt(2 * np.log(2) / np.pi)   
    B     = 10 ** (b - np.log10(scfB))
    tauB  = tau[i]
    lineB = scfB * B * np.ones(len(tau))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(tau, adev,  label=f'σ ({units})')
    ax.loglog(tau, lineN, '--', label=f'σ_N  N={N:.4e}')
    ax.loglog(tau, lineK, '--', label=f'σ_K  K={K:.4e}')
    ax.loglog(tau, lineB, '--', label=f'σ_B  0.664B={scfB*B:.4e}')
    ax.plot(tauN, N,       'o', markersize=8)
    ax.plot(tauK, K,       'o', markersize=8)
    ax.plot(tauB, scfB*B,  'o', markersize=8)
    ax.text(tauN, N,       '  N')
    ax.text(tauK, K,       '  K')
    ax.text(tauB, scfB*B,  '  0.664B')
    ax.set_xlabel('τ (s)')
    ax.set_ylabel(f'Allan Deviation ({units})')
    ax.set_title(f'Allan Deviation — {axis_name}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/allan_{axis_name.replace(" ", "_")}.png', dpi=150)
    plt.show()

    print(f"{axis_name:<12}  N={N:.4e}  K={K:.4e}  B={B:.4e}")
    return N, K, B

print(f"\n{'Axis':<12}  {'N (ARW)':<18} {'K (RRW)':<18} {'B (Bias)':<18}")
print("-" * 66)

N_gx, K_gx, B_gx = detailed_allan_variance(taus_gyro_x, adev_gyro_x, 'Gyro X', 'rad/s')
N_gy, K_gy, B_gy = detailed_allan_variance(taus_gyro_y, adev_gyro_y, 'Gyro Y', 'rad/s')
N_gz, K_gz, B_gz = detailed_allan_variance(taus_gyro_z, adev_gyro_z, 'Gyro Z', 'rad/s')

N_ax, K_ax, B_ax = detailed_allan_variance(taus_acc_x, adev_acc_x, 'Accel X', 'm/s²')
N_ay, K_ay, B_ay = detailed_allan_variance(taus_acc_y, adev_acc_y, 'Accel Y', 'm/s²')
N_az, K_az, B_az = detailed_allan_variance(taus_acc_z, adev_acc_z, 'Accel Z', 'm/s²')

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(taus_gyro_x, adev_gyro_x, label='Gyro X')
ax.loglog(taus_gyro_y, adev_gyro_y, label='Gyro Y')
ax.loglog(taus_gyro_z, adev_gyro_z, label='Gyro Z')
ax.set_xlabel('τ (s)')
ax.set_ylabel('Allan Deviation (rad/s)')
ax.set_title('Gyroscope Allan Deviation')
ax.legend()
ax.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('plots/gyro_allan.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(taus_acc_x, adev_acc_x, label='Acc X')
ax.loglog(taus_acc_y, adev_acc_y, label='Acc Y')
ax.loglog(taus_acc_z, adev_acc_z, label='Acc Z')
ax.set_xlabel('τ (s)')
ax.set_ylabel('Allan Deviation (m/s²)')
ax.set_title('Accelerometer Allan Deviation')
ax.legend()
ax.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('plots/acc_allan.png', dpi=150)
plt.show()

