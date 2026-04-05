import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from imu_msg.msg import IMUmsg
from gps_msgs.msg import GpsMsg
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import os

bag_path = '/home/yash/Github/EECE5554/LAB4/data/data_going_in_circles/'

save_dir = 'plots_circular'
os.makedirs(save_dir, exist_ok=True)

storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = SequentialReader()
reader.open(storage_options, converter_options)

imu_timestamps = []
gps_timestamps = []
gyro_x, gyro_y, gyro_z = [], [], []
acc_x, acc_y, acc_z = [], [], []
mag_x, mag_y, mag_z = [], [], []
euler_x, euler_y, euler_z = [], [], []  
altitude = []
latitude = []
longitude = []
altitude = []
hdop = []
utm_easting = []
utm_northing = []



while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/imu':
        msg = deserialize_message(data, IMUmsg)

        imu_timestamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

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
    if topic == '/gps':
        msg = deserialize_message(data, GpsMsg)
        altitude.append(msg.altitude)
        gps_timestamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        latitude.append(msg.latitude)
        longitude.append(msg.longitude)
        hdop.append(msg.hdop)
        utm_easting.append(msg.utm_easting)
        utm_northing.append(msg.utm_northing)



imu_timestamps = np.array(imu_timestamps)
imu_t = imu_timestamps - imu_timestamps[0]

gyro_x  = np.array(gyro_x);  gyro_y  = np.array(gyro_y);  gyro_z  = np.array(gyro_z)
acc_x   = np.array(acc_x);   acc_y   = np.array(acc_y);   acc_z   = np.array(acc_z)
mag_x   = np.array(mag_x);   mag_y   = np.array(mag_y);   mag_z   = np.array(mag_z)
euler_x = np.array(euler_x); euler_y = np.array(euler_y); euler_z = np.array(euler_z)

print(f"Total messages: {len(imu_t)}")
print(f"Duration: {imu_t[-1]:.2f} s")

gps_timestamps = np.array(gps_timestamps)
gps_t = gps_timestamps - gps_timestamps[0]


def fit_ellipse(x, y):
    D = np.column_stack([x**2, x*y, y**2, x, y])
    ones = np.ones(len(x))
    coeffs, _, _, _ = np.linalg.lstsq(D, ones, rcond=None)
    return coeffs  # [A, B, C, D, E]

def get_center(coeffs):
    A, B, C, D, E = coeffs
    M = np.array([[2*A, B],
                  [B,  2*C]])
    rhs = np.array([-D, -E])
    center = np.linalg.solve(M, rhs)
    return center  # hard iron offset [x0, y0]

def get_soft_iron_matrix(coeffs, center):
    A, B, C, D, E = coeffs
    x0, y0 = center
    
    # Ellipse shape matrix
    M = np.array([[A,    B/2],
                  [B/2,  C  ]])
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    eigenvalues = np.abs(eigenvalues)   
    scale = np.diag(np.sqrt(eigenvalues))
    W = eigenvectors @ scale @ eigenvectors.T
    
    W = W / np.mean(np.diag(W))
    
    return W

def correct(x, y, W, center):
    raw = np.column_stack([x, y])
    shifted = raw - center          # remove hard iron
    corrected = (W @ shifted.T).T   # remove soft iron
    return corrected


def get_heading(corrected):
    heading = np.degrees(np.arctan2(corrected[:, 1], 
                                    corrected[:, 0]))
    return heading

coeffs = fit_ellipse(mag_x, mag_y)
center = get_center(coeffs)
W = get_soft_iron_matrix(coeffs, center)
corrected = correct(mag_x, mag_y, W, center)
heading = get_heading(corrected)


radii = np.sqrt(mag_x**2 + mag_y**2)
radii_corrected = np.sqrt(corrected[:, 0]**2 + corrected[:, 1]**2)
print(f"Radius mean: {radii.mean():.4f}, std: {radii.std():.4f}")
print(f"Radius mean (corrected): {radii_corrected.mean():.4f}, std: {radii_corrected.std():.4f}")

print(f"Hard iron offset (center): {center}")
print(f"Soft iron matrix W:\n{W}")



#without calibration
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
axes.scatter(mag_x, mag_y, s=1, alpha=0.3)
axes.set_title('XY plane'); axes.set_xlabel('Mx'); axes.set_ylabel('My')
plt.suptitle('Magnetometer Raw Data')
plt.tight_layout()
plt.text(0.5, 0.01, f"Radius mean: {radii.mean():.4f}, std: {radii.std():.4f}",
         ha='center', va='bottom', transform=axes.transAxes, fontsize=10)
plt.plot()
plt.savefig(os.path.join(save_dir, 'magnetometer_raw_xy.png'))
plt.show()


#after calibration
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
axes.scatter(corrected[:, 0], corrected[:, 1], s=1, alpha=0.3)
axes.set_title('XY plane'); axes.set_xlabel('Mx'); axes.set_ylabel('My')
plt.suptitle('Magnetometer Calibrated')
plt.tight_layout()
plt.text(0.5, 0.01, f"Radius mean: {radii_corrected.mean():.4f}, std: {radii_corrected.std():.4f}",
         ha='center', va='bottom', transform=axes.transAxes, fontsize=10)
plt.savefig(os.path.join(save_dir, 'magnetometer_calibrated_xy.png'))
plt.show()


