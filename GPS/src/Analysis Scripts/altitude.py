import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from gps_msgs.msg import CustomGpsMsg
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np


bag_path = '/home/yash/ros2_ws/src/gps_driver/data/outdoor_walking2'

storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions(
    input_serialization_format='cdr', 
    output_serialization_format='cdr'
)

reader = SequentialReader()
reader.open(storage_options, converter_options)


altitude =[]
timestamps =[]
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/gps':  
        msg = deserialize_message(data, CustomGpsMsg)
        altitude.append(msg.altitude)
        timestamps.append(msg.header.stamp.sec)


timestamps = np.array(timestamps)
altitude = np.array(altitude)



avg_altitude = np.mean(altitude)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(timestamps,altitude, s=30, label='GPS points')
ax.axhline(y=avg_altitude, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_altitude:.4f} m')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Altitude')
ax.set_title('Trajectory (Altitude vs Time)')
ax.legend()

ax.text(0.5, -0.1, f'Total data points: {len(altitude)}', 
        transform=ax.transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/outdoor_walking_altitude.png', dpi=150, bbox_inches='tight')
plt.show()



