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


easting =[]
northing =[]
timestamps =[]
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/gps':  
        msg = deserialize_message(data, CustomGpsMsg)
        easting.append(msg.utm_easting)
        northing.append(msg.utm_northing)
        timestamps.append(msg.header.stamp.sec)


timestamps = np.array(timestamps) - timestamps[0]
eastings = np.array(easting)
northings = np.array(northing)


eastings_scaled = eastings -eastings[0]
northings_scaled = northings -northings[0]


avg_easting = np.mean(eastings_scaled)
avg_northing = np.mean(northings_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(eastings_scaled, northings_scaled, s=30, label='GPS points')
ax.scatter(avg_easting, avg_northing, s=100, c='red', marker='x', linewidths=2, label=f'Avg ({avg_easting:.4f}, {avg_northing:.4f})')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_title('Trajectory (Northing vs Easting)')
ax.legend()

ax.text(0.5, -0.1, f'Total data points: {len(eastings_scaled)}', 
        transform=ax.transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/outdoor_walking_easting_northing.png', dpi=150, bbox_inches='tight')
plt.show()



