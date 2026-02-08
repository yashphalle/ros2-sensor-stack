import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from gps_msgs.msg import CustomGpsMsg
from rclpy.serialization import deserialize_message
import numpy as np
import utm
import matplotlib.pyplot as plt

#known start end points
start_lat = 42 + 20/60 + 17/3600
start_lon = -(71 + 5/60 + 6/3600)

end_lat = 42 + 20/60 + 20/3600
end_lon = -(71 + 5/60 + 2/3600)

physical_distance_m = 140.0


bag_path = '/home/yash/ros2_ws/src/gps_driver/data/outdoor_walking2'

storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions(
    input_serialization_format='cdr', 
    output_serialization_format='cdr'
)

reader = SequentialReader()
reader.open(storage_options, converter_options)


eastings = []
northings = []

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/gps':
        msg = deserialize_message(data, CustomGpsMsg)
        eastings.append(msg.utm_easting)
        northings.append(msg.utm_northing)

eastings = np.array(eastings)
northings = np.array(northings)

E_start_ref, N_start_ref, _, _ = utm.from_latlon(start_lat, start_lon)
E_end_ref, N_end_ref, _, _ = utm.from_latlon(end_lat, end_lon)


start_pos_error = np.sqrt((eastings[0] - E_start_ref)**2 + (northings[0] - N_start_ref)**2)
end_pos_error = np.sqrt((eastings[-1] - E_end_ref)**2 + (northings[-1] - N_end_ref)**2)

# Straight line distance between first and last GPs point
straight_line_distance = np.sqrt((eastings[-1] - eastings[0])**2 + (northings[-1] - northings[0])**2)

#Linear regression (best-fit line) 
A = np.vstack([eastings, np.ones(len(eastings))]).T
m, b = np.linalg.lstsq(A, northings, rcond=None)[0]

# Perpendicular distance to regression line 
distances_to_line = np.abs(m * eastings - northings + b) / np.sqrt(m**2 + 1)
avg_pos_error_line = np.mean(distances_to_line)

# Plot GPS start and end points and regression line 
plt.figure(figsize=(10,7))
plt.plot(eastings, northings, 'bo', markersize=4, label='GPS Points')

x_line = np.linspace(eastings.min(), eastings.max(), 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Regression Line')

plt.plot(E_start_ref, N_start_ref, 'gs', markersize=8, label='Reference Start')
plt.plot(E_end_ref, N_end_ref, 'ms', markersize=8, label='Reference End')

plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('Walking Positional Error Plot')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.text(0.02, 0.95, f"Start Pos Error: {start_pos_error:.2f} m", transform=plt.gca().transAxes)
plt.text(0.02, 0.91, f"End Pos Error: {end_pos_error:.2f} m", transform=plt.gca().transAxes)
plt.text(0.02, 0.87, f"Straight Line Distance: {straight_line_distance:.2f} m", transform=plt.gca().transAxes)
plt.text(0.02, 0.83, f"Avg Pos Error to Line: {avg_pos_error_line:.2f} m", transform=plt.gca().transAxes)
plt.text(0.02, 0.79, f"Physical Distance: {physical_distance_m:.2f} m", transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
