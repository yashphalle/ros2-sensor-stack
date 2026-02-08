import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from gps_msgs.msg import CustomGpsMsg
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np
import utm

lat = 42 + 20/60 + 18/3600
lon = -(71 + 5/60 + 4/3600)

# 42°20'18"N 71°05'04"W


bag_path = '/home/yash/outdoor_standing'   

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
hdops=[]

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == '/gps':  
        msg = deserialize_message(data, CustomGpsMsg)
        easting.append(msg.utm_easting)
        northing.append(msg.utm_northing)
        timestamps.append(msg.header.stamp.sec)
        hdops.append(msg.hdop)


timestamps = np.array(timestamps) - timestamps[0]
eastings = np.array(easting)
northings = np.array(northing)


eastings_scaled = eastings -eastings[0]
northings_scaled = northings -northings[0]


avg_easting = np.mean(eastings_scaled)
avg_northing = np.mean(northings_scaled)
avg_hdops = np.mean(hdops)


E_ref, N_ref, zone, letter = utm.from_latlon(lat, lon)

errors = np.sqrt((eastings - E_ref)**2 + (northings - N_ref)**2)

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30)
plt.xlabel("Position Error (m)")
plt.ylabel("Count")
plt.title("GPS Position Error Histogram")

mean_err = np.mean(errors)


plt.axvline(mean_err, linestyle='--', label=f"Mean = {mean_err:.2f} m")

plt.text(
    0.5, -0.1,
    f'Avg HDOP: {avg_hdops:.3f}',
    transform=plt.gca().transAxes,
    ha='center',
    va='top'
)


plt.legend()
plt.tight_layout()
plt.savefig("plots/outdoor_standing_error_histogram.png", dpi=150, bbox_inches='tight')
plt.show()


print(f"Mean error : {mean_err:.3f} m")
print(f"Avg Hdops",avg_hdops)




