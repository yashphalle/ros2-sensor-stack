import serial
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header, Char
from rtk_msgs.msg import Customrtk
import utm
import argparse

class pubNode(Node):
    def __init__(self,port):
        super().__init__('publisher_node')
        self.publisher_node = self.create_publisher(Customrtk, 'rtk_gps', 10)
        timer_period = 0.05                          
        self.timer = self.create_timer(timer_period, self.pub_callback)
        self.port = serial.Serial(port, baudrate=4800, timeout=1)

    def extract_data(self,raw_data):
        data = raw_data.split(',')
        return data


    def pub_callback(self):
        msg = Customrtk()
        raw_data = self.port.readline().decode('utf-8').strip()
        print(raw_data)
        if raw_data.startswith('$GNGGA'):
           result = self.extract_data(raw_data)
           #$GPGGA,184211.000,4158.3847,N,08754.0098,W,1,07,1.1,185.2,M,-34.1,M,,0000*6C
           #conversions
           #1) Time
           utc_raw = result[1]
           hours = int(utc_raw[0:2])      
           minutes = int(utc_raw[2:4])    
           seconds = float(utc_raw[4:])
           total_seconds = hours * 3600 + minutes * 60 + seconds

           #2) Lat Long
           lat_raw = result[2]
           lat = int(lat_raw[:2]) + float(lat_raw[2:]) / 60.0

           lon_raw = result[4]
           lon = int(lon_raw[:3]) + float(lon_raw[3:]) / 60.0

           if result[3] == 'S': lat = -lat
           if result[5] == 'W': lon = -lon

           #3) Utm
           print("Lattitude", lat)
           print("Longitute", lon)
           easting, northing, zone, letter = utm.from_latlon(lat, lon)

           fix_quality = int(result[6])
           #ROS Message
           
           msg.header.frame_id = "GPS1_FRAME"
           msg.header.stamp.sec = int(total_seconds)
           msg.header.stamp.nanosec = int((total_seconds % 1) * 1e9) 

           msg.latitude = lat
           msg.longitude = lon

           msg.altitude = float(result[9])
           msg.utc = float(result[1])
           msg.hdop = float(result[8])
           msg.utm_easting = easting
           msg.utm_northing = northing
           msg.zone = int(zone)
           msg.letter = letter
           msg.fix_quality = fix_quality
           self.publisher_node.publish(msg)
           self.get_logger().info('Publishing driver 1: "%s"' % msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=str)
    parsed_args, unknown = parser.parse_known_args() 
    
    rclpy.init(args=args)
    publisher = pubNode(parsed_args.port)
    rclpy.spin(publisher)
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()