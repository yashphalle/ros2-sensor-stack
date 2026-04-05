import serial
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header, Char
from imu_msg.msg import IMUmsg
import utm
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

class pubNode(Node):
    def __init__(self,port):
        super().__init__('publisher_node')
        self.publisher_node = self.create_publisher(IMUmsg, 'imu', 10)
        timer_period = 0.005                          
        self.timer = self.create_timer(timer_period, self.pub_callback)
        self.port = serial.Serial(port, baudrate=115200, timeout=1)
        command = "$VNWRG,07,40*59\r\n"
        self.port.write(command.encode())
    


    def extract_data(self,raw_data):
        data = raw_data.split(',')
        return data


    def pub_callback(self):
        msg = IMUmsg()
        raw_data = self.port.readline().decode('utf-8', errors='ignore').strip()
        print(raw_data)
        if raw_data.startswith('$VNYMR'):
            try:
                result = self.extract_data(raw_data)
                print(result)
                yaw = float(result[1])
                pitch = float(result[2])
                roll = float(result[3])
                mag_x = float(result[4])    
                mag_y = float(result[5])
                mag_z = float(result[6])
                acc_x = float(result[7])
                acc_y = float(result[8])
                acc_z = float(result[9])
                angular_rate_x = float(result[10])
                angular_rate_y = float(result[11])
                angular_rate_z = float(result[12].split('*')[0]) 

                r = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)
                q = r.as_quat()  

                # self.get_logger().info(f"Quaternion (ZYX): {q}")

                #ROS Message
                msg.header.frame_id = "IMU1_FRAME"
                msg.header.stamp = self.get_clock().now().to_msg()

                msg.imu.orientation.x = q[0]
                msg.imu.orientation.y = q[1]
                msg.imu.orientation.z = q[2]
                msg.imu.orientation.w = q[3]

                msg.imu.angular_velocity.x = angular_rate_x
                msg.imu.angular_velocity.y = angular_rate_y
                msg.imu.angular_velocity.z = angular_rate_z

                msg.imu.linear_acceleration.x = acc_x
                msg.imu.linear_acceleration.y = acc_y
                msg.imu.linear_acceleration.z = acc_z
                msg.mag_field.magnetic_field.x = mag_x
                msg.mag_field.magnetic_field.y = mag_y
                msg.mag_field.magnetic_field.z = mag_z
                msg.imu_str = raw_data
                self.publisher_node.publish(msg)
                self.get_logger().info(str(msg))

            except (ValueError, IndexError) as e:
                self.get_logger().warn(f"Parse error: {e}")
                return


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