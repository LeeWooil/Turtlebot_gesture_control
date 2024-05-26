#!/usr/bin/env python
import rclpy
from std_msgs.msg import String

def callback(msg):
    print("Received: ", msg.data)

def main():
    rclpy.init()
    node = rclpy.create_node('hand_tracking_subscriber')
    sub = node.create_subscription(String, 'status_hand', callback, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

