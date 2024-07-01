#!/usr/bin/env python

import rclpy
from rclpy.node import Node
#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import threading

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

msg = """
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity
a/d : increase/decrease angular velocity

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""

class TeleopTurtleBot3(Node):
    def __init__(self):
        super().__init__('turtlebot3_teleop')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(
            String,
            'status_hand',
            self.listener_callback,
            10)
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.last_msg_time = self.get_clock().now()

        self.timer = self.create_timer(0.1, self.check_timeout)  # Check every 0.1 seconds

    def listener_callback(self, msg):
        self.last_msg_time = self.get_clock().now()
        key = msg.data
        twist = Twist()
        if key == "Front":
            self.target_linear_vel = LIN_VEL_STEP_SIZE + 0.5
            self.target_angular_vel = 0.0
        elif key == 'Back':
            self.target_linear_vel = -LIN_VEL_STEP_SIZE - 0.5
            self.target_angular_vel = 0.0
        elif key == 'Left':
            self.target_linear_vel = 0.0
            self.target_angular_vel = ANG_VEL_STEP_SIZE + 0.5
        elif key == 'Right':
            self.target_linear_vel = 0.0
            self.target_angular_vel = -ANG_VEL_STEP_SIZE - 0.5
        elif key == 'Stop' or key == 's':
            self.target_linear_vel = 0.0
            self.target_angular_vel = 0.0

        twist.linear.x = self.target_linear_vel
        twist.angular.z = self.target_angular_vel
        self.publisher_.publish(twist)
        print("currently:\tlinear vel %s\t angular vel %s " % (self.target_linear_vel, self.target_angular_vel))

    def check_timeout(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.last_msg_time).nanoseconds / 1e9
        if elapsed_time > 1.0:  # 1 second timeout
            self.target_linear_vel = 0.0
            self.target_angular_vel = 0.0
            twist = Twist()
            twist.linear.x = self.target_linear_vel
            twist.angular.z = self.target_angular_vel
            self.publisher_.publish(twist)
            print("No gesture detected. Stopping the robot.")

def main(args=None):
    rclpy.init(args=args)
    teleop_turtlebot3 = TeleopTurtleBot3()
    print(msg)
    rclpy.spin(teleop_turtlebot3)
    teleop_turtlebot3.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
