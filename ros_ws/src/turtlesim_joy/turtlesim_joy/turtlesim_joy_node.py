import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class TurtlesimJoyNode(Node):
    def __init__(self):
        super().__init__("turtlesim_joy_node")
        self.publisher = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        self.get_logger().info("Joystick node started successfully")

    def joy_callback(self, joy_msg):
        twist = Twist()
        twist.linear.x = joy_msg.axes[1]
        twist.angular.z = joy_msg.axes[0]

        if joy_msg.axes[2] < 1:  # A button
            self.get_logger().info("Going faster")
            twist.linear.x *= 4

        self.publisher.publish(twist)
        self.get_logger().info(f"Joystick input: linear={twist.linear.x}, angular={twist.angular.z}")


def main(args=None):
    rclpy.init(args=args)
    node = TurtlesimJoyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
