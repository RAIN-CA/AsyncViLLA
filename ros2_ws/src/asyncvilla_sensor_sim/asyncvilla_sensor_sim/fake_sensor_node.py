import math

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


class FakeSensorNode(Node):
    def __init__(self):
        super().__init__("fake_sensor_node")

        self.declare_parameter("camera_topic", "/robot_1/camera/front/image_raw")
        self.declare_parameter("odom_topic", "/robot_1/odom")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("publish_hz", 5.0)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.image_pub = self.create_publisher(Image, self.camera_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)

        self.t = 0.0
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

        self.get_logger().info(f"Publishing fake image to: {self.camera_topic}")
        self.get_logger().info(f"Publishing fake odom to: {self.odom_topic}")

    def on_timer(self):
        now = self.get_clock().now().to_msg()
        self.t += 0.1

        img = self.make_fake_image(now)
        odom = self.make_fake_odom(now)

        self.image_pub.publish(img)
        self.odom_pub.publish(odom)

    def make_fake_image(self, stamp):
        h, w = self.image_height, self.image_width
        img_np = np.zeros((h, w, 3), dtype=np.uint8)

        cx = int((math.sin(self.t) * 0.4 + 0.5) * (w - 1))
        cy = int((math.cos(self.t * 0.7) * 0.4 + 0.5) * (h - 1))

        img_np[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img_np[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        img_np[max(0, cy - 20):min(h, cy + 20), max(0, cx - 20):min(w, cx + 20), :] = [255, 255, 255]

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_front"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img_np.tobytes()
        return msg

    def make_fake_odom(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = math.sin(self.t)
        msg.pose.pose.position.y = math.cos(self.t)
        msg.pose.pose.position.z = 0.0

        msg.twist.twist.linear.x = 0.5
        msg.twist.twist.angular.z = 0.2 * math.sin(self.t)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FakeSensorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
