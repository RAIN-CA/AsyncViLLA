import itertools

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VLMNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.declare_parameter("mode_topic", "/asyncvilla/high_level_mode")
        self.declare_parameter("publish_hz", 1.0)

        self.mode_topic = self.get_parameter("mode_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.publisher = self.create_publisher(String, self.mode_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

        self.mode_iter = itertools.cycle(["forward", "left", "right", "stop"])

        self.get_logger().info(f"Publishing high-level mode to: {self.mode_topic}")

    def on_timer(self):
        msg = String()
        msg.data = next(self.mode_iter)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published mode: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
