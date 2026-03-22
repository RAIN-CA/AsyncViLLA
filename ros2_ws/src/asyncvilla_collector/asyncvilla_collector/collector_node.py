import json
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


class CollectorNode(Node):
    def __init__(self):
        super().__init__("collector_node")

        self.declare_parameter("camera_topic", "/robot_1/camera/front/image_raw")
        self.declare_parameter("odom_topic", "/robot_1/odom")
        self.declare_parameter("image_save_interval_sec", 0.5)
        self.declare_parameter("max_buffer_size", 32)
        self.declare_parameter("export_window_size", 4)
        self.declare_parameter("shared_cache_dir", "/dev/shm/asyncvilla_cache")

        self.camera_topic = self.get_parameter("camera_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.image_save_interval_sec = float(self.get_parameter("image_save_interval_sec").value)
        self.max_buffer_size = int(self.get_parameter("max_buffer_size").value)
        self.export_window_size = int(self.get_parameter("export_window_size").value)
        self.shared_cache_dir = Path(self.get_parameter("shared_cache_dir").value)

        self.latest_image = None
        self.latest_odom = None
        self.last_store_time_sec = None
        self.frame_idx = 0

        self.buffer = deque(maxlen=self.max_buffer_size)
        self.shared_cache_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(Image, self.camera_topic, self.on_image, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.status_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info(f"Collector subscribed image : {self.camera_topic}")
        self.get_logger().info(f"Collector subscribed odom  : {self.odom_topic}")
        self.get_logger().info(f"RAM FIFO buffer size      : {self.max_buffer_size}")
        self.get_logger().info(f"Export window size        : {self.export_window_size}")
        self.get_logger().info(f"Shared cache dir          : {self.shared_cache_dir}")

    def on_image(self, msg: Image):
        self.latest_image = msg
        self.try_store()

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def try_store(self):
        if self.latest_image is None or self.latest_odom is None:
            return

        stamp_sec = self.latest_image.header.stamp.sec + self.latest_image.header.stamp.nanosec * 1e-9
        if self.last_store_time_sec is not None:
            if stamp_sec - self.last_store_time_sec < self.image_save_interval_sec:
                return

        self.frame_idx += 1
        self.last_store_time_sec = stamp_sec

        odom = self.latest_odom
        sample = {
            "frame_id": self.frame_idx,
            "timestamp_sec": self.latest_image.header.stamp.sec,
            "timestamp_nanosec": self.latest_image.header.stamp.nanosec,
            "image": {
                "height": self.latest_image.height,
                "width": self.latest_image.width,
                "encoding": self.latest_image.encoding,
                "step": self.latest_image.step,
                "frame_id": self.latest_image.header.frame_id,
                "data": bytes(self.latest_image.data),
            },
            "state": {
                "pose": {
                    "x": odom.pose.pose.position.x,
                    "y": odom.pose.pose.position.y,
                    "z": odom.pose.pose.position.z,
                },
                "twist": {
                    "linear_x": odom.twist.twist.linear.x,
                    "linear_y": odom.twist.twist.linear.y,
                    "linear_z": odom.twist.twist.linear.z,
                    "angular_z": odom.twist.twist.angular.z,
                },
                "frame_id": odom.header.frame_id,
                "child_frame_id": odom.child_frame_id,
            },
        }

        was_full = len(self.buffer) == self.max_buffer_size
        self.buffer.append(sample)
        self.export_latest_window()

        if was_full:
            self.get_logger().info(f"FIFO full, dropped oldest frame and appended frame {self.frame_idx}")
        else:
            self.get_logger().info(f"Buffered frame {self.frame_idx} | buffer size: {len(self.buffer)}/{self.max_buffer_size}")

    def export_latest_window(self):
        window = list(self.buffer)[-self.export_window_size:]

        # 清理旧导出文件
        for p in self.shared_cache_dir.glob("frame_*.raw"):
            p.unlink(missing_ok=True)
        for p in self.shared_cache_dir.glob("frame_*.json"):
            p.unlink(missing_ok=True)

        export_records = []
        for idx, sample in enumerate(window):
            raw_path = self.shared_cache_dir / f"frame_{idx}.raw"
            json_path = self.shared_cache_dir / f"frame_{idx}.json"

            raw_path.write_bytes(sample["image"]["data"])

            meta = {
                "buffer_frame_id": sample["frame_id"],
                "timestamp_sec": sample["timestamp_sec"],
                "timestamp_nanosec": sample["timestamp_nanosec"],
                "role": "current" if idx == len(window) - 1 else "history",
                "image_path": str(raw_path),
                "image_meta": {
                    "height": sample["image"]["height"],
                    "width": sample["image"]["width"],
                    "encoding": sample["image"]["encoding"],
                    "step": sample["image"]["step"],
                    "frame_id": sample["image"]["frame_id"],
                },
                "state": sample["state"],
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            export_records.append(meta)

        latest_window_path = self.shared_cache_dir / "latest_window.json"
        with open(latest_window_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "window_size": len(export_records),
                    "export_window_size": self.export_window_size,
                    "shared_cache_dir": str(self.shared_cache_dir),
                    "records": export_records,
                },
                f,
                indent=2,
            )

    def print_status(self):
        if not self.buffer:
            self.get_logger().info("Buffer is empty")
            return

        oldest = self.buffer[0]["frame_id"]
        newest = self.buffer[-1]["frame_id"]
        summary = {
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.max_buffer_size,
            "oldest_frame_id": oldest,
            "newest_frame_id": newest,
            "shared_cache_dir": str(self.shared_cache_dir),
        }
        self.get_logger().info(f"Buffer status: {json.dumps(summary)}")


def main(args=None):
    rclpy.init(args=args)
    node = CollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
