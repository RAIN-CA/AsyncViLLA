import json
import re
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image as PILImage
import torch


VALID_ACTIONS = {"forward", "left", "right", "stop"}
VALID_INTENSITIES = {"low", "medium", "high"}


def parse_command(text: str) -> Tuple[str, str]:
    text = text.strip().lower()

    strict = re.search(
        r"\b(forward|left|right|stop)\s*,\s*(low|medium|high)\b",
        text,
    )
    if strict:
        return strict.group(1), strict.group(2)

    found_action = "empty"
    found_intensity = "empty"

    for action in ["forward", "left", "right", "stop"]:
        if re.search(rf"\b{action}\b", text):
            found_action = action
            break

    for intensity in ["low", "medium", "high"]:
        if re.search(rf"\b{intensity}\b", text):
            found_intensity = intensity
            break

    return found_action, found_intensity


def pooled_feature_from_last_hidden(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    masked_hidden = last_hidden * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = masked_hidden.sum(dim=1) / denom
    return pooled


def maybe_get_backbone(model):
    return getattr(model, "model", model)


def load_raw_rgb_image(image_path: str, width: int, height: int, encoding: str) -> PILImage.Image:
    if encoding.lower() != "rgb8":
        raise ValueError(f"Unsupported encoding: {encoding}. Only rgb8 is supported now.")

    raw = Path(image_path).read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint8)

    expected = width * height * 3
    if arr.size != expected:
        raise ValueError(
            f"RAW size mismatch for {image_path}: got {arr.size}, expected {expected}"
        )

    arr = arr.reshape((height, width, 3))
    return PILImage.fromarray(arr, mode="RGB")


class RealVLMNode(Node):
    def __init__(self):
        super().__init__("real_vlm_node")

        self.declare_parameter("cache_json_path", "/dev/shm/asyncvilla_cache/latest_window.json")
        self.declare_parameter("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.declare_parameter("publish_hz", 0.5)
        self.declare_parameter("use_all_frames", True)
        self.declare_parameter("max_new_tokens", 6)

        self.declare_parameter("command_topic", "/asyncvilla/high_level_command")
        self.declare_parameter("action_topic", "/asyncvilla/high_level_action")
        self.declare_parameter("intensity_topic", "/asyncvilla/high_level_intensity")
        self.declare_parameter("latent_topic", "/asyncvilla/latent_feature")

        self.cache_json_path = Path(self.get_parameter("cache_json_path").value)
        self.model_name = self.get_parameter("model_name").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.use_all_frames = bool(self.get_parameter("use_all_frames").value)
        self.max_new_tokens = int(self.get_parameter("max_new_tokens").value)

        self.command_topic = self.get_parameter("command_topic").value
        self.action_topic = self.get_parameter("action_topic").value
        self.intensity_topic = self.get_parameter("intensity_topic").value
        self.latent_topic = self.get_parameter("latent_topic").value

        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.action_pub = self.create_publisher(String, self.action_topic, 10)
        self.intensity_pub = self.create_publisher(String, self.intensity_topic, 10)
        self.latent_pub = self.create_publisher(Float32MultiArray, self.latent_topic, 10)

        self.get_logger().info(f"Loading model: {self.model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.backbone = maybe_get_backbone(self.model)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.timer = self.create_timer(1.0 / self.publish_hz, self.on_timer)

        self.get_logger().info(f"Reading cache from: {self.cache_json_path}")
        self.get_logger().info(f"Publishing command   -> {self.command_topic}")
        self.get_logger().info(f"Publishing action    -> {self.action_topic}")
        self.get_logger().info(f"Publishing intensity -> {self.intensity_topic}")
        self.get_logger().info(f"Publishing latent    -> {self.latent_topic}")

    def build_prompt(self) -> str:
        return """
You are a robot navigation intention predictor.

You are given visual observations where:
- earlier frames are history frames
- the last frame is the current frame

Choose exactly:
- one action from [forward, left, right, stop]
- one intensity from [low, medium, high]

Output rules:
- Output exactly one line
- Output format must be: action,intensity
- Do not explain
- Do not output any extra text

Example valid outputs:
forward,medium
left,low
right,high
stop,low
""".strip()

    def load_records(self) -> List[dict]:
        if not self.cache_json_path.exists():
            raise FileNotFoundError(f"Cache JSON not found: {self.cache_json_path}")

        with open(self.cache_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return obj.get("records", [])

    def build_messages(self, records: List[dict]):
        prompt = self.build_prompt()
        content = []

        selected = records if self.use_all_frames else [records[-1]]

        for rec in selected:
            image_meta = rec["image_meta"]
            pil_img = load_raw_rgb_image(
                image_path=rec["image_path"],
                width=image_meta["width"],
                height=image_meta["height"],
                encoding=image_meta["encoding"],
            )
            role = rec.get("role", "history")
            content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": f"This frame is a {role} frame."})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return messages

    def publish_outputs(self, action: str, intensity: str, feature: np.ndarray):
        command_msg = String()
        action_msg = String()
        intensity_msg = String()
        latent_msg = Float32MultiArray()

        command_msg.data = f"{action},{intensity}"
        action_msg.data = action
        intensity_msg.data = intensity
        latent_msg.data = feature.astype(np.float32).tolist()

        self.command_pub.publish(command_msg)
        self.action_pub.publish(action_msg)
        self.intensity_pub.publish(intensity_msg)
        self.latent_pub.publish(latent_msg)

        self.get_logger().info(
            f"Published command={command_msg.data} | latent_dim={len(latent_msg.data)}"
        )

    def publish_empty(self):
        self.publish_outputs("empty", "empty", np.zeros((1,), dtype=np.float32))

    def on_timer(self):
        try:
            records = self.load_records()
            if len(records) == 0:
                self.get_logger().warn("No records in latest_window.json")
                self.publish_empty()
                return

            messages = self.build_messages(records)

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            raw_output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            action, intensity = parse_command(raw_output)

            del generated_ids
            del generated_ids_trimmed
            torch.cuda.empty_cache()

            with torch.inference_mode():
                outputs = self.backbone(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            last_hidden = outputs.hidden_states[-1]
            pooled = pooled_feature_from_last_hidden(last_hidden, inputs["attention_mask"])
            pooled_cpu = torch.nan_to_num(
                pooled[0].detach().float().cpu(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).numpy()

            self.get_logger().info(f"Raw VLM output: {raw_output}")
            self.publish_outputs(action, intensity, pooled_cpu)

            del outputs
            del last_hidden
            del pooled
            del pooled_cpu
            del inputs
            del image_inputs
            del video_inputs
            torch.cuda.empty_cache()

        except Exception as e:
            self.get_logger().error(f"VLM node inference failed: {e}")
            self.publish_empty()


def main(args=None):
    rclpy.init(args=args)
    node = RealVLMNode()
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
