"""
Terminal 1: ros2 launch stretch_core stretch_driver.launch.py
Terminal 2: ros2 launch stretch_core d435i_low_resolution.launch.py

validating FOV for Head camera RGB, raw_depth, and aligned depth
"""


#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo

def fovs_from_caminfo(ci: CameraInfo):
    fx = ci.k[0]; fy = ci.k[4]
    w  = ci.width; h  = ci.height
    hfov = 2.0 * math.degrees(math.atan(w / (2.0 * fx)))
    vfov = 2.0 * math.degrees(math.atan(h / (2.0 * fy)))
    return hfov, vfov

class FOVAll(Node):
    def __init__(self):
        super().__init__('fov_all')

        self.topics = {
            "RGB (color)":   "/camera/color/camera_info",
            "Aligned Depth": "/camera/aligned_depth_to_color/camera_info",
            "Raw Depth":     "/camera/depth/camera_info",
        }
        self.data = {name: None for name in self.topics}
        self.printed = set()
        self.done = False

        self.subs = []
        for name, topic in self.topics.items():
            self.subs.append(self.create_subscription(CameraInfo, topic, self._make_cb(name), 10))

        self.get_logger().info("Listening for CameraInfo on:")
        for n, t in self.topics.items():
            self.get_logger().info(f"  - {n}: {t}")

    def _make_cb(self, name):
        def _cb(msg: CameraInfo):
            if self.data[name] is None:
                self.data[name] = msg
                self._maybe_print()
        return _cb

    def _maybe_print(self):
        # print each stream once
        for name, ci in self.data.items():
            if ci is not None and name not in self.printed:
                hf, vf = fovs_from_caminfo(ci)
                self.get_logger().info(f"[{name}] size={ci.width}x{ci.height}  HFOV={hf:.2f}°, VFOV={vf:.2f}°")
                self.printed.add(name)

        # when all arrived, print summary and mark done
        if all(ci is not None for ci in self.data.values()) and not self.done:
            color   = self.data["RGB (color)"]
            aligned = self.data["Aligned Depth"]
            raw     = self.data["Raw Depth"]
            ch, cv = fovs_from_caminfo(color)
            ah, av = fovs_from_caminfo(aligned)
            rh, rv = fovs_from_caminfo(raw)

            self.get_logger().info("--- FOV comparison ---")
            self.get_logger().info(f"Aligned vs RGB: ΔH={abs(ah-ch):.2f}°, ΔV={abs(av-cv):.2f}°")
            self.get_logger().info(f"RawDepth vs RGB: ΔH={abs(rh-ch):.2f}°, ΔV={abs(rv-cv):.2f}°")
            self.done = True

def main():
    rclpy.init()
    node = FOVAll()
    exec = SingleThreadedExecutor()
    exec.add_node(node)

    t0 = time.time()
    TIMEOUT = 10.0  # seconds

    try:
        while rclpy.ok() and not node.done and (time.time() - t0) < TIMEOUT:
            exec.spin_once(timeout_sec=0.2)
    finally:
        exec.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

    # If you want a non-zero exit when timed out without all three:
    # import sys
    # if not node.done:
    #     sys.exit(1)

if __name__ == "__main__":
    main()
