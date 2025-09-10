"""
SETUP INSTRUCTIONS:
1. First, run these commands in separate terminals to start the robot and camera:
   Terminal 1: ros2 launch stretch_core stretch_driver.launch.py
   Terminal 2: ros2 launch stretch_core d435i_low_resolution.launch.py
    
2. Then run this script directly:
   python3 head_camera_full_scan_test.py
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import time
import cv2
import math
import os
import threading
import math

def compute_stops(min_angle, max_angle, fov_deg, overlap=0.2):
    """Return a list of center angles (radians) that tile [min,max] with FOV and overlap."""
    fov = math.radians(fov_deg)
    step = math.radians(fov_deg * (1.0 - overlap))
    centers = []
    first = min_angle + 0.5 * fov
    last  = max_angle - 0.5 * fov
    if last < first:
        # FOV is wider than the range; just center once
        return [0.5 * (min_angle + max_angle)]
    a = first
    # march by step
    while a <= last - 1e-9:
        centers.append(a)
        a += step
    # ensure we end exactly at 'last' to cover the right edge
    if not centers or centers[-1] < last - 1e-9:
        centers.append(last)

    # iterate from the positive radians
    centers.reverse()
    return centers

class AsyncScanner(Node):
    def __init__(self):
        super().__init__('async_scanner')
        self.lock = threading.Lock()
        
        self.action_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        self.image_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        self.output_dir = "scan_images_async"
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"Saving images to '{self.output_dir}/' directory.")

        # --- State Machine Setup ---
        # Camera FOVs (physical, not the rotated image's axes)
        RGB_FOV_H_DEG = 42.0
        RGB_FOV_V_DEG = 69.0

        # Choose your overlap (experiment with 0.2 ~ 0.3 for reliable stitching/detection)
        OVERLAP = 0.20

        # Joint limits you measured / from specs
        PAN_MIN  = -4.061981126321178
        PAN_MAX  =  1.7410681942502029
        TILT_MIN = -1.8469128686143121
        TILT_MAX =  0.4893398713355195

        # Compute stops
        self.pan_angles  = compute_stops(PAN_MIN, PAN_MAX, RGB_FOV_H_DEG, overlap=OVERLAP)
        self.tilt_angles = [0.0] # Top, middle, bottom
        self.current_tilt_idx = 0
        self.current_pan_idx = 0

        self.image_ct = 1

        self.get_logger().info(f"Pan stops (deg): {[round(math.degrees(a),1) for a in self.pan_angles]}")
        self.get_logger().info(f"Tilt stops (deg): {[round(math.degrees(a),1) for a in self.tilt_angles]}")
        
        self.get_logger().info("Waiting for Action Server...")
        self.action_client.wait_for_server()
        self.get_logger().info("Action Server is ready. Starting scan.")
        self.trigger_next_move()

    def image_callback(self, msg):
        """Receives and stores the latest image in a thread-safe way."""
        with self.lock:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.latest_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

    def trigger_next_move(self):
        """Sends the goal for the current pan/tilt index."""
        if self.current_tilt_idx >= len(self.tilt_angles):
            self.get_logger().info("Scan complete.")
            # Move back to center and shut down
            self.send_goal(0.0, 0.0, self.shutdown_callback)
            return

        pan_angle = self.pan_angles[self.current_pan_idx]
        tilt_angle = self.tilt_angles[self.current_tilt_idx]
        
        self.get_logger().info(f"Moving to Pan: {math.degrees(pan_angle):.1f}°, Tilt: {math.degrees(tilt_angle):.1f}°")
        self.send_goal(pan_angle, tilt_angle, self.goal_done_callback)

    def send_goal(self, pan_angle, tilt_angle, done_callback):
        """Creates and sends a goal, attaching a callback for when it's done."""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
        point = JointTrajectoryPoint()
        point.positions = [pan_angle, tilt_angle]
        point.time_from_start = Duration(sec=2, nanosec=0)
        goal_msg.trajectory.points.append(point)

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        # Attach the callback function to be executed when the goal is accepted
        send_goal_future.add_done_callback(
            lambda future: self.goal_accepted_callback(future, done_callback)
        )

    def goal_accepted_callback(self, future, done_callback):
        """Checks if the goal was accepted and gets the result future."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return
        
        result_future = goal_handle.get_result_async()
        # Attach the final callback to be executed when the move is finished
        result_future.add_done_callback(done_callback)

    def goal_done_callback(self, future):
        """This is the main callback. It runs AFTER a move is complete."""
        result = future.result().result
        if result:
            self.get_logger().info('Movement finished successfully.')
            # Give the camera stream a brief moment to publish a clean frame
            time.sleep(0.5)
            
            # Save the image
            with self.lock:
                if self.latest_image is not None:
                    pan = self.pan_angles[self.current_pan_idx]
                    tilt = self.tilt_angles[self.current_tilt_idx]
                    pan_deg  = math.degrees(pan)
                    tilt_deg = math.degrees(tilt)

                    # zero-padded, lexicographically sortable name
                    filename = os.path.join(
                        self.output_dir,
                        f"{self.image_ct:03d}_tilt_{tilt_deg:+06.1f}_pan_{pan_deg:+06.1f}.jpg"
                    )
                    
                    cv2.imwrite(filename, self.latest_image)
                    self.get_logger().info(f"Image saved to {filename}")

                    self.image_ct += 1
                else:
                    self.get_logger().warn("Latest image was None.")
            
            # --- Move to the next position in the grid ---
            self.current_pan_idx += 1
            if self.current_pan_idx >= len(self.pan_angles):
                self.current_pan_idx = 0
                self.current_tilt_idx += 1
            
            # Trigger the next movement
            self.trigger_next_move()
        else:
            self.get_logger().error(f'Goal failed with error code: {result.error_code}')
    
    def shutdown_callback(self, future):
        self.get_logger().info("Script finished. Shutting down.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    scanner_node = AsyncScanner()
    executor.add_node(scanner_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
        
if __name__ == '__main__':
    main()