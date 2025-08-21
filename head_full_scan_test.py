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
        self.pan_angles = [-4.0, -3.24, -2.48, -1.72, -0.96, -0.2, 0.56, 1.32]
        self.tilt_angles = [0.44, -0.7, -1.84] # Top, middle, bottom
        self.current_tilt_idx = 0
        self.current_pan_idx = 0
        
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
                    pan_deg = int(math.degrees(self.pan_angles[self.current_pan_idx]))
                    tilt_deg = int(math.degrees(self.tilt_angles[self.current_tilt_idx]))
                    filename = os.path.join(self.output_dir, f"scan_tilt_{tilt_deg}_pan_{pan_deg}.jpg")
                    cv2.imwrite(filename, self.latest_image)
                    self.get_logger().info(f"Image saved to {filename}")
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