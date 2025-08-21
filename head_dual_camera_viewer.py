"""
SETUP INSTRUCTIONS:
1. First, run these commands in separate terminals to start the robot and camera:
   Terminal 1: ros2 launch stretch_core stretch_driver.launch.py
   Terminal 2: ros2 launch stretch_core d435i_low_resolution.launch.py
   Terminal 3: ros2 launch stretch_core navigation_camera.launch.py

2. Then run this script directly:
   python3 head_dual_camera_viewer.py

"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class DualCameraViewer(Node):
    def __init__(self):
        super().__init__('dual_camera_viewer')
        self.bridge = CvBridge()

        # Placeholders for the latest frame from each camera
        self.nav_frame = None
        self.d435if_frame = None

        # --- KEY CHANGE: Create the window as resizable ---
        # This allows you to click and drag the window edges.
        cv2.namedWindow("Side-by-Side Camera Comparison", cv2.WINDOW_NORMAL)

        # Create two separate subscribers for each camera
        self.create_subscription(
            Image,
            '/navigation_camera/image_raw',
            self.nav_camera_callback,
            10)

        self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.d435if_callback,
            10)

        # Create a timer to update the display at ~30Hz
        self.timer = self.create_timer(1.0/30.0, self.update_display)

        self.get_logger().info("Dual camera viewer started. Waiting for both streams...")

    def nav_camera_callback(self, msg):
        """Processes images from the wide-angle navigation camera."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.nav_frame = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            self.get_logger().error(f"Failed to process nav camera image: {e}")

    def d435if_callback(self, msg):
        """Processes images from the D435if head camera."""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            self.d435if_frame = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        except Exception as e:
            self.get_logger().error(f"Failed to process D435if image: {e}")

    def update_display(self):
        """Combines and displays the latest frames from both cameras."""
        if self.nav_frame is None or self.d435if_frame is None:
            return

        # --- KEY CHANGE: Dynamic resizing logic ---
        # Instead of a fixed height, we'll match the D435if's height
        # to the navigation camera's native height. OpenCV's resizable
        # window will handle the rest.

        # Get the native height of the nav camera frame
        h_nav, w_nav, _ = self.nav_frame.shape
        
        # Get the dimensions of the D435if frame
        h_d435, w_d435, _ = self.d435if_frame.shape

        # Calculate the new width for the D435if frame to match the nav frame's height
        scale = h_nav / h_d435
        new_w_d435 = int(w_d435 * scale)
        d435_resized = cv2.resize(self.d435if_frame, (new_w_d435, h_nav))

        # Add labels to each image (use copies to not modify originals)
        nav_display = self.nav_frame.copy()
        d435_display = d435_resized.copy()
        cv2.putText(nav_display, 'Nav Camera (Wide)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(d435_display, 'D435if Head Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine the two images horizontally
        combined_image = cv2.hconcat([nav_display, d435_display])

        # Show the combined view in the resizable window
        cv2.imshow("Side-by-Side Camera Comparison", combined_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node.")
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()