import rclpy
from rclpy.node import Node
# IMPORTANT: Change message type from CompressedImage to Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class NavCameraViewer(Node):
    def __init__(self):
        super().__init__('nav_camera_viewer')
        
        cv2.namedWindow("Navigation Camera Stream", cv2.WINDOW_NORMAL)
        self.bridge = CvBridge()
        
        # Subscribe to the navigation camera's raw image topic
        self.create_subscription(
            Image,  # Use the standard 'Image' message type
            '/navigation_camera/image_raw', 
            self.image_callback, 
            10)
            
        self.get_logger().info("Navigation camera viewer started. Waiting for images...")

    def image_callback(self, msg):
        """This function is called every time a new image is received."""
        try:
            # Use 'imgmsg_to_cv2' for uncompressed image messages
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # The navigation camera is likely mounted correctly, 
            # so the 90-degree rotation is probably not needed.
            # If it's still sideways, you can add it back:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            cv2.imshow("Navigation Camera Stream", img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NavCameraViewer()
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