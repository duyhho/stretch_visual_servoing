"""
SETUP INSTRUCTIONS:
1. First, run these commands in separate terminals to start the robot and camera:
   Terminal 1: ros2 launch stretch_core stretch_driver.launch.py
   Terminal 2: ros2 launch stretch_core navigation_camera.launch.py
    
2. Then run this script directly:
   python3 nav_camera_test.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class NavCameraViewer(Node):
    def __init__(self):
        super().__init__('nav_camera_viewer')
        
        cv2.namedWindow("Navigation Camera Stream", cv2.WINDOW_NORMAL)
        self.bridge = CvBridge()

        # Initialize YOLO model
        self.model = YOLO('yolo11n-seg.pt')

        # CONFIGURABLE: Change this to detect different objects!
        self.TARGET_OBJECTS = ['banana', 'apple', 'sports ball']
        # self.TARGET_OBJECTS = ['sports ball', 'banana', 'apple', 'orange', 'backpack', 'bottle', 'cup', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book']

        # RESIZE STREAMING WINDOW SIZE
        # After 90° counterclockwise rotation: width becomes height, height becomes width
        original_width, original_height = 800, 600
        rotated_width = original_height
        rotated_height = original_width
        # Apply scale factor (2x for now)
        scale_factor = 2.0
        final_width = int(rotated_width * scale_factor)
        final_height = int(rotated_height * scale_factor)
        # Set window size upfront
        cv2.resizeWindow("Navigation Camera Stream", final_width, final_height)

        
        # Subscribe to the navigation camera's raw image topic
        self.create_subscription(
            Image,  # Use the standard 'Image' message type
            '/navigation_camera/image_raw', 
            self.image_callback, 
            10)
            
        self.get_logger().info("Navigation camera YOLO detector started. Waiting for images...")
        self.get_logger().info(f"Currently detecting: {self.TARGET_OBJECTS}")

    def image_callback(self, msg):
        """This function is called every time a new image is received."""
        try:
            # Use 'imgmsg_to_cv2' for uncompressed image messages
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Rotate the image
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Run YOLO detection
            results = self.model(img, verbose=False, conf=0.5)[0]

            # Process ALL detected objects and show their confidence scores
            detected_objects = []

            if results.masks is not None:
                for i, box in enumerate(results.boxes):
                    cls_name = results.names[int(box.cls[0])]
                    
                    # Check if this object is in our target list
                    if cls_name in self.TARGET_OBJECTS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        area = (x2 - x1) * (y2 - y1)
                        
                        detected_objects.append({
                            'box': (x1, y1, x2, y2),
                            'mask': results.masks[i].xy[0],
                            'class': cls_name,
                            'confidence': confidence,
                            'area': area
                        })

            # Sort by confidence (highest first) and select the best one
            if detected_objects:
                detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Log all detections for debugging
                self.get_logger().info("All detections this frame:")
                for obj in detected_objects:
                    self.get_logger().info(f"  {obj['class']}: {obj['confidence']:.3f} confidence, area: {obj['area']}")
                
                # Use the highest confidence detection
                best_object = detected_objects[0]
                
                # Optional: Only use detections above a certain confidence threshold
                if best_object['confidence'] > 0.7:  # Adjust this threshold as needed
                    self.get_logger().info(f">>> SELECTED: {best_object['class']} (confidence: {best_object['confidence']:.3f})")
                else:
                    self.get_logger().info(f">>> All detections below confidence threshold (0.7)")
                    best_object = None
            else:
                best_object = None

            # Draw ALL detection results (not just the best one)
            for obj in detected_objects:
                x1, y1, x2, y2 = obj['box']
                object_name = obj['class']
                confidence = obj['confidence']
                
                # Use different colors based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw mask outline
                cv2.polylines(img, [obj['mask'].astype(np.int32)], isClosed=True, color=color, thickness=2)
                
                # Draw label
                label = f"{object_name} ({confidence:.2f})"
                self.draw_label(img, label, (x1, y1), text_color=color)

            # Scale the image for display
            scale_factor = 2.0
            width = int(img.shape[1] * scale_factor)
            height = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (width, height))
            
            cv2.imshow("Navigation Camera Stream", img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def draw_label(self, img, text, position, font_scale=0.7, font_thickness=2, text_color=(255, 0, 255)):
        """Draw a text label with background for better visibility"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, font_thickness)

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