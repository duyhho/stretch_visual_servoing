"""
Execute belows to get camera streaming:
ros2 launch stretch_core stretch_driver.launch.py
ros2 launch stretch_core d435i_low_resolution.launch.py

"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs

import message_filters

class HeadCameraBallDetector(Node):
    def __init__(self):
        super().__init__('head_camera_ball_detector')
        self.bridge = CvBridge()
        
        # APPLIED CHANGE: Using the segmentation model for more precise detection
        self.model = YOLO('yolov8n-seg.pt')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.camera_info = None
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)

        self.sub_image = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        # self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_rect_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.image_depth_callback)

        self.get_logger().info("Ball detector node initialized and waiting for messages.")

    def info_callback(self, msg):
        self.camera_info = msg
        self.destroy_subscription(self.info_sub)
        self.get_logger().info("Received camera info and unsubscribed.")

    def image_depth_callback(self, image_msg, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera_info...")
            return

        img = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # ######################################################################
        # ## ADD THESE TWO LINES TO ROTATE THE IMAGES ##
        # ######################################################################
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
        # ######################################################################

        results = self.model(img, verbose=False)[0]

        # APPLIED CHANGE: Logic to find the single "best" ball (largest on screen)
        best_ball = None
        largest_area = 0

        # The results from a segmentation model have a .masks attribute
        if results.masks is not None:
            for i, box in enumerate(results.boxes):
                cls_name = results.names[int(box.cls[0])]
                if cls_name == 'sports ball':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > largest_area:
                        largest_area = area
                        best_ball = {
                            'box': (x1, y1, x2, y2),
                            'mask': results.masks[i].xy[0] # Get the mask polygon
                        }
        
        # APPLIED CHANGE: All processing is now done only on the best_ball, outside the loop
        if best_ball is not None:
            x1, y1, x2, y2 = best_ball['box']

            # APPLIED CHANGE: Use the segmentation mask for a highly robust depth calculation
            try:
                # Crop the depth image to the bounding box
                depth_crop = depth_img[y1:y2, x1:x2]
                
                # Create a blank mask of the same size as the crop
                mask_crop = np.zeros(depth_crop.shape, dtype=np.uint8)

                # Get the mask polygon and shift its origin to the crop's top-left corner
                polygon = best_ball['mask']
                polygon_shifted = polygon - np.array([x1, y1])

                # Draw the filled polygon on the blank mask
                cv2.fillPoly(mask_crop, [polygon_shifted.astype(np.int32)], 255)

                # Get all valid depth points within the mask
                valid_depths = depth_crop[mask_crop == 255]
                valid_depths = valid_depths[valid_depths > 0] # Filter out zero-depth values

                if len(valid_depths) == 0:
                    self.get_logger().warn("Detected ball has no valid depth data within its mask.")
                    return # Skip this frame
                
                # Calculate the median depth
                depth_mm = np.median(valid_depths)
                depth_m = depth_mm / 1000.0

            except Exception as e:
                self.get_logger().error(f"Error in depth calculation: {e}")
                # We draw the image at the end, so just return to skip processing this frame
                return

            # --- The rest of the logic is the same, using the new robust depth ---
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            fx, fy = self.camera_info.k[0], self.camera_info.k[4]
            cx_k, cy_k = self.camera_info.k[2], self.camera_info.k[5]
            X = (cx - cx_k) * depth_m / fx
            Y = (cy - cy_k) * depth_m / fy
            Z = depth_m

            pt_cam = PointStamped()
            pt_cam.header.frame_id = "camera_color_optical_frame"
            pt_cam.header.stamp = rclpy.time.Time().to_msg()
            pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = X, Y, Z

            try:
                pt_base = self.tf_buffer.transform(pt_cam, 'base_link', timeout=rclpy.duration.Duration(seconds=1.0))
                px, py, pz = pt_base.point.x, pt_base.point.y, pt_base.point.z
                text = f"({px:.2f}, {py:.2f}, {pz:.2f}) m"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # Draw the mask outline for visualization
                cv2.polylines(img, [best_ball['mask'].astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
                self.get_logger().info(f"Ball position (base_link): {text}")
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")

        cv2.imshow("YOLO Ball Detection", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HeadCameraBallDetector()
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
