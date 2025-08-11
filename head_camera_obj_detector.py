"""
SETUP INSTRUCTIONS:
1. First, run these commands in separate terminals to start the robot and camera:
   Terminal 1: ros2 launch stretch_core stretch_driver.launch.py
   Terminal 2: ros2 launch stretch_core d435i_low_resolution.launch.py

2. Then run this script directly:
   python3 head_camera_ball_detector.py

3. To detect different objects, modify the TARGET_OBJECTS list in the __init__ method below.

USEFUL DEBUGGING COMMANDS:
ros2 topic list (show all topics)

ros2 topic info "topic_end_point" (get topic metadata)
ros2 topic info /camera/color/image_raw/compressed

ros2 topic echo "topic_end_point" (get actual data)
ros2 topic echo /camera/color/image_raw/compressed

CONFIRMED OBJECT (confidence level):
'sports ball' (50-70%)
'banana' (80-90%)
'apple' (70-90%)
'orange' (70-90%, yolo puts strawberry and lemon as orange)
'backpack'
'bottle'
'cup'
'laptop'
'mouse' (LOW, confused with cell phone)
'remote'
'keyboard' (70%)
'cell phone' (80-90%)
'book' (LOW)


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

class HeadCameraObjDetector(Node):
    def __init__(self):
        super().__init__('head_camera_ball_detector')
        self.USE_BASE_FRAME = False
        
        # CONFIGURABLE: Easily change this to detect different objects!
        # Popular options: 'person', 'cup', 'bottle', 'book', 'laptop', 'cell phone', 'remote'
        # 'chair', 'couch', 'tv', 'bowl', 'banana', 'apple', 'orange', 'sports ball'
        self.TARGET_OBJECTS = ['sports ball', 'banana', 'apple', 'orange', 'backpack', 'bottle', 'cup', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book']  # <-- CHANGE THIS LINE to try different objects

        # The original dimensions of your camera image
        original_w = 240
        original_h = 424

        # 1. Define your desired window width
        window_width = 500

        # 2. Calculate the corresponding height to maintain the aspect ratio
        window_height = int(original_h * (window_width / original_w))  # Result is 543

        # 3. Create and size the window with the calculated dimensions
        cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Object Detection", window_width, window_height)

        # You can do the same for the depth camera window if you uncomment it
        # cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Depth Camera", window_width, window_height)

        self.bridge = CvBridge()
        
        # APPLIED CHANGE: Using the segmentation model for more precise detection
        self.model = YOLO('yolo11s-seg.pt')

        if self.USE_BASE_FRAME:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.camera_info = None
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.sub_image = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # TODO: HOW TO USE OTHER TOPICS LIKE COMPRESSED IMAGE 
        # self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_rect_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.image_depth_callback)

        self.get_logger().info("Object detector node initialized and waiting for messages.")
        self.get_logger().info(f"Currently detecting: {self.TARGET_OBJECTS}")
        self.get_logger().info("To see all available object classes, check the logs after first detection.")

    def print_available_classes(self):
        """Print all available YOLO classes for easy reference"""
        available_classes = list(self.model.names.values())
        self.get_logger().info("Available YOLO object classes:")
        for i, class_name in enumerate(available_classes):
            if i % 10 == 0:  # Print 10 classes per line
                self.get_logger().info(f"  {', '.join(available_classes[i:i+10])}")

    def info_callback(self, msg):
        self.get_logger().info("Original Camera Info Received.")

        # --- Create a corrected CameraInfo for the rotated image ---
        self.corrected_camera_info = msg

        # The image is rotated 90 deg CW, so width and height are swapped.
        original_width = msg.width
        original_height = msg.height
        self.corrected_camera_info.width = original_height
        self.corrected_camera_info.height = original_width

        # The principal point (cx, cy) and focal lengths (fx, fy) also change.
        # Original values from the K matrix:
        fx, cx_k = msg.k[0], msg.k[2]
        fy, cy_k = msg.k[4], msg.k[5]

        # Corrected values after 90 deg CW rotation:
        corrected_fx = fy
        corrected_fy = fx
        # swap and flip for cx_k for 90 deg clockwise
        # -1 for the zero-based counting
        corrected_cx_k = original_height - 1 - cy_k
        corrected_cy_k = cx_k

        # Update the K and P matrices in our corrected_camera_info
        self.corrected_camera_info.k = np.array([corrected_fx, 0.0, corrected_cx_k,
                                                0.0, corrected_fy, corrected_cy_k,
                                                0.0, 0.0, 1.0])
        self.corrected_camera_info.p = np.array([corrected_fx, 0.0, corrected_cx_k, 0.0,
                                                0.0, corrected_fy, corrected_cy_k, 0.0,
                                                0.0, 0.0, 1.0, 0.0])
        # --- End of correction ---

        self.camera_info = self.corrected_camera_info # Use the corrected info from now on
        self.destroy_subscription(self.info_sub)
        self.get_logger().info("Corrected camera info has been generated.")

    def image_depth_callback(self, image_msg, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera_info...")
            return

        img = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # ROTATE IMAGE (orignally 90 degrees rotated counter clockwise)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)

        # Normalize the 16-bit depth data to an 8-bit scale (0-255)
        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply a colormap to make it colorful and easier to see
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        results = self.model(img, verbose=False)[0]
        
        # Print available classes on first run (for user reference)
        if not hasattr(self, '_classes_printed'):
            self.print_available_classes()
            self._classes_printed = True

        # Logic to find the largest detected object
        best_object = None
        largest_area = 0

        # The results from a segmentation model have a .masks attribute
        if results.masks is not None:
            for i, box in enumerate(results.boxes):
                cls_name = results.names[int(box.cls[0])]
                
                # Check if this object is in our target list
                if cls_name in self.TARGET_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > largest_area:
                        largest_area = area
                        best_object = {
                            'box': (x1, y1, x2, y2),
                            'mask': results.masks[i].xy[0],
                            'class': cls_name,
                            'confidence': confidence
                        }
        
        # Process the largest detected object
        if best_object is not None:
            x1, y1, x2, y2 = best_object['box']

            # APPLIED CHANGE: Use the segmentation mask for a highly robust depth calculation
            try:
                # Crop the depth image to the bounding box
                depth_crop = depth_img[y1:y2, x1:x2]
                
                # Create a blank mask of the same size as the crop
                mask_crop = np.zeros(depth_crop.shape, dtype=np.uint8)

                # Get the mask polygon and shift its origin to the crop's top-left corner
                polygon = best_object['mask']
                polygon_shifted = polygon - np.array([x1, y1])

                # Draw the filled polygon on the blank mask
                cv2.fillPoly(mask_crop, [polygon_shifted.astype(np.int32)], 255)

                # Get all valid depth points within the mask
                valid_depths = depth_crop[mask_crop == 255]
                valid_depths = valid_depths[valid_depths > 0] # Filter out zero-depth values



                # Calculate the total number of pixels in the object's mask
                total_mask_pixels = cv2.countNonZero(mask_crop)

                # Avoid division by zero if the mask is empty for some reason
                if total_mask_pixels == 0:
                    return

                # Calculate the percentage of valid pixels
                valid_pixel_percentage = len(valid_depths) / total_mask_pixels
                
                # Set your desired threshold (e.g., 5% = 0.05)
                MIN_VALID_PERCENTAGE = 0.05 

                if valid_pixel_percentage < MIN_VALID_PERCENTAGE:
                    # self.get_logger().info(f"Skipping: Only {valid_pixel_percentage:.2%} valid depth.")
                    return # Exit the function early
                


                # Calculate the median depth
                depth_mm = np.median(valid_depths)
                depth_m = depth_mm / 1000.0

            except Exception as e:
                self.get_logger().error(f"Error in depth calculation: {e}")
                # We draw the image at the end, so just return to skip processing this frame
                return

            # ##############################################################
            # ## 2D to 3D Conversion ##
            # ##############################################################
            # This calculation converts the 2D pixel coordinate of the object's
            # center into a 3D coordinate (X, Y, Z) from the camera's perspective.
            # It's based on the geometry of similar triangles.

            # The core formula is: X (real world width) / Z (depth) = (cx - cx_k) / fx
            # - X: How far the object is to the left or right of the camera's center, in meters.
            # - depth_m: How far the object is straight ahead from the camera, in meters.
            # - (cx - cx_k): The horizontal distance in pixels from the optical center to the object's 2D center.
            # - fx: The camera's horizontal focal length in pixel units.
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            fx, fy = self.camera_info.k[0], self.camera_info.k[4]
            cx_k, cy_k = self.camera_info.k[2], self.camera_info.k[5]
            X = (cx - cx_k) * depth_m / fx
            Y = (cy - cy_k) * depth_m / fy
            Z = depth_m


            # --- Remap coordinates to be more intuitive ---
            # The camera's native optical frame has +Y pointing down.
            # Our OpenCV rotation makes its native +X (right) appear as left.
            # We flip the signs of X and Y to match a standard graph where +Y is up and +X is right.
            x_intuitive = -X
            y_intuitive = -Y
            z_intuitive = Z # Z (depth) is already intuitive, pointing forward.
            # --- End of remapping block ---

            pt_cam = PointStamped()
            pt_cam.header.frame_id = "camera_color_optical_frame"
            pt_cam.header.stamp = rclpy.time.Time().to_msg()
            pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = X, Y, Z
            
            # GET X, Y, and Z from the head camera
            # --- Displaying Camera-Perspective Coordinates ---

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the mask outline for visualization
            cv2.polylines(img, [best_object['mask'].astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

            if self.USE_BASE_FRAME:
                # --- Get and Display Coordinates from the BASE ---
                try:
                    pt_base = self.tf_buffer.transform(pt_cam, 'base_link', timeout=rclpy.duration.Duration(seconds=0.1))
                    px, py, pz = pt_base.point.x, pt_base.point.y, pt_base.point.z
                    
                    text_lines = [f"X (fwd):  {px * 100:.1f} cm", f"Y (left): {py * 100:.1f} cm", f"Z (up):   {pz * 100:.1f} cm"]
                    self.draw_labeled_text_box(img, text_lines, (x1, y1, x2, y2), text_color=(0, 0, 255))
                    self.get_logger().info(f"Ball (base): ({px:.2f}, {py:.2f}, {pz:.2f}) m | Depth: {depth_m:.2f} m")
                except Exception as e:
                    self.get_logger().warn(f"TF transform to 'base_link' failed: {e}")
            else:
                # --- Get and Display Coordinates from the CAMERA ---
                object_name = best_object['class']
                confidence = best_object['confidence']
                text_lines = [
                    f"{object_name} ({confidence:.2f})",
                    f"X (right): {x_intuitive * 100:.1f} cm", 
                    f"Y (up):    {y_intuitive * 100:.1f} cm", 
                    f"Z (fwd):   {z_intuitive * 100:.1f} cm"
                ]
                self.draw_labeled_text_box(img, text_lines, (x1, y1, x2, y2), text_color=(0, 255, 0))
                # self.get_logger().info(f"{object_name} (cam): ({x_intuitive:.2f}, {y_intuitive:.2f}, {z_intuitive:.2f}) m | Depth: {depth_m:.2f} m")

            # --- Draw the depth text (common to both modes) ---
            depth_text_lines = [f"Depth: {depth_mm / 10.0:.1f} cm"]
            self.draw_labeled_text_box(depth_colormap, depth_text_lines, (x1, y1, x2, y2), text_color=(255, 255, 255))


            # # Define the text lines to display, now in centimeters
            # text_lines = [
            #     f"X (right): {x_intuitive * 100:.1f} cm",
            #     f"Y (up):    {y_intuitive * 100:.1f} cm",
            #     f"Z (fwd):   {z_intuitive * 100:.1f} cm",
            # ]
            
            # # Call the helper function to draw the text box
            # self.draw_labeled_text_box(
            #     img=img, 
            #     text_lines=text_lines, 
            #     box=(x1, y1, x2, y2),  # Pass the whole box
            #     text_color=(0, 255, 0)
            # )

            # # Create and draw the depth text, now in centimeters
            # depth_text_lines = [f"Depth: {depth_mm / 10.0:.1f} cm"]
            # self.draw_labeled_text_box(
            #     img=depth_colormap,
            #     text_lines=depth_text_lines,
            #     box=(x1, y1, x2, y2),  # Pass the whole box here too
            #     text_color=(255, 255, 255),
            #     font_scale=0.5
            # )

            
            # # Update the logger to show the NEW intuitive coordinates
            # self.get_logger().info(f"Ball position (intuitive cam): ({x_intuitive:.2f}, {y_intuitive:.2f}, {z_intuitive:.2f}) m | Depth: {depth_m:.2f} m")

            # if self.USE_BASE_FRAME:
            # # GET X, Y, and Z from the base
            #     try:
            #         pt_base = self.tf_buffer.transform(pt_cam, 'base_link', timeout=rclpy.duration.Duration(seconds=1.0))
            #         px, py, pz = pt_base.point.x, pt_base.point.y, pt_base.point.z

            #         # Text for the base_link coordinates
            #         text = f"({px:.2f}, {py:.2f}, {pz:.2f}) m"
            #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            #         # --- ADD THIS to show camera coordinates ---
            #         # Create text for the camera's perspective
            #         text_cam = f"Cam: ({X:.2f}, {Y:.2f}, {Z:.2f}) m"
            #         # Display it slightly below the first text in a different color (cyan)
            #         cv2.putText(img, text_cam, (x1, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            #         # --- End of new code ---

            #         # --- Draw Depth Value on Depth Window ---
            #         # Create the text string with the depth in millimeters
            #         depth_text = f"Depth: {depth_mm:.0f} mm"

            #         # Draw the text on the colorized depth image (using white color)
            #         cv2.putText(depth_colormap, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #         # --- End of New Block ---

            #         # Draw the mask outline for visualization
            #         cv2.polylines(img, [best_ball['mask'].astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
            #         self.get_logger().info(f"Ball position (base_link): {text} | Depth: {depth_m:.2f} m")
            #     except Exception as e:
            #         self.get_logger().warn(f"TF transform failed: {e}")
        

        # resize the image sizes
        # Define your desired display width
        display_width = 960 

        # Get original image dimensions
        h, w, _ = img.shape

        # Calculate the new height to maintain aspect ratio
        display_height = int(h * (display_width / w))

        # Create the new dimensions tuple
        new_dim = (display_width, display_height)

        # Resize both images using the new dimensions and a high-quality filter
        img_display = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        depth_display = cv2.resize(depth_colormap, new_dim, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("YOLO Object Detection", img_display)
        # cv2.imshow("Depth Camera", depth_display)

        cv2.waitKey(1)

    def draw_labeled_text_box(self, img, text_lines, box, font_scale=0.5, font_thickness=2, text_color=(255, 0, 255)):
        """
        Draws a text box that automatically repositions horizontally to stay on screen.
        """
        # Unpack box coordinates and get image dimensions
        x1, y1, x2, _ = box
        img_width = img.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Calculate required width of the text box ---
        max_width = 0
        for line in text_lines:
            (line_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            if line_width > max_width:
                max_width = line_width

        # --- Decide horizontal position (x-coordinate) ---
        # If drawing at x1 would push the text off-screen...
        if x1 + max_width > img_width:
            # ...anchor the text to the right side of the box (x2) instead.
            start_x = x2 - max_width
        else:
            # ...otherwise, anchor it to the left side (x1) as usual.
            start_x = x1
        
        # Vertical position is always 10px above the box's top edge
        start_y = y1 - 10

        # --- Draw the text lines (same logic as before) ---
        (_, text_height), _ = cv2.getTextSize(text_lines[0], font, font_scale, font_thickness)
        line_spacing = text_height + 8
        for i, line in enumerate(reversed(text_lines)):
            y = start_y - (i * line_spacing)
            # Draw a black outline for visibility
            cv2.putText(img, line, (start_x, y), font, font_scale, (0, 0, 0), font_thickness + 2)
            # Draw the colored text on top
            cv2.putText(img, line, (start_x, y), font, font_scale, text_color, font_thickness)

def main(args=None):
    rclpy.init(args=args)
    node = HeadCameraObjDetector()
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
