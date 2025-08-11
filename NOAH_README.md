## BEFORE START
stretch_free_robot_process.py
stretch_robot_home.py

## BEFORE END
stretch_robot_stow.py

## Note: Fixing Camera Rotation
Problem: The head camera image was rotated 90 degrees.

Attempted Fix: Tried to fix it with a custom launch file (start_corrected_camera.launch.py).

Result: This failed because a camera hardware error (Motion Module failure) and other default software were overriding the settings.

Solution: Manually rotated the color and depth images inside the head_camera_ball_detector.py script using OpenCV.

```
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
```

## Note: Reconfigure 3D Conversion
Problem: X,Y, and Z is incorrect (Suspecting the head camera image was rotated 90 degrees)

Solution: Corrected the camera info and flip the signs of X and Y to match a standard graph where +Y is up and +X is right.

```
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
```

```
cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
fx, fy = self.camera_info.k[0], self.camera_info.k[4]
cx_k, cy_k = self.camera_info.k[2], self.camera_info.k[5]
X = (cx - cx_k) * depth_m / fx
Y = (cy - cy_k) * depth_m / fy
Z = depth_m

x_intuitive = -X
y_intuitive = -Y
z_intuitive = Z
```

## IMPLEMENTED
Add depth camera open cv window - completed, the depth value is pretty accurate.

what small camera attached to the head camera does? - This smaller camera is an Arducam 1MP RGB camera with a wide-angle lens and global shutter. It is primarily useful when operating the robot remotely, as we will see in an upcoming Web Interface Demo tutorial.

Understand the Coordinate System: The (x, y, z) values displayed on the ball are its position in meters. Investigate how these coordinates relate to the robot's movement. - I changed the code to make the head camera as the origin. These coordinates are relative to the robot's base_link (the center of the robot at the floor), not the camera. We know this because the script transforms the point to the base_link frame.The camera is lotated 90 degrees counter clockwise by default.

## Future Improvement
Improve resolution

Improve Performance: To reduce lag, especially over Wi-Fi, research how to use compressed image topics (e.g., subscribing to /camera/color/image_raw/compressed) instead of the raw image stream.

Min and Max object detection range
Move Head Camera (min and max for each movements)
Connect with LLM to answer the question (e.g. I'm hungry and please suggest what I can eat in the image -> suggest banana and lemon but recommend banana because the lemon can hurt the empty stomach)

## TERMS
cx_k, cy_k - The Optical Center

fx, fy - The Focal Length

### CameraInfo.K
msg.k[0] is fx

msg.k[2] is cx_k

msg.k[4] is fy

msg.k[5] is cy_k

### CameraInfo.P - similar to CameraInfo.K with stream version