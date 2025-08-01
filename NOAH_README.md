# BEFORE START
stretch_free_robot_process.py
stretch_robot_home.py

## Note: Fixing Camera Rotation

Problem: The head camera image was rotated 90 degrees.

Attempted Fix: Tried to fix it with a custom launch file (start_corrected_camera.launch.py).

Result: This failed because a camera hardware error (Motion Module failure) and other default software were overriding the settings.

Solution: Manually rotated the color and depth images inside the head_camera_ball_detector.py script using OpenCV.

```
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
```

## IMPLEMENTED

Add depth camera open cv window - completed, the depth value is pretty accurate.

what small camera attached to the head camera does? - This smaller camera is an Arducam 1MP RGB camera with a wide-angle lens and global shutter. It is primarily useful when operating the robot remotely, as we will see in an upcoming Web Interface Demo tutorial.


Understand the Coordinate System: The (x, y, z) values displayed on the ball are its position in meters. Investigate how these coordinates relate to the robot's movement. - I changed the code to make the head camera as the origin. These coordinates are relative to the robot's base_link (the center of the robot at the floor), not the camera. We know this because the script transforms the point to the base_link frame.The camera is lotated 90 degrees counter clockwise by default.

## Next Steps

Improve resolution

Improve Performance: To reduce lag, especially over Wi-Fi, research how to use compressed image topics (e.g., subscribing to /camera/color/image_raw/compressed) instead of the raw image stream.


## OPTICAL TERMS
cx_k, cy_k - The Optical Center

fx, fy - The Focal Length

### CameraInfo.K
msg.k[0] is fx

msg.k[2] is cx_k

msg.k[4] is fy

msg.k[5] is cy_k

### CameraInfo.P - similar to CameraInfo.K with stream version