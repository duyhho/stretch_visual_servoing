# Notes: Stretch Visual Servoing Q&A

## What range does the gripper grab?

- Original: From 0.05m to 0.085m (5cm to 8.5cm)
- Modified: From 0.02m to 0.085m (2cm to 8.5cm)
- Action plan: test with various object sizes to find optimal range

## When does the arm retract?

- When gripper effort < -14.0 (successful grasp detected)
- When fingertip distance is within the grasp range
- When target error is small enough (< 0.02m)
- All three conditions must be met

## Why sometimes the gripper releases?

- Target error becomes too large (> 0.10m) - "I LOST THE BALL!!!"
- Fingertips get too close together (< 0.038m) - object slipped out
- No object detected for too many frames (10 frames)
- Finger markers not detected for too many frames (10 frames)
- Action plan: improve object tracking and finger detection

## What are three numbers on the screen when running recv_and_yolo_d405_images.py?

X, Y, Z coordinates of the grasp center in centimeters:

- X: Left/Right position (+ = right, - = left)
- Y: Up/Down position (+ = down, - = up)
- Z: Distance from camera (+ = farther away)

Example: "12.3, -4.5, 28.7 cm" means 12.3cm right, 4.5cm up, 28.7cm away

## Can we grasp objects other than balls?

Yes, but with limitations:

- Works best with small, round objects (e.g., apples, oranges)
- Struggles with large, flat objects (e.g., books, boxes)
- May have difficulty with very small or irregularly shaped objects
- Action plan: try grasping various objects and observe performance
- Future: consider adding more object classes and improving depth estimation
