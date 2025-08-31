# Developing An Automatic Emergency Braking System-Based Kalman Filtering For Autonomous Racing Car

## Authors
- Cong-Danh Huynh  
- Nhu-Y Pham  
- Hoang-Dung Bui  
- Duong-Tai Au  
- Duy-Nhat Tran  

## Description
This paper presents the development of an Automatic Emergency Braking (AEB) system for autonomous racing cars,  
using sensor fusion (LiDAR + Camera) with a Kalman filter. The system improves reliability of obstacle distance  
estimation and ensures timely braking decisions for safety in dynamic environments.  

## How to Run
1. **Yolo**  
   - Publish object detection and object tracking with bounding box .
   - Run:
     ```bash
     git clone https://github.com/NhatTran-97/F1Tenth-Racing/tree/main/f1tenth_perception/f1tenth_detection
     ros2 run yolov7_objectDetection.py
     ```
2. **CameraData** 
   - Subscribe depth camera and YOLO results, publish `/cam_obs = [d_cam, v_rel_cam, valid_v]`.  
   - Run:  
     ```bash
     ros2 run your_pkg camera_data.py
     ```
3. **AEBFusionNode**  
   - Fuse LiDAR and Camera observations using Kalman Filter, publish `/aeb_status` and `/drive`.  
   - Run:  
     ```bash
     ros2 run your_pkg AEB.py
     ```
4. **Visulization (Live Monitor GUI)**  
   - Visualize distance, velocity, TTC, and braking mode in real-time.  
   - Run:  
     ```bash
     ros2 run your_pkg visualization.py
     ```
---

