#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
)

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from vision_msgs.msg import YoloResults


def clamp_bbox(x, y, w, h, W, H) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def depth_msg_to_np(msg: Image) -> np.ndarray:
    if msg.encoding not in ('16UC1', '32FC1'):
        raise RuntimeError(f'Unsupported encoding: {msg.encoding}')
    dtype = np.uint16 if msg.encoding == '16UC1' else np.float32
    buf = np.frombuffer(msg.data, dtype=dtype)
    if buf.size != msg.height * msg.width:
        raise RuntimeError(f'Unexpected data size: got {buf.size}, expected {msg.height * msg.width}')
    img = buf.reshape(msg.height, msg.width)
    if msg.is_bigendian:
        img = img.byteswap().newbyteorder()
    if dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    return img.astype(np.float32)

def _fmt(x, nd=2):
    try:
        if x == math.inf:
            return "inf"
        if not (isinstance(x, (float, int)) and math.isfinite(x)):
            return "nan"
        return f"{x:.{nd}f}"
    except Exception:
        return "nan"


class CamObsFromDepthBBox(Node):
    def __init__(self):
        super().__init__('cam_obs_depth_bbox')

        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('yolo_topic', '/yolov7_inference/bounding_box')
        self.declare_parameter('out_topic', '/cam_obs')
        self.declare_parameter('roi_w', 120)
        self.declare_parameter('roi_h', 90)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 80.0)
        self.declare_parameter('min_valid_ratio', 0.05)
        self.declare_parameter('alpha_d', 0.3)
        self.declare_parameter('alpha_v', 0.4)
        self.declare_parameter('max_abs_v', 50.0)
        self.declare_parameter('min_dt', 1e-3)
        self.declare_parameter('cos_longitudinal', True)
        self.declare_parameter('angle_clip_deg', 20.0)
        self.declare_parameter('debug_log', False)
        self.declare_parameter('print_status', True)
        self.declare_parameter('print_hz', 2.0)
        self.declare_parameter('fallback_when_no_bbox', False)
        self.declare_parameter('fallback_mode', 'nan')
        self.declare_parameter('hold_last_ms', 300)
        self.declare_parameter('centerline_w', 120)
        self.declare_parameter('centerline_h', 90)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.yolo_topic = self.get_parameter('yolo_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.roi_w = int(self.get_parameter('roi_w').value)
        self.roi_h = int(self.get_parameter('roi_h').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.min_valid_ratio = float(self.get_parameter('min_valid_ratio').value)
        self.alpha_d = float(self.get_parameter('alpha_d').value)
        self.alpha_v = float(self.get_parameter('alpha_v').value)
        self.max_abs_v = float(self.get_parameter('max_abs_v').value)
        self.min_dt = float(self.get_parameter('min_dt').value)
        self.use_cos_long = bool(self.get_parameter('cos_longitudinal').value)
        self.angle_clip_deg = float(self.get_parameter('angle_clip_deg').value)
        self.debug = bool(self.get_parameter('debug_log').value)
        self.print_status = bool(self.get_parameter('print_status').value)
        self.print_hz = float(self.get_parameter('print_hz').value)
        self.fallback = bool(self.get_parameter('fallback_when_no_bbox').value)
        self.fb_mode  = str(self.get_parameter('fallback_mode').value)
        self.hold_ms  = int(self.get_parameter('hold_last_ms').value)
        self.cl_w     = int(self.get_parameter('centerline_w').value)
        self.cl_h     = int(self.get_parameter('centerline_h').value)

        self.fx = self.fy = self.cx = self.cy = None
        self.depth_img: Optional[np.ndarray] = None
        self.last_d: Optional[float] = None
        self.last_t: Optional[float] = None
        self.d_ema: Optional[float] = None
        self.v_ema: Optional[float] = None
        self.last_bbox = None
        self.last_theta = 0.0
        self.last_good_d: Optional[float] = None
        self.last_good_t: float = 0.0
        self.last_out = (math.nan, math.nan, 0.0)
        self.last_valid_ratio = 0.0
        self.last_roi = (0, 0, 0, 0)

        sensor_best = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        sensor_rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub_obs = self.create_publisher(Float32MultiArray, self.out_topic, 10)
        self.create_subscription(Image,      self.depth_topic, self.depth_cb, sensor_best)
        self.create_subscription(Image,      self.depth_topic, self.depth_cb, sensor_rel)
        self.create_subscription(CameraInfo, self.info_topic,  self.info_cb,  sensor_best)
        self.create_subscription(CameraInfo, self.info_topic,  self.info_cb,  sensor_rel)
        self.create_subscription(YoloResults, self.yolo_topic, self.yolo_cb, 10)

        if self.print_status and self.print_hz > 0:
            period = 1.0 / max(self.print_hz, 0.1)
            self.create_timer(period, self._print_status)

        self.get_logger().info(
            'CamObsFromDepthBBox ready. Publish /cam_obs=[d_cam, v_rel_cam, valid_v] '
            f'| depth="{self.depth_topic}" info="{self.info_topic}" yolo="{self.yolo_topic}"'
        )

    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg: Image):
        try:
            self.depth_img = depth_msg_to_np(msg)
        except Exception as e:
            self.get_logger().error(f'Depth convert error: {e}')
            return
        if self.depth_img is not None:
            if self.last_bbox is None and self.fallback:
                self._fallback_prepare_bbox()
            if self.last_bbox is not None:
                self.process_once(time_stamp=self._stamp_to_sec(msg.header))

    def _fallback_prepare_bbox(self):
        if self.depth_img is None:
            return
        H, W = self.depth_img.shape[:2]
        cx_img = self.cx if self.cx is not None else W / 2.0
        cy_img = self.cy if self.cy is not None else H / 2.0

        if self.fb_mode == 'centerline':
            w = self.cl_w; h = self.cl_h
            x = int(cx_img) - w // 2
            y = int(cy_img) - h // 2
            x, y, w, h = clamp_bbox(x, y, w, h, W, H)
            self.last_bbox = (x, y, w, h, int(cx_img), int(cy_img))
            self.last_theta = 0.0
        elif self.fb_mode == 'hold':
            t_curr = time.time()
            if (self.last_good_d is not None) and ((t_curr - self.last_good_t) * 1000.0 <= self.hold_ms):
                out = Float32MultiArray()
                out.data = [float(self.last_good_d), float('nan'), 0.0]
                self.pub_obs.publish(out)
                self.last_out = (float(self.last_good_d), float('nan'), 0.0)
            else:
                out = Float32MultiArray()
                out.data = [float('nan'), float('nan'), 0.0]
                self.pub_obs.publish(out)
                self.last_out = (mat_

