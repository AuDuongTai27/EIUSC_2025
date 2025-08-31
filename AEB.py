#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class KalmanCV:
    def __init__(self, q_d: float = 0.05, q_v: float = 0.2):
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.q_d, self.q_v = q_d, q_v

    def _valid(self):
        return self.x is not None and np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P))

    def predict(self, dt: float):
        if not self._valid() or not math.isfinite(dt):
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([self.q_d * dt, self.q_v * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _init_distance(self, d: float, R: float):
        self.x = np.array([[d], [0.0]])
        self.P = np.diag([max(R, 1e-6), 5.0])

    def _init_velocity(self, v: float, ego_speed: float, R: float):
        d0 = max(ego_speed, 0.1) * 1.5
        self.x = np.array([[d0], [v]])
        self.P = np.diag([25.0, max(R, 1e-6)])

    def update(self, z: float, H: np.ndarray, R: float, ego_speed: float):
        if not math.isfinite(z):
            return
        if self.x is None:
            if float(H[0, 0]) == 1.0:
                self._init_distance(z, R)
            else:
                self._init_velocity(z, ego_speed, R)
            return
        if not self._valid():
            return
        S = H @ self.P @ H.T + R
        if not np.isfinite(S) or S <= 0:
            return
        K = (self.P @ H.T) / S
        y = np.array([[z]]) - H @ self.x
        self.x += K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    @property
    def distance(self) -> float:
        return float(self.x[0, 0]) if self._valid() else math.inf

    @property
    def v_rel(self) -> float:
        return float(self.x[1, 0]) if self._valid() else 0.0

    @property
    def ttc(self) -> float:
        v = self.v_rel
        return math.inf if v <= 0 else self.distance / v


class AEBFusionNode(Node):
    def __init__(self):
        super().__init__('aeb_fusion_kalman_v2_3')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cam_topic',  '/cam_obs')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic','/drive')
        self.declare_parameter('status_topic','/aeb_status')

        self.declare_parameter('max_scan_gap', 0.5)
        self.declare_parameter('a_max', 5.0)
        self.declare_parameter('buffer', 1.1)
        self.declare_parameter('speed_target', 1.0)

        self.declare_parameter('R_lidar', 1.0)
        self.declare_parameter('R_cam_d', 0.01)
        self.declare_parameter('R_cam_v', 0.7)

        self.declare_parameter('print_status', True)
        self.declare_parameter('print_hz', 2.0)

        self.declare_parameter('brake_on_dwell_ms', 0.0)
        self.declare_parameter('brake_off_dwell_ms', 5000.0)
        self.declare_parameter('brake_min_hold_ms', 3000.0)
        self.declare_parameter('release_factor', 2.0)

        self.scan_topic  = self.get_parameter('scan_topic').value
        self.cam_topic   = self.get_parameter('cam_topic').value
        self.odom_topic  = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.status_topic= self.get_parameter('status_topic').value
        self.max_gap     = self.get_parameter('max_scan_gap').value
        self.a_max       = self.get_parameter('a_max').value
        self.buffer      = self.get_parameter('buffer').value
        self.v_cmd       = self.get_parameter('speed_target').value
        self.Rl          = self.get_parameter('R_lidar').value
        self.Rc_d        = self.get_parameter('R_cam_d').value
        self.Rc_v        = self.get_parameter('R_cam_v').value

        self.print_status = bool(self.get_parameter('print_status').value)
        self.print_hz     = float(self.get_parameter('print_hz').value)

        self.on_dwell = float(self.get_parameter('brake_on_dwell_ms').value)  / 1000.0
        self.off_dwell= float(self.get_parameter('brake_off_dwell_ms').value) / 1000.0
        self.min_hold = float(self.get_parameter('brake_min_hold_ms').value)  / 1000.0
        self.rel_k    = float(self.get_parameter('release_factor').value)

        self.kf = KalmanCV()
        now = self.get_clock().now().nanoseconds * 1e-9
        self.last_time = now
        self.last_scan = now
        self.ego_speed = 0.0
        self.brake = False

        self._on_since = None
        self._off_since = None
        self._last_switch_t = now

        self.last_lidar_d = math.inf
        self.last_cam_d   = math.nan
        self.last_cam_v   = math.nan
        self.last_cam_valid = 0

        self.last_s_req = math.nan
        self.last_cmd   = 0.0
        self.last_mode  = "RUN"

        self.drive_pub   = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.status_pub  = self.create_publisher(Float32MultiArray,      self.status_topic, 10)

        self.create_subscription(LaserScan,         self.scan_topic, self.scan_cb, 10)
        self.create_subscription(Float32MultiArray, self.cam_topic,  self.cam_cb,  10)
        self.create_subscription(Odometry,          self.odom_topic, self.odom_cb, 10)

        if self.print_status and self.print_hz > 0.0:
            period = 1.0 / max(self.print_hz, 0.1)
            self.create_timer(period, self._print_status)

        self.get_logger().info('AEB Fusion v2.3 (LiDAR + Camera depth & v_rel + Debounce) ready.')

    def odom_cb(self, msg: Odometry):
        self.ego_speed = msg.twist.twist.linear.x

    def scan_cb(self, msg: LaserScan):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._predict(now)
        self.last_scan = now

        d_min = math.inf
        a_min = msg.angle_min
        inc   = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            if not (0.1 < r < 50.0):
                continue
            angle = a_min + i * inc
            if -0.174 < angle < 0.174:
                d_min = min(d_min, r)

        if d_min != math.inf:
            self.last_lidar_d = d_min
            self.kf.update(d_min, np.array([[1.0, 0.0]]), self.Rl, self.ego_speed)

        self._decide_publish()

    def cam_cb(self, msg: Float32MultiArray):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._predict(now)

        d_cam = float('nan')
        v_cam = float('nan')
        valid_v = 0

        if len(msg.data) >= 1:
            d_cam = float(msg.data[0])
        if len(msg.data) >= 2:
            v_cam = float(msg.data[1])
        if len(msg.data) >= 3:
            valid_v = int(msg.data[2])

        self.last_cam_d = d_cam
        self.last_cam_v = v_cam
        self.last_cam_valid = valid_v

        if math.isfinite(d_cam) and 0.1 < d_cam < 200.0:
            self.kf.update(d_cam, np.array([[1.0, 0.0]]), self.Rc_d, self.ego_speed)

        if valid_v and math.isfinite(v_cam) and abs(v_cam) < 100.0:
            self.kf.update(v_cam, np.array([[0.0, 1.0]]), self.Rc_v, self.ego_speed)

        self._decide_publish()

    def _predict(self, now: float):
        dt_raw = now - self.last_time
        if not math.isfinite(dt_raw):
            return
        dt = max(min(dt_raw, 0.1), 0.001)
        self.kf.predict(dt)
        self.last_time = now

    def _decide_publish(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        d = self.kf.distance
        v_ego = self.ego_speed
        s_req = v_ego * v_ego / (2 * self.a_max) + self.buffer

        cond_on  = (d <= s_req)
        cond_off = (d >  self.rel_k * s_req)

        if not self.brake:
            if cond_on:
                if self._on_since is None:
                    self._on_since = now
                if (now - self._on_since) >= self.on_dwell:
                    self.brake = True
                    self._last_switch_t = now
                    self._off_since = None
            else:
                self._on_since = None
        else:
            if (now - self._last_switch_t) >= self.min_hold and cond_off:
                if self._off_since is None:
                    self._off_since = now
                if (now - self._off_since) >= self.off_dwell:
                    self.brake = False
                    self._last_switch_t = now
                    self._on_since = None
            else:
                self._off_since = None

        cmd_speed = 0.0 if self.brake else self.v_cmd
        drive = AckermannDriveStamped()
        drive.drive.speed = cmd_speed
        self.drive_pub.publish(drive)

        status = Float32MultiArray()
        status.data = [
            float(d),
            float(self.kf.v_rel),
            float(self.kf.ttc),
            float(s_req),
            1.0 if self.brake else 0.0,
            float(v_ego)
        ]
        self.status_pub.publish(status)

        self.last_s_req = s_req
        self.last_cmd   = cmd_speed
        self.last_mode  = "BRAKE" if self.brake else "RUN"

        self.get_logger().debug(
            f'd={d:.2f} | v_ego={v_ego:.2f} | v_rel={self.kf.v_rel:.2f} | '
            f'TTC={self.kf.ttc:.2f} | s_req={s_req:.2f} | '
            f'mode={self.last_mode} | cmd={cmd_speed:.2f}')

    def _print_status(self):
        def fmt(x, nd=2):
            return "inf" if x == math.inf else ("nan" if not math.isfinite(x) else f"{x:.{nd}f}")

        d_est = self.kf.distance
        v_rel = self.kf.v_rel
        ttc   = self.kf.ttc
        v_ego = max(self.ego_speed, 0.0)
        s_req = (v_ego * v_ego) / (2 * self.a_max) + self.buffer

        self.get_logger().info(
            "[AEB] d_est=%s m | v_rel=%s m/s | TTC=%s s | v_ego=%s m/s | "
            "s_req=%s m | mode=%s | cmd=%s m/s | d_lidar=%s m | "
            "d_cam=%s m | v_cam=%s m/s | valid_v=%d" % (
                fmt(d_est), fmt(v_rel), fmt(ttc), fmt(v_ego),
                fmt(s_req), self.last_mode, fmt(self.last_cmd),
                fmt(self.last_lidar_d),
                fmt(self.last_cam_d), fmt(self.last_cam_v), int(self.last_cam_valid)
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = AEBFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

