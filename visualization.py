#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import threading
from collections import deque

import numpy as np
if not hasattr(np, "float"): np.float = float
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class RingBuffer:
    def __init__(self, maxlen=6000):
        self.maxlen = maxlen
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)
        self.data = {
            'd_est':   deque(maxlen=maxlen),
            'd_cam':   deque(maxlen=maxlen),
            'd_lidar': deque(maxlen=maxlen),
            's_req':   deque(maxlen=maxlen),
            'v_rel':   deque(maxlen=maxlen),
            'v_cam':   deque(maxlen=maxlen),
            'v_lidar': deque(maxlen=maxlen),
            'ttc':     deque(maxlen=maxlen),
            'brake':   deque(maxlen=maxlen),
        }

    def append(self, t, sample_dict):
        with self.lock:
            self.t.append(t)
            for k in self.data.keys():
                self.data[k].append(sample_dict.get(k, math.nan))

    def snapshot(self):
        with self.lock:
            t = np.array(self.t, dtype=float)
            out = {k: np.array(v, dtype=float) for k, v in self.data.items()}
        return t, out


class AEBPlotNode(Node):
    def __init__(self, buf: RingBuffer, half_deg=10.0):
        super().__init__('aeb_live_plot')
        self.buf = buf
        self.half_rad = math.radians(half_deg)

        self.last = {
            'd_est': math.nan, 'v_rel': math.nan, 'ttc': math.nan,
            's_req': math.nan, 'brake': 0.0, 'v_ego': math.nan,
            'd_cam': math.nan, 'v_cam': math.nan,
            'd_lidar': math.nan
        }

        self.vlp_alpha   = 0.6
        self.vlp_max_abs = 15.0
        self.jump_gate_m = 1.5
        self.prev_dl     = math.nan
        self.prev_tl     = math.nan
        self.v_lidar_ema = 0.0

        self.create_subscription(Float32MultiArray, '/aeb_status', self.cb_aeb, 10)
        self.create_subscription(Float32MultiArray, '/cam_obs', self.cb_cam, 10)
        self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.create_subscription(Odometry, '/odom', self.cb_odom, 10)

        self.create_timer(1.0/20.0, self.tick)

        self.get_logger().info('aeb_live_plot subscribed to /aeb_status, /cam_obs, /scan, /odom')

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cb_aeb(self, msg: Float32MultiArray):
        d = list(msg.data)
        if len(d) >= 1: self.last['d_est'] = float(d[0])
        if len(d) >= 2: self.last['v_rel'] = float(d[1])
        if len(d) >= 3: self.last['ttc']   = float(d[2])
        if len(d) >= 4: self.last['s_req'] = float(d[3])
        if len(d) >= 5: self.last['brake'] = float(d[4])
        if len(d) >= 6: self.last['v_ego'] = float(d[5])

    def cb_cam(self, msg: Float32MultiArray):
        d = list(msg.data)
        if len(d) >= 1: self.last['d_cam'] = float(d[0])
        if len(d) >= 2: self.last['v_cam'] = float(d[1])

    def cb_scan(self, msg: LaserScan):
        dmin = math.inf
        a0 = msg.angle_min
        inc = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            if not (0.05 < r < 60.0):
                continue
            ang = a0 + i * inc
            if -self.half_rad < ang < self.half_rad:
                if r < dmin:
                    dmin = r
        self.last['d_lidar'] = dmin if dmin != math.inf else math.nan

    def cb_odom(self, msg: Odometry):
        self.last['v_ego'] = msg.twist.twist.linear.x

    def _estimate_v_lidar(self, d_lidar: float, t_now: float):
        if not math.isfinite(d_lidar):
            self.prev_dl = math.nan
            self.prev_tl = t_now
            return math.nan

        v_out = math.nan
        if math.isfinite(self.prev_dl) and math.isfinite(self.prev_tl):
            dt = t_now - self.prev_tl
            if 1e-3 < dt < 0.5:
                d_diff = d_lidar - self.prev_dl
                if abs(d_diff) <= self.jump_gate_m:
                    v_raw = -(d_diff / dt)
                    if abs(v_raw) <= self.vlp_max_abs:
                        self.v_lidar_ema = self.vlp_alpha * self.v_lidar_ema + (1.0 - self.vlp_alpha) * v_raw
                        v_out = self.v_lidar_ema
                else:
                    self.v_lidar_ema = 0.0

        self.prev_dl = d_lidar
        self.prev_tl = t_now
        return v_out

    def tick(self):
        t = self.now_s()
        ttc_clip = self.last['ttc']
        if isinstance(ttc_clip, float) and math.isfinite(ttc_clip):
            ttc_clip = max(0.0, min(10.0, ttc_clip))
        else:
            ttc_clip = math.nan
        v_lidar = self._estimate_v_lidar(self.last['d_lidar'], t)
        sample = {
            'd_est':   self.last['d_est'],
            'd_cam':   self.last['d_cam'],
            'd_lidar': self.last['d_lidar'],
            's_req':   self.last['s_req'],
            'v_rel':   self.last['v_rel'],
            'v_cam':   self.last['v_cam'],
            'v_lidar': v_lidar,
            'ttc':     ttc_clip,
            'brake':   1.0 if self.last['brake'] >= 0.5 else 0.0,
        }
        self.buf.append(t, sample)


PALETTE = {
    'd_est':    {'width': 2},
    'd_cam':    {'style': QtCore.Qt.DashLine, 'width': 2},
    'd_lidar':  {'style': QtCore.Qt.DotLine,  'width': 2},
    's_req':    {'width': 2},
    'v_rel':    {'width': 2},
    'v_cam':    {'style': QtCore.Qt.DashLine, 'width': 2},
    'v_lidar':  {'style': QtCore.Qt.DotLine,  'width': 2},
    'ttc':      {'width': 2},
    'mode':     {'width': 2},
}

COLOR = {
    'd_est':   (0, 170, 255),
    'd_cam':   (255, 170, 0),
    'd_lidar': (170, 170, 170),
    's_req':   (255, 0, 0),
    'v_rel':   (0, 200, 0),
    'v_cam':   (200, 120, 0),
    'v_lidar': (120, 120, 200),
    'ttc':     (200, 0, 200),
    'mode':    (50, 220, 255),
}

def mkpen(key):
    kw = PALETTE.get(key, {})
    return pg.mkPen(COLOR.get(key, (255, 255, 255)), **kw)


class LivePanel(QtWidgets.QWidget):
    def __init__(self, buf, time_window_s=30.0, parent=None):
        super().__init__(parent)
        self.buf = buf
        self.time_window_s = time_window_s

        layout = QtWidgets.QVBoxLayout(self)
        pg.setConfigOptions(antialias=True)

        self.p1 = pg.PlotWidget(title='Distance (m)')
        self.p2 = pg.PlotWidget(title='Velocity (m/s)')
        self.p3 = pg.PlotWidget(title='TTC (clipped to 10s)')
        self.p4 = pg.PlotWidget(title='Mode (RUN=0 / BRAKE=1)')

        for p in (self.p1, self.p2, self.p3, self.p4):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.addLegend(offset=(10, 10))
            layout.addWidget(p)

        self.c_d_est   = self.p1.plot(pen=mkpen('d_est'),   name='d_est')
        self.c_d_cam   = self.p1.plot(pen=mkpen('d_cam'),   name='d_cam')
        self.c_d_lidar = self.p1.plot(pen=mkpen('d_lidar'), name='d_lidar')
        self.c_s_req   = self.p1.plot(pen=mkpen('s_req'),   name='s_req')
        self.c_v_rel   = self.p2.plot(pen=mkpen('v_rel'),   name='v_rel (KF)')
        self.c_v_cam   = self.p2.plot(pen=mkpen('v_cam'),   name='v_cam')
        self.c_v_lidar = self.p2.plot(pen=mkpen('v_lidar'), name='v_lidar')
        self.c_ttc     = self.p3.plot(pen=mkpen('ttc'),     name='TTC')
        self.c_mode    = self.p4.plot(pen=mkpen('mode'), name='mode')

        self.last_brake_val = 0
        self.vlines = {p: [] for p in (self.p1, self.p2, self.p3, self.p4)}

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)

        self.t0 = None

    def _add_transition_line(self, t_line):
        for p in (self.p1, self.p2, self.p3, self.p4):
            line = pg.InfiniteLine(pos=t_line, angle=90,
                                   pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine))
            p.addItem(line)
            self.vlines[p].append(line)

    def _trim_vlines(self, t_min):
        for p, lines in self.vlines.items():
            for line in lines[:]:
                if line.pos().x() < t_min:
                    p.removeItem(line)
                    lines.remove(line)

    def _stepify(self, x, y):
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        xs = np.repeat(x, 2)
        ys = np.repeat(y, 2)
        return xs[1:], ys[:-1]

    def set_grid(self, enabled: bool):
        for p in (self.p1, self.p2, self.p3, self.p4):
            p.showGrid(x=enabled, y=enabled, alpha=0.3)

    def auto_range(self):
        for p in (self.p1, self.p2, self.p3, self.p4):
            p.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def refresh(self):
        t, d = self.buf.snapshot()
        if t.size == 0:
            return

        if self.t0 is None:
            self.t0 = t[0]
        tx = t - self.t0
        t_max = tx[-1]
        t_min = max(0.0, t_max - self.time_window_s)
        mask = tx >= t_min

        def sd(key):
            arr = d.get(key, None)
            if arr is None:
                return tx[mask], np.zeros_like(tx[mask])
            return tx[mask], arr[mask]

        self.c_d_est.setData(*sd('d_est'))
        self.c_d_cam.setData(*sd('d_cam'))
        self.c_d_lidar.setData(*sd('d_lidar'))
        self.c_s_req.setData(*sd('s_req'))

        self.c_v_rel.setData(*sd('v_rel'))
        self.c_v_cam.setData(*sd('v_cam'))
        self.c_v_lidar.setData(*sd('v_lidar'))

        tx_ttc, y_ttc = sd('ttc')
        if y_ttc.size:
            y_ttc = np.minimum(y_ttc, 10.0)
        self.c_ttc.setData(tx_ttc, y_ttc)

        t_mode, y_mode = sd('brake')
        sx, sy = self._stepify(t_mode, y_mode)
        self.c_mode.setData(sx, sy)

        if y_mode.size >= 2:
            cur_b = int(y_mode[-1] >= 0.5)
            if cur_b != self.last_brake_val:
                self._add_transition_line(t_mode[-1])
                self.last_brake_val = cur_b

        self._trim_vlines(t_min)

        for p in (self.p1, self.p2, self.p3, self.p4):
            p.setXRange(t_min, t_max, padding=0.01)

        self.p1.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.p2.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.p3.setYRange(0, 10, padding=0.0)
        self.p4.setYRange(-0.2, 1.2, padding=0.0)


class AEBMainWindow(QtWidgets.QMainWindow):
    def __init__(self, buf, time_window_s=30.0):
        super().__init__()
        self.setWindowTitle('AEB Live Monitor')

        self.panel = LivePanel(buf, time_window_s, parent=self)
        self.setCentralWidget(self.panel)

        self.distanceDock = self._make_dock('Distance', self.panel.p1)
        self.velocityDock = self._make_dock('Velocity', self.panel.p2)
        self.ttcDock      = self._make_dock('TTC',      self.panel.p3)
        self.modeDock     = self._make_dock('Mode',     self.panel.p4)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,   self.distanceDock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,  self.velocityDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.ttcDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.modeDock)

        self._build_menus()

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

    def _make_dock(self, title, widget):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title.replace(' ', '_') + 'Dock')
        dock.setWidget(widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable |
                         QtWidgets.QDockWidget.DockWidgetClosable)
        return dock

    def _build_menus(self):
        menubar = self.menuBar()
        viewMenu = menubar.addMenu("&View")
        for dock in (self.distanceDock, self.velocityDock, self.ttcDock, self.modeDock):
            act = dock.toggleViewAction()
            viewMenu.addAction(act)

        actMenu = menubar.addMenu("&Actions")

        actAuto = QtWidgets.QAction("Auto Range All", self)
        actAuto.triggered.connect(self.panel.auto_range)
        actMenu.addAction(actAuto)

        self.gridOn = True
        actGrid = QtWidgets.QAction("Toggle Grid", self)
        actGrid.setCheckable(True)
        actGrid.setChecked(True)
        def _toggle_grid(checked):
            self.gridOn = checked
            self.panel.set_grid(checked)
        actGrid.toggled.connect(_toggle_grid)
        actMenu.addAction(actGrid)


def main():
    rclpy.init()
    buf = RingBuffer(maxlen=12000)
    node = AEBPlotNode(buf, half_deg=10.0)
    exec_ = MultiThreadedExecutor()
    exec_.add_node(node)
    ros_thread = threading.Thread(target=exec_.spin, daemon=True)
    ros_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    win = AEBMainWindow(buf, time_window_s=30.0)
    win.resize(1400, 900)
    win.show()

    ret = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    ros_thread.join(timeout=1.0)
    sys.exit(ret)


if __name__ == '__main__':
    main()

