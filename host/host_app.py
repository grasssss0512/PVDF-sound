#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32-C3 内置 ADC 单通道实时采集上位机 (GUI)

功能：
  * 界面内选择串口 / 波特率，点击「连接」即开始采集
  * 同时实时显示 原始信号 + 滤波后信号 两条波形
  * 实时 FFT 频谱
  * 滤波模式一键切换：ECG/PCG（10~600Hz 带通）或 PPG（0.5~15Hz 带通）
  * 50Hz 陷波（工频）界面可开关
  * 陷波 Q 可调；「自动跟踪工频」锁定实际市电频率，消除频率漂移导致的漏陷
  * 一键保存 CSV 数据

绘图使用 pyqtgraph（实时波形，自带降采样），Qt 绑定优先 PySide6、回退 PyQt5。

串口协议（与固件 main.c 一致）：
    D:v1,v2,...,vN\n    一批原始 12bit 样本 (0~4095)
    其余行（I(...) 等）为日志，上位机自动忽略

用法：
    pip install -r requirements.txt
    python host_app.py [--port COM10] [--rate 2000]
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import serial
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi
from serial.tools import list_ports

# Qt 绑定：优先 PySide6，回退 PyQt5（pyqtgraph 会自动识别）
try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError:
        print("缺少 Qt 绑定（PySide6 或 PyQt5），请先安装依赖:  pip install -r requirements.txt")
        sys.exit(1)

import pyqtgraph as pg

# ---------------------------------------------------------------------------
# 滤波参数
# ---------------------------------------------------------------------------
# 滤波模式预设：界面「滤波模式」下拉框一键切换
# ECG/PCG：心电/心音，10~600Hz 带通
# PPG    ：脉搏波（光电容积），0.5~15Hz 带通（主要能量在 0.5~5Hz，保留重搏切迹）
FILTER_MODES = {
    "ECG/PCG 心电/心音": {"low": 10.0, "high": 600.0, "order": 4},
    "PPG 脉搏":          {"low": 0.5, "high": 15.0, "order": 4},
}
DEFAULT_MODE = "ECG/PCG 心电/心音"

NOTCH_FREQ = 50.0        # 陷波中心频率 Hz（工频）
NOTCH_Q = 20.0           # 陷波品质因数（越小陷波越宽、容错越大；过小会损伤邻近信号）
TRACK_BAND = (45.0, 55.0)   # 自动跟踪时在市频附近的搜索范围 Hz

# 深色主题配色
BG = "#1e1e28"
FG = "#e6e6e6"
COLOR_RAW = "#4fc3f7"    # 原始信号 浅蓝
COLOR_FILT = "#ffb74d"   # 滤波信号 琥珀
COLOR_FFT = "#69f0ae"    # 频谱 绿

pg.setConfigOption("background", BG)
pg.setConfigOption("foreground", FG)
pg.setConfigOptions(antialias=True)


# ---------------------------------------------------------------------------
# 工频估计：在原始信号 FFT 中找 45~55Hz 内的最强峰（抛物线插值提高精度）
# 返回频率值；若峰不够突出（认为没有明显工频）则返回 None
# ---------------------------------------------------------------------------
def estimate_mains_freq(y, fs, band=TRACK_BAND):
    if y is None or len(y) < 128:
        return None
    x = np.asarray(y, dtype=np.float64)
    n = min(len(x), 8192)
    x = x[-n:] - x[-n:].mean()
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    lo, hi = np.searchsorted(f, band[0]), np.searchsorted(f, band[1])
    if hi - lo < 3:
        return None
    band_mag = X[lo:hi]
    k = int(np.argmax(band_mag))
    if band_mag[k] <= np.median(band_mag) * 1.5:   # 峰不突出 -> 无工频
        return None
    kk = lo + k
    # 抛物线插值
    if 0 < kk < len(X) - 1:
        a, b, c = X[kk - 1], X[kk], X[kk + 1]
        den = a - 2 * b + c
        if den != 0:
            kk += 0.5 * (a - c) / den
    return float(np.clip(f[min(int(round(kk)), len(f) - 1)], band[0], band[1]))


# ---------------------------------------------------------------------------
# 串口读取线程：后台解析 D: 行，放入队列（只做解析，不做 GUI）
# ---------------------------------------------------------------------------
class SerialReader(threading.Thread):
    def __init__(self, ser, q):
        super().__init__(daemon=True)
        self.ser = ser
        self.q = q
        self._buf = b''
        self.stopped = threading.Event()
        self.dropped = 0
        self.error = False   # 串口异常（如拔出）时置位

    def run(self):
        while not self.stopped.is_set():
            try:
                data = self.ser.read(4096)
            except serial.SerialException:
                self.error = True
                break
            if not data:
                continue
            self._buf += data
            while b'\n' in self._buf:
                line, self._buf = self._buf.split(b'\n', 1)
                if not line.startswith(b'D:'):
                    continue
                try:
                    vals = [int(x) for x in line[2:].split(b',') if x]
                except ValueError:
                    continue
                if vals:
                    try:
                        self.q.put(vals, timeout=0.1)
                    except queue.Full:
                        self.dropped += len(vals)
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 滤波链路：50Hz 陷波 + 10~600Hz 带通，sosfilt 状态保持，支持动态开关/调Q/重调中心
# ---------------------------------------------------------------------------
class FilterChain:
    def __init__(self, fs):
        self.fs = fs
        self.notch_on = True
        self.bp_on = True
        self.notch_q = NOTCH_Q
        self.notch_freq = NOTCH_FREQ
        self.mode = DEFAULT_MODE
        self.bp_low = 10.0
        self.bp_high = 600.0
        self.bp_order = 4
        self.dc_ref = 0.0     # 输入直流电平参考，用于重建时初始化状态、避免瞬态
        self.apply_preset(DEFAULT_MODE)   # 按默认模式设置带通参数并重建

    def apply_preset(self, mode):
        """切换滤波模式预设（会重置滤波器状态，产生短暂过渡）"""
        m = FILTER_MODES.get(mode)
        if not m:
            return
        self.mode = mode
        self.bp_low, self.bp_high, self.bp_order = m["low"], m["high"], m["order"]
        self.rebuild()

    def rebuild(self):
        """根据当前开关/Q/中心频率/带通参数重新设计滤波器（会重置内部状态）"""
        sos_list = []
        if self.notch_on:
            b, a = iirnotch(self.notch_freq, self.notch_q, self.fs)
            sos_list.append(tf2sos(b, a))
        if self.bp_on:
            sos_list.append(butter(self.bp_order,
                                   [self.bp_low, self.bp_high],
                                   btype='band', fs=self.fs, output='sos'))
        self.sos = np.vstack(sos_list) if sos_list else None
        if self.sos is not None:
            zi = sosfilt_zi(self.sos)
            # 按当前输入直流电平缩放初始状态，避免重建后出现大幅瞬态振荡
            self.zi = zi * self.dc_ref if self.dc_ref != 0.0 else zi
        else:
            self.zi = None

    def retune(self, f0):
        """把陷波中心改为 f0（自动跟踪工频用）；变化过小时不重建，避免频繁重置"""
        f0 = float(np.clip(f0, TRACK_BAND[0], TRACK_BAND[1]))
        if abs(f0 - self.notch_freq) < 0.15:   # 死区：市频测量抖动在此范围内不重建
            return
        self.notch_freq = f0
        self.rebuild()

    def process(self, chunk):
        if self.sos is None:            # 所有滤波都关闭 -> 直通
            return chunk.copy()
        m = float(np.mean(chunk))
        if self.dc_ref == 0.0:
            # 第一次处理：用当前输入直流电平初始化滤波器状态，消除启动瞬态
            self.dc_ref = m
            self.zi = sosfilt_zi(self.sos) * m
        else:
            # 滚动更新直流参考，供下次重建滤波器时初始化状态
            self.dc_ref = 0.9 * self.dc_ref + 0.1 * m
        out, self.zi = sosfilt(self.sos, chunk, zi=self.zi)
        return out

    @property
    def label(self):
        parts = []
        if self.notch_on:
            parts.append(f"陷波@{self.notch_freq:.2f}Hz(Q={self.notch_q:.0f})")
        if self.bp_on:
            parts.append(f"{self.bp_low:g}-{self.bp_high:g}Hz 带通")
        return " + ".join(parts) if parts else "未滤波(直通)"


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, default_port, default_rate):
        super().__init__()
        self.setWindowTitle("ESP32-C3 ADC 实时采集上位机")
        self.resize(1100, 760)

        self.default_port = default_port
        self.fs = float(default_rate)
        self.window_s = 5.0
        self.win = 0
        self.raw_buf = np.zeros(1)
        self.filt_buf = np.zeros(1)
        self.n_avail = 0

        self.q = queue.Queue(maxsize=400)
        self.filt = FilterChain(self.fs)
        self.reader = None
        self.connected = False
        self.rec_path = None
        self.rec_file = None
        self.total = 0
        self.t0 = time.time()
        self.last_track_t = 0.0
        self.last_fft_t = 0.0
        self.no_data_start = None

        self._build_ui()
        self._make_buffers()
        self._refresh_ports()

        # 定时刷新（主线程）：驱动波形/频谱/状态
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

        self.statusBar().showMessage("请选择串口后点击「连接」｜纵轴：滚轮缩放 / 拖动平移 / 右键 View All 复位")

    # ---- 界面搭建 ----------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(6)

        # 控制栏第 1 行：串口 / 采样 / 连接
        bar1 = QtWidgets.QHBoxLayout()
        bar1.setSpacing(8)
        bar1.addWidget(QtWidgets.QLabel("串口"))
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(220)
        bar1.addWidget(self.port_combo)
        self.refresh_btn = QtWidgets.QToolButton()
        self.refresh_btn.setText("⟳")
        self.refresh_btn.setToolTip("刷新串口列表")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        bar1.addWidget(self.refresh_btn)

        bar1.addWidget(QtWidgets.QLabel("波特率"))
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["115200", "460800", "921600"])
        self.baud_combo.setCurrentText("115200")
        bar1.addWidget(self.baud_combo)

        bar1.addWidget(QtWidgets.QLabel("采样率"))
        self.rate_spin = QtWidgets.QSpinBox()
        self.rate_spin.setRange(100, 10000)
        self.rate_spin.setSingleStep(100)
        self.rate_spin.setValue(int(self.fs))
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setToolTip("须与固件 adc_app.h 中 ADC_SAMPLE_RATE_HZ 一致")
        self.rate_spin.valueChanged.connect(self._on_rate_change)
        bar1.addWidget(self.rate_spin)

        bar1.addWidget(QtWidgets.QLabel("窗口"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(1.0, 30.0)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setDecimals(1)
        self.window_spin.setValue(self.window_s)
        self.window_spin.setSuffix(" s")
        self.window_spin.valueChanged.connect(self._on_window_change)
        bar1.addWidget(self.window_spin)

        bar1.addStretch(1)

        self.status_dot = QtWidgets.QLabel("●")
        self.status_dot.setAlignment(QtCore.Qt.AlignCenter)
        bar1.addWidget(self.status_dot)
        self.conn_btn = QtWidgets.QPushButton("连接")
        self.conn_btn.setMinimumWidth(84)
        self.conn_btn.clicked.connect(self._toggle_connect)
        bar1.addWidget(self.conn_btn)
        self.reset_btn = QtWidgets.QPushButton("清屏")
        self.reset_btn.setToolTip("清空波形并重置统计")
        self.reset_btn.clicked.connect(self._reset)
        bar1.addWidget(self.reset_btn)
        self.rec_btn = QtWidgets.QPushButton("录制 CSV")
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self._toggle_record)
        bar1.addWidget(self.rec_btn)

        # 控制栏第 2 行：滤波控制
        bar2 = QtWidgets.QHBoxLayout()
        bar2.setSpacing(10)
        bar2.addWidget(QtWidgets.QLabel("滤波模式"))
        self.mode_combo = QtWidgets.QComboBox()
        for name in FILTER_MODES:
            self.mode_combo.addItem(name)
        self.mode_combo.setCurrentText(DEFAULT_MODE)
        self.mode_combo.setToolTip("一键切换信号类型对应的滤波参数（带通范围）")
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        bar2.addWidget(self.mode_combo)
        bar2.addWidget(QtWidgets.QLabel("滤波:"))
        self.notch_check = QtWidgets.QCheckBox("50Hz 陷波")
        self.notch_check.setChecked(True)
        self.notch_check.toggled.connect(self._on_filter_toggle)
        bar2.addWidget(self.notch_check)
        self.bp_check = QtWidgets.QCheckBox("10-600Hz 带通")
        self.bp_check.setChecked(True)
        self.bp_check.toggled.connect(self._on_filter_toggle)
        bar2.addWidget(self.bp_check)
        bar2.addWidget(QtWidgets.QLabel("陷波Q"))
        self.q_spin = QtWidgets.QDoubleSpinBox()
        self.q_spin.setRange(5.0, 100.0)
        self.q_spin.setSingleStep(1.0)
        self.q_spin.setDecimals(0)
        self.q_spin.setValue(NOTCH_Q)
        self.q_spin.setToolTip("Q 越小陷波越宽（对频率漂移容错越大），越小越伤邻近信号")
        self.q_spin.valueChanged.connect(self._on_q_change)
        bar2.addWidget(self.q_spin)
        self.track_check = QtWidgets.QCheckBox("自动跟踪工频")
        self.track_check.setChecked(True)
        self.track_check.setToolTip("每隔 2s 从原始信号估计实际市电频率，自动对准陷波中心")
        bar2.addWidget(self.track_check)
        self.fft_check = QtWidgets.QCheckBox("FFT 频谱")
        self.fft_check.setChecked(True)
        self.fft_check.toggled.connect(self._apply_fft_visible)
        bar2.addWidget(self.fft_check)
        self.track_label = QtWidgets.QLabel("")
        self.track_label.setStyleSheet(f"color:{COLOR_FFT}; font-weight:bold;")
        bar2.addWidget(self.track_label)
        bar2.addStretch(1)

        root.addLayout(bar1)
        root.addLayout(bar2)

        # 波形区（pyqtgraph：raw / filtered / fft 三行）
        self.pw = pg.GraphicsLayoutWidget()
        self.p_raw = self.pw.addPlot(row=0, col=0)
        self.p_filt = self.pw.addPlot(row=1, col=0)
        self.p_fft = self.pw.addPlot(row=2, col=0)
        self.p_filt.setXLink(self.p_raw)
        self.pw.ci.layout.setRowStretchFactor(0, 12)
        self.pw.ci.layout.setRowStretchFactor(1, 12)
        self.pw.ci.layout.setRowStretchFactor(2, 6)
        root.addWidget(self.pw, 1)

        self._style_plots()
        self._refresh_filter_ui()      # 设置初始带通标签/标题
        self._apply_theme()
        self._apply_fft_visible()

    def _style_plots(self):
        for p in (self.p_raw, self.p_filt, self.p_fft):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getAxis("left").setPen(pg.mkPen(FG))
            p.getAxis("bottom").setPen(pg.mkPen(FG))

        self.curve_raw = self.p_raw.plot(pen=pg.mkPen(COLOR_RAW, width=1))
        self.curve_filt = self.p_filt.plot(pen=pg.mkPen(COLOR_FILT, width=1))
        self.curve_fft = self.p_fft.plot(pen=pg.mkPen(COLOR_FFT, width=1))

        # 关键：峰值降采样 + 只裁剪视图可见范围，保证任意窗口下都流畅
        for c in (self.curve_raw, self.curve_filt, self.curve_fft):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)

        self.p_raw.setTitle("Raw ADC (12-bit)")
        self.p_filt.setTitle(f"Filtered  ({self.filt.label})")
        self.p_raw.setLabel("left", "Raw ADC (12-bit)")
        self.p_raw.setLabel("bottom", "Time (s)")
        self.p_filt.setLabel("left", "Filtered")
        self.p_filt.setLabel("bottom", "Time (s)")
        self.p_fft.setLabel("left", "Mag")
        self.p_fft.setLabel("bottom", "Freq (Hz)")

        # 纵轴均为初始范围，不自动缩放；用户用鼠标滚轮缩放、按住拖动平移，
        # 右键菜单选择 View All 可一键复位到完整数据范围。
        self.p_raw.setYRange(-50, 4145, padding=0)   # 原始信号 12bit 0~4095
        self.p_filt.setYRange(-1000, 1000, padding=0)  # 滤波信号初始范围
        self.p_fft.setYRange(0, 10000, padding=0)   # 频谱初始范围
        self.p_fft.setMouseEnabled(x=False)     # 频谱固定 x 轴

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {BG}; color: {FG}; }}
            QLabel {{ color: {FG}; }}
            QPushButton, QToolButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox {{
                background-color: #2a2a3a; color: {FG};
                border: 1px solid #3d3d55; border-radius: 5px; padding: 3px 8px;
            }}
            QPushButton:hover, QToolButton:hover, QComboBox:hover {{
                border-color: #5a5a80; background-color: #34344a;
            }}
            QPushButton:checked {{ background-color: #2e5d43; border-color: #69f0ae; }}
            QComboBox QAbstractItemView {{
                background-color: #26263a; color: {FG};
                selection-background-color: #3d3d60;
            }}
            QToolTip {{ background-color: #26263a; color: {FG}; border: 1px solid #555; }}
            QStatusBar {{ background: #16161f; color: {FG}; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; }}
            QCheckBox::indicator:checked {{ background-color: #69f0ae; border: 1px solid #69f0ae; border-radius: 3px; }}
        """)

    def _apply_fft_visible(self):
        self.p_fft.setVisible(self.fft_check.isChecked())

    # ---- 串口 ----------------------------------------------------------
    def _refresh_ports(self):
        prev = self.port_combo.currentData()
        self.port_combo.clear()
        for p in list_ports.comports():
            label = p.device
            if p.description:
                label = f"{label}   {p.description}"
            self.port_combo.addItem(label, p.device)
        if self.port_combo.count() > 0:
            idx = self.port_combo.findData(prev if prev else self.default_port)
            self.port_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_combo.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "无法连接", "串口列表为空，请点击 ⟳ 刷新。")
            return
        baud = int(self.baud_combo.currentText())
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
        except serial.SerialException as e:
            QtWidgets.QMessageBox.warning(self, "串口打开失败", f"{e}\n\n请检查端口是否被占用、设备是否已上电。")
            return
        self.reader = SerialReader(ser, self.q)
        self.reader.start()
        self.connected = True
        self.status_dot.setStyleSheet("color:#69f0ae;")
        self.conn_btn.setText("断开")
        self.t0 = time.time()
        self.no_data_start = None
        self.statusBar().showMessage(f"已连接 {port} @ {baud} baud，等待数据…")

    def _disconnect(self):
        if self.reader:
            self.reader.stopped.set()
            self.reader.join(timeout=1.0)
            self.reader = None
        self.connected = False
        self.status_dot.setStyleSheet("color:#666666;")
        self.conn_btn.setText("连接")
        self.statusBar().showMessage("已断开")

    # ---- 数据缓冲 --------------------------------------------------------
    def _make_buffers(self, preserve=True):
        """重建显示缓冲；preserve=True 时保留已有数据（最新样本靠右），
        否则清空（清屏/重置用）。时间轴固定为 [0, window_s]，最新样本锚定右边缘。"""
        new_win = max(int(self.fs * self.window_s), 16)
        if preserve and self.win and self.n_avail > 0:
            keep = min(self.n_avail, new_win)
            old_raw = self.raw_buf[:self.n_avail]
            old_filt = self.filt_buf[:self.n_avail]
        else:
            keep = 0
        self.win = new_win
        self.raw_buf = np.zeros(self.win)
        self.filt_buf = np.zeros(self.win)
        if keep:
            self.raw_buf[:keep] = old_raw[-keep:]
            self.filt_buf[:keep] = old_filt[-keep:]
        self.n_avail = keep
        # 时间轴固定为窗口宽度，避免自动缩放把横轴拉错
        self.p_raw.disableAutoRange(axis='x')
        self.p_raw.setXRange(0, self.window_s, padding=0)
        self._update_plots()

    def _on_rate_change(self, fs):
        self.fs = float(fs)
        self.filt.rebuild()
        self._make_buffers()
        self.p_fft.setXRange(0, self.fs / 2)

    def _on_window_change(self, w):
        self.window_s = float(w)
        self._make_buffers()

    def _on_mode_change(self, mode):
        self.filt.apply_preset(mode)
        self._refresh_filter_ui()

    def _refresh_filter_ui(self):
        """同步界面上的滤波相关文字（带通标签、标题、工频跟踪提示）"""
        self.bp_check.setText(f"{self.filt.bp_low:g}-{self.filt.bp_high:g}Hz 带通")
        self.p_filt.setTitle(f"Filtered  ({self.filt.label})")
        self._update_track_label()

    def _on_filter_toggle(self):
        self.filt.notch_on = self.notch_check.isChecked()
        self.filt.bp_on = self.bp_check.isChecked()
        self.filt.rebuild()
        self._refresh_filter_ui()

    def _on_q_change(self, q):
        self.filt.notch_q = float(q)
        self.filt.rebuild()
        self._refresh_filter_ui()

    def _update_track_label(self):
        if self.filt.notch_on and self.track_check.isChecked() and abs(self.filt.notch_freq - NOTCH_FREQ) >= 0.05:
            self.track_label.setText(f"工频跟踪 @ {self.filt.notch_freq:.2f} Hz")
        else:
            self.track_label.setText("")

    def _reset(self):
        self.total = 0
        self.t0 = time.time()
        self.no_data_start = None
        self.filt.rebuild()
        self._make_buffers(preserve=False)
        self.statusBar().showMessage("已清屏")

    # ---- 录制 ----------------------------------------------------------
    def _toggle_record(self):
        if self.rec_btn.isChecked():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存 CSV", os.path.join(os.getcwd(), "adc_data.csv"), "CSV (*.csv)")
            if not path:
                self.rec_btn.setChecked(False)
                return
            self.rec_path = path
            self.rec_file = open(path, 'w')
            self.rec_file.write("index,raw,filtered,time_ms\n")
            self.statusBar().showMessage(f"录制中 → {path}")
        else:
            if self.rec_file:
                self.rec_file.close()
                self.rec_file = None
            self.statusBar().showMessage(f"已停止录制（{self.rec_path}）")

    # ---- 主循环 ----------------------------------------------------------
    def _tick(self):
        if self.connected and self.reader and self.reader.error:
            self._disconnect()
            QtWidgets.QMessageBox.warning(self, "连接断开", "串口读取出错（设备可能被拔出）。")
            return

        got = False
        while True:
            try:
                vals = self.q.get_nowait()
            except queue.Empty:
                break
            got = True
            arr = np.asarray(vals, dtype=np.float64)
            self._push_chunk(arr, self.filt.process(arr))

        if self.connected and got:
            self.no_data_start = None
            self._update_plots()
            self._maybe_track_mains()
        elif self.connected and not got:
            self._check_no_data()

        self._update_status()

    def _push_chunk(self, arr, f):
        m = len(arr)
        if m >= self.win:
            self.raw_buf[:] = arr[-self.win:]
            self.filt_buf[:] = f[-self.win:]
            self.n_avail = self.win
        elif self.n_avail + m > self.win:
            shift = self.n_avail + m - self.win
            self.raw_buf[:self.win - shift] = self.raw_buf[shift:]
            self.filt_buf[:self.win - shift] = self.filt_buf[shift:]
            self.raw_buf[self.win - m:] = arr
            self.filt_buf[self.win - m:] = f
            self.n_avail = self.win
        else:
            self.raw_buf[self.n_avail:self.n_avail + m] = arr
            self.filt_buf[self.n_avail:self.n_avail + m] = f
            self.n_avail += m
        self.total += m

        if self.rec_file is not None:
            now_ms = (time.time() - self.t0) * 1000
            for i in range(m):
                self.rec_file.write(f"{self.total - m + i},{int(arr[i])},{f[i]:.3f},{now_ms:.1f}\n")

    def _update_plots(self):
        n = self.n_avail
        if n >= 2:
            # 时间轴：最新样本固定在右边缘 x=window_s，更早样本向左排列
            x = self.window_s - (n - 1 - np.arange(n)) / self.fs
            self.curve_raw.setData(x, self.raw_buf[:n])
            self.curve_filt.setData(x, self.filt_buf[:n])
        else:
            self.curve_raw.setData([], [])
            self.curve_filt.setData([], [])

        if self.fft_check.isChecked() and time.time() - self.last_fft_t >= 0.2 and self.n_avail >= 32:
            self.last_fft_t = time.time()
            y = self.filt_buf[max(0, self.n_avail - 4096):self.n_avail]
            y = y - y.mean()
            freqs = np.fft.rfftfreq(len(y), 1.0 / self.fs)
            mag = np.abs(np.fft.rfft(y)) / len(y) * 2
            self.curve_fft.setData(freqs, mag)

    def _maybe_track_mains(self):
        if not (self.filt.notch_on and self.track_check.isChecked()):
            return
        now = time.time()
        if now - self.last_track_t < 2.0:
            return
        self.last_track_t = now
        if self.n_avail >= 128:
            f0 = estimate_mains_freq(self.raw_buf[:self.n_avail], self.fs)
            if f0 is not None:
                self.filt.retune(f0)
                self.p_filt.setTitle(f"Filtered  ({self.filt.label})")
                self._update_track_label()

    def _check_no_data(self):
        if self.no_data_start is None:
            self.no_data_start = time.time()
        elif time.time() - self.no_data_start > 3.0:
            self.statusBar().showMessage(
                "已连接但未收到 D: 数据 — 请确认固件已烧录并运行、信号接 GPIO1、采样率与波特率一致")

    def _update_status(self):
        if not self.connected:
            return
        elapsed = max(1.0, time.time() - self.t0)
        rate = self.total / elapsed
        status = (f"已连接 {self.port_combo.currentData()} | "
                  f"累计 {self.total} | 实时 {rate:.0f} Hz | "
                  f"滤波: {self.filt.label}")
        if self.reader and self.reader.dropped:
            status += f" | 丢弃 {self.reader.dropped}"
        self.statusBar().showMessage(status)

    def closeEvent(self, event):
        self.timer.stop()
        if self.rec_file:
            self.rec_file.close()
        if self.reader:
            self.reader.stopped.set()
            self.reader.join(timeout=1.0)
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser(description="ESP32-C3 ADC 实时采集上位机 (GUI)")
    ap.add_argument("--port", default="COM10", help="默认串口号（界面内可改）")
    ap.add_argument("--rate", type=int, default=2000,
                    help="采样率 Hz，须与固件 ADC_SAMPLE_RATE_HZ 一致（默认 2000）")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ADC Host")
    win = MainWindow(args.port, args.rate)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
