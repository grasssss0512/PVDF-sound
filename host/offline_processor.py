#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线信号处理与回放工具

功能：
  1. 打开 CSV 数据文件（支持 host_app.py 录制格式：index,raw,filtered,time_ms，
     也支持任意一列或多列纯数值 CSV）
  2. 上下两个波形：上为处理前（原始），下为处理后（可对比）
  3. 处理步骤流水线，可自由组合添加/删除：
       带通、陷波、高通、低通、小波去噪、中值滤波
  4. 导出 WAV，让用户直接听处理后的心音效果（导出后自动用系统播放器打开）

依赖：
    pip install -r requirements.txt

用法：
    python offline_processor.py
"""
import argparse
import csv
import os
import sys
import wave

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt, medfilt

try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError:
        print("缺少 Qt 绑定（PySide6 或 PyQt5），请先安装依赖:  pip install -r requirements.txt")
        sys.exit(1)

import pywt
import pyqtgraph as pg

# ---------------------------------------------------------------------------
# 配色（与实时上位机一致）
# ---------------------------------------------------------------------------
BG = "#1e1e28"
FG = "#e6e6e6"
COLOR_ORIG = "#4fc3f7"    # 处理前 浅蓝
COLOR_PROC = "#ffb74d"    # 处理后 琥珀

pg.setConfigOption("background", BG)
pg.setConfigOption("foreground", FG)
pg.setConfigOptions(antialias=True)

# ---------------------------------------------------------------------------
# 处理步骤定义
# ---------------------------------------------------------------------------
STEP_TYPES = {
    "bandpass": {"label": "带通", "params": {"low": 20.0, "high": 600.0, "order": 4}},
    "notch":    {"label": "陷波", "params": {"freq": 50.0, "q": 20.0}},
    "highpass": {"label": "高通", "params": {"cutoff": 20.0, "order": 4}},
    "lowpass":  {"label": "低通", "params": {"cutoff": 500.0, "order": 4}},
    "impulse":  {"label": "脉冲去除", "params": {"window": 41, "k": 6.0, "max_len": 15}},
    "mains":    {"label": "工频抑制", "params": {"base": 50.0, "q": 30.0, "harmonics": 3}},
    "wavelet":  {"label": "小波去噪", "params": {"wavelet": "db4", "level": 0, "mode": "soft"}},
    "median":   {"label": "中值滤波", "params": {"size": 5}},
}
COMBO_OPTS = {
    "wavelet": ["db4", "db8", "sym4", "sym8", "coif2", "haar"],
    "mode":    ["soft", "hard"],
}


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------
def load_signal_csv(path):
    """读取 CSV，返回 (数据数组 shape(N,C), 列名列表或 None)。

    自动识别首行是否为表头；跳过无法转成数字的行。
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        raise ValueError("CSV 文件为空")

    # 判断表头：首行前两列是否为数字
    try:
        float(rows[0][0])
        float(rows[0][1] if len(rows[0]) > 1 else rows[0][0])
        has_header = False
    except ValueError:
        has_header = True

    cols = rows[0] if has_header else None
    data = []
    for row in (rows[1:] if has_header else rows):
        try:
            data.append([float(x) for x in row])
        except ValueError:
            continue
    if not data:
        raise ValueError("CSV 中没有可用的数值数据")
    return np.array(data), cols


# ---------------------------------------------------------------------------
# 滤波处理
# ---------------------------------------------------------------------------
def wavelet_denoise(sig, wavelet="db4", level=0, mode="soft"):
    """小波去噪：wavedec -> 通用阈值收缩细节系数 -> waverec。

    level=0 表示按信号长度自动取最大可分解层数（上限 8）。
    """
    import pywt
    w = pywt.Wavelet(wavelet)
    if level <= 0:
        level = min(pywt.dwt_max_level(len(sig), w.dec_len), 8)
    level = max(level, 1)
    coeffs = pywt.wavedec(sig, w, level=level)
    # 用最细尺度系数估计噪声标准差（MAD 稳健估计）
    cD = coeffs[-1]
    sigma = np.median(np.abs(cD - np.median(cD))) / 0.6745
    thr = sigma * np.sqrt(2.0 * np.log(max(len(sig), 2)))
    thr = max(thr, 1e-9)
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(new_coeffs, w)
    return rec[:len(sig)]


def remove_impulses(sig, window=41, k=6.0, max_len=15):
    """脉冲（碰撞噪声）去除：稳健检测 + 线性插值替换。

    用中值滤波算出基线，残差偏离 k·MAD 以上、且连续长度不超过 max_len
    的样本段判定为脉冲，仅替换这些段（线性插值），其余样本原样保留。
    """
    from scipy.ndimage import median_filter
    x = np.asarray(sig, dtype=np.float64)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    baseline = median_filter(x, size=window, mode="nearest")
    residual = x - baseline
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        return x.copy()
    thr = float(k) * sigma
    mask = np.abs(residual) > thr
    if not mask.any():
        return x.copy()

    out = x.copy()
    n = len(x)
    start = 0
    while start < n:
        if not mask[start]:
            start += 1
            continue
        end = start
        while end < n and mask[end]:
            end += 1
        run_len = end - start
        if run_len <= int(max_len):
            # 找两侧最近的干净样本，做线性插值
            a = start - 1
            while a >= 0 and mask[a]:
                a -= 1
            b = end
            while b < n and mask[b]:
                b += 1
            if a >= 0 and b < n:
                out[start:end] = np.interp(
                    np.arange(start, end, dtype=float), [a, b], [x[a], x[b]])
            elif a >= 0:
                out[start:end] = x[a]
            elif b < n:
                out[start:end] = x[b]
        start = end
    return out


def estimate_mains_freq(sig, fs, base=50.0, band_half=5.0):
    """从整段信号实测市电频率：加窗 rfft + 抛物线插值。"""
    x = np.asarray(sig, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 16:
        return float(base)
    X = np.fft.rfft(x * np.hanning(n))
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    lo, hi = base - band_half, base + band_half
    m = (fr >= lo) & (fr <= hi)
    if not m.any():
        return float(base)
    idx = np.where(m)[0]
    k = int(idx[np.argmax(np.abs(X[idx]))])
    if k <= 0 or k >= len(X) - 1:
        return float(fr[k])
    y0, y1, y2 = np.abs(X[k - 1]), np.abs(X[k]), np.abs(X[k + 1])
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    return float(fr[k] + delta * (fr[k + 1] - fr[k]))


def mains_cascade_notch(sig, fs, base=50.0, q=30.0, harmonics=3):
    """实测工频 + 谐波级联陷波。返回 (处理结果, 实测频率, 已陷波频率列表)。

    对 f0/2f0/3f0... 逐阶陷波；谐波带宽按 q*h 收紧，保持绝对带宽一致，
    避免误伤心音邻频信号。
    """
    f0 = estimate_mains_freq(sig, fs, base=base)
    f0 = float(np.clip(f0, 0.5 * base, 1.5 * base))
    sos_list, freqs = [], []
    for h in range(1, int(harmonics) + 1):
        f = h * f0
        if f >= 0.45 * fs:
            break
        b, a = iirnotch(f, q * h, fs)
        sos_list.append(tf2sos(b, a))
        freqs.append(f)
    if not sos_list:
        return sig.copy(), f0, freqs
    return sosfiltfilt(np.vstack(sos_list), sig), f0, freqs


def apply_step(sig, fs, step):
    """对信号应用单个处理步骤（离线零相位滤波）"""
    t = step["type"]
    p = step["params"]
    if t == "bandpass":
        sos = butter(int(p["order"]), [p["low"], p["high"]], btype="band", fs=fs, output="sos")
        return sosfiltfilt(sos, sig)
    if t == "notch":
        b, a = iirnotch(p["freq"], p["q"], fs)
        return sosfiltfilt(tf2sos(b, a), sig)
    if t == "highpass":
        sos = butter(int(p["order"]), p["cutoff"], btype="high", fs=fs, output="sos")
        return sosfiltfilt(sos, sig)
    if t == "lowpass":
        sos = butter(int(p["order"]), p["cutoff"], btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, sig)
    if t == "wavelet":
        return wavelet_denoise(sig, p["wavelet"], int(p["level"]), p["mode"])
    if t == "median":
        k = int(p["size"])
        if k % 2 == 0:
            k += 1
        return medfilt(sig, k) if k >= 3 else sig.copy()
    if t == "impulse":
        return remove_impulses(sig, p["window"], p["k"], p["max_len"])
    if t == "mains":
        out, f0, freqs = mains_cascade_notch(sig, fs, p["base"], p["q"], p["harmonics"])
        step["info"] = "工频≈%.2fHz→陷波%sHz" % (
            f0, "/".join("%.1f" % f for f in freqs))
        return out
    return sig.copy()


def apply_pipeline(sig, fs, steps):
    """按顺序应用所有处理步骤"""
    out = np.asarray(sig, dtype=np.float64).copy()
    for st in steps:
        out = apply_step(out, fs, st)
    return out


# ---------------------------------------------------------------------------
# WAV 导出
# ---------------------------------------------------------------------------
def export_wav(sig, fs, path):
    """把信号写成 16bit 单声道 WAV（归一化到 0.9 峰值，避免削波）"""
    y = np.nan_to_num(np.asarray(sig, dtype=np.float64))
    if y.size == 0:
        raise ValueError("信号为空")
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        peak = 1.0
    pcm = (np.clip(y / peak * 0.9, -0.9999, 0.9999) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# CSV / 波形图导出
# ---------------------------------------------------------------------------
def export_csv(sig, fs, path, t0=None, t1=None, original=None, source="both"):
    """把信号导出为 CSV（可选时间窗口）。

    t0/t1 为秒，None 表示全段。
    source: "both" 原始+处理后两列（默认），"proc" 仅处理后，"orig" 仅原始。
    """
    n = len(sig)
    t = np.arange(n) / fs
    if t0 is None:
        t0 = float(t[0])
    if t1 is None:
        t1 = float(t[-1])
    m = (t >= t0) & (t <= t1)
    if m.sum() < 1:
        raise ValueError("时间窗口内没有数据（t0=%.3f, t1=%.3f）" % (t0, t1))
    tt = t[m]
    ss = np.asarray(sig)[m]
    oo = np.asarray(original)[m] if original is not None else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if source == "orig" and oo is not None:
            w.writerow(["index", "time_s", "original"])
            for i in range(len(tt)):
                w.writerow([i, "%.6f" % tt[i], "%.6f" % oo[i]])
        elif source == "proc":
            w.writerow(["index", "time_s", "processed"])
            for i in range(len(tt)):
                w.writerow([i, "%.6f" % tt[i], "%.6f" % ss[i]])
        else:  # both（默认）
            if oo is not None:
                header, cols = ["index", "time_s", "original", "processed"], [oo, ss]
            else:
                header, cols = ["index", "time_s", "processed"], [ss]
            w.writerow(header)
            for i in range(len(tt)):
                w.writerow([i, "%.6f" % tt[i]] + ["%.6f" % c[i] for c in cols])
    return path


def export_waveform_png(sig, fs, path, t0=None, t1=None, ymin=None, ymax=None,
                        title="", xlabel="Time (s)", ylabel="Amplitude", original=None):
    """导出 Origin 风格波形图：白色背景、黑色四边盒框、蓝色曲线、浅色网格。

    t0/t1 指定时间窗口（None=全长），ymin/ymax 指定 Y 幅值窗口（None=自动）。
    original 不为 None 时把原始信号（去直流后灰色细线）叠加对比。
    用 matplotlib 的 Agg 后端，纯静态渲染，不影响界面。
    """
    import matplotlib
    matplotlib.use("Agg")          # 无窗口后端，避免与 Qt 冲突
    import matplotlib.pyplot as plt
    t = np.arange(len(sig)) / fs
    if t0 is None:
        t0 = float(t[0])
    if t1 is None:
        t1 = float(t[-1])
    x0, x1 = max(t0, float(t[0])), min(t1, float(t[-1]))
    m = (t >= x0) & (t <= x1)
    if m.sum() < 2:
        raise ValueError("时间窗口内数据不足（t0=%.3f, t1=%.3f）" % (t0, t1))
    tt, yy = t[m], np.asarray(sig)[m]
    try:
        plt.rcParams["font.family"] = "Arial"
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if original is not None:
        oo = np.asarray(original, dtype=float)[m]
        oo = oo - np.mean(oo)      # 去直流，便于与处理后同轴对比
        ax.plot(tt, oo, color="#a0a5ad", lw=0.9, label="Raw (detrended)")
        ax.plot(tt, yy, color="#1a4fd8", lw=1.3, label="Processed")
        ax.legend(fontsize=10, frameon=False)
    else:
        ax.plot(tt, yy, color="#1a4fd8", lw=1.3)
    # 四边黑色盒框
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(1.0)
    # 刻度朝内（Origin 风格），主/次刻度
    ax.tick_params(direction="in", which="both", labelsize=10, width=1.0)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.minorticks_on()
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(x0, x1)
    if ymin is not None and ymax is not None and ymax > ymin:
        ax.set_ylim(ymin, ymax)
    else:
        ax.autoscale(axis="y")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("离线信号处理与回放")
        self.resize(1240, 820)

        self.path = None
        self.data = None       # 完整 CSV 数据 (N, C)
        self.cols = None       # 列名
        self.sig = None        # 当前选中的原始信号
        self.fs = 2000.0
        self.proc = None       # 处理后的信号
        self.steps = []        # [{"type","params"}, ...]

        self._build_ui()
        self.statusBar().showMessage("打开一个 CSV 文件开始 ｜ 纵轴：滚轮缩放 / 拖动平移 / 右键 View All 复位")

    # ---- 界面 ----
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(6)

        # 行1：文件 + 采样率 + 数据列
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)
        self.open_btn = QtWidgets.QPushButton("打开 CSV")
        self.open_btn.clicked.connect(self._open_file)
        row1.addWidget(self.open_btn)
        self.path_label = QtWidgets.QLabel("（未打开文件）")
        self.path_label.setStyleSheet("color:#888;")
        row1.addWidget(self.path_label, 1)
        row1.addWidget(QtWidgets.QLabel("采样率 Hz"))
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(100, 10000)
        self.fs_spin.setSingleStep(100)
        self.fs_spin.setDecimals(0)
        self.fs_spin.setValue(self.fs)
        self.fs_spin.valueChanged.connect(lambda v: setattr(self, "fs", float(v)))
        row1.addWidget(self.fs_spin)
        row1.addWidget(QtWidgets.QLabel("数据列"))
        self.col_combo = QtWidgets.QComboBox()
        self.col_combo.currentIndexChanged.connect(self._on_col_change)
        row1.addWidget(self.col_combo)
        root.addLayout(row1)

        # 行2：处理步骤 + 参数
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("处理步骤（按顺序执行）"))
        self.step_list = QtWidgets.QListWidget()
        self.step_list.setMinimumWidth(300)
        self.step_list.currentRowChanged.connect(self._rebuild_param_panel)
        left.addWidget(self.step_list, 1)
        btns = QtWidgets.QHBoxLayout()
        self.add_type_combo = QtWidgets.QComboBox()
        for k, v in STEP_TYPES.items():
            self.add_type_combo.addItem(v["label"], k)
        btns.addWidget(self.add_type_combo)
        self.add_btn = QtWidgets.QPushButton("添加")
        self.add_btn.clicked.connect(self._add_step)
        self.del_btn = QtWidgets.QPushButton("删除")
        self.del_btn.clicked.connect(self._del_step)
        self.up_btn = QtWidgets.QPushButton("上移")
        self.up_btn.clicked.connect(lambda: self._move_step(-1))
        self.down_btn = QtWidgets.QPushButton("下移")
        self.down_btn.clicked.connect(lambda: self._move_step(1))
        for b in (self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            btns.addWidget(b)
        left.addLayout(btns)
        row2.addLayout(left, 0)

        # 参数面板
        param_box = QtWidgets.QVBoxLayout()
        param_box.addWidget(QtWidgets.QLabel("参数（选中步骤）"))
        self.param_panel = QtWidgets.QWidget()
        self.param_form = QtWidgets.QFormLayout(self.param_panel)
        self.param_form.setContentsMargins(4, 4, 4, 4)
        param_box.addWidget(self.param_panel, 1)
        row2.addLayout(param_box, 1)

        root.addLayout(row2, 0)

        # 行3：应用 + 导出
        row3 = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("应用处理")
        self.apply_btn.setToolTip("对当前数据列按步骤顺序执行全部滤波")
        self.apply_btn.clicked.connect(self._apply)
        self.export_btn = QtWidgets.QPushButton("导出 WAV 试听")
        self.export_btn.setToolTip("把处理后信号写成 WAV，并用系统默认播放器打开")
        self.export_btn.clicked.connect(self._export)
        self.export_csv_btn = QtWidgets.QPushButton("导出 CSV")
        self.export_csv_btn.setToolTip("把处理后（含原始列）数据导出为 CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_png_btn = QtWidgets.QPushButton("导出波形图")
        self.export_png_btn.setToolTip("导出 Origin 风格波形图（白底蓝线 PNG）")
        self.export_png_btn.clicked.connect(self._export_png)
        row3.addWidget(self.apply_btn)
        row3.addWidget(self.export_btn)
        row3.addWidget(self.export_csv_btn)
        row3.addWidget(self.export_png_btn)
        row3.addStretch(1)
        root.addLayout(row3)

        # 行3.5：导出范围设置（时间窗口 / Y幅值窗口 / 同步视图）
        row3b = QtWidgets.QHBoxLayout()
        row3b.setSpacing(6)
        row3b.addWidget(QtWidgets.QLabel("导出范围:"))
        self.time_chk = QtWidgets.QCheckBox("时间窗")
        self.time_chk.setToolTip("勾选后按下方起止时间导出/出图，不勾选则用全段")
        self.time_chk.toggled.connect(self._on_range_toggle)
        row3b.addWidget(self.time_chk)
        self.t0_spin = QtWidgets.QDoubleSpinBox()
        self.t0_spin.setRange(0, 1e6)
        self.t0_spin.setDecimals(3)
        self.t0_spin.setValue(0.0)
        self.t1_spin = QtWidgets.QDoubleSpinBox()
        self.t1_spin.setRange(0, 1e6)
        self.t1_spin.setDecimals(3)
        self.t1_spin.setValue(0.0)
        for sp in (self.t0_spin, self.t1_spin):
            sp.setEnabled(False)
        row3b.addWidget(self.t0_spin)
        row3b.addWidget(QtWidgets.QLabel("~"))
        row3b.addWidget(self.t1_spin)
        row3b.addWidget(QtWidgets.QLabel("s"))
        self.y_chk = QtWidgets.QCheckBox("Y范围")
        self.y_chk.setToolTip("勾选后按下方 Ymin~Ymax 出图，不勾选则自动")
        self.y_chk.toggled.connect(self._on_range_toggle)
        row3b.addWidget(self.y_chk)
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setRange(-1e6, 1e6)
        self.ymin_spin.setDecimals(1)
        self.ymin_spin.setValue(-1000.0)
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setRange(-1e6, 1e6)
        self.ymax_spin.setDecimals(1)
        self.ymax_spin.setValue(1000.0)
        for sp in (self.ymin_spin, self.ymax_spin):
            sp.setEnabled(False)
        row3b.addWidget(self.ymin_spin)
        row3b.addWidget(QtWidgets.QLabel("~"))
        row3b.addWidget(self.ymax_spin)
        self.sync_btn = QtWidgets.QPushButton("同步视图")
        self.sync_btn.setToolTip("把下方「处理后」波形当前可见范围填入导出设置")
        self.sync_btn.clicked.connect(self._sync_from_view)
        row3b.addWidget(self.sync_btn)
        row3b.addSpacing(12)
        row3b.addWidget(QtWidgets.QLabel("内容"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setToolTip(
            "导出内容（同时作用于 CSV 与波形图）：两者对比 / 处理后 / 原始（滤波前）")
        self.src_combo.addItem("两者对比", "both")
        self.src_combo.addItem("处理后", "proc")
        self.src_combo.addItem("原始（滤波前）", "orig")
        row3b.addWidget(self.src_combo)
        row3b.addStretch(1)
        root.addLayout(row3b)

        # 行4：波形（上=处理前 下=处理后）
        self.pw = pg.GraphicsLayoutWidget()
        self.p_orig = self.pw.addPlot(row=0, col=0)
        self.p_proc = self.pw.addPlot(row=1, col=0)
        self.p_proc.setXLink(self.p_orig)
        self.pw.ci.layout.setRowStretchFactor(0, 1)
        self.pw.ci.layout.setRowStretchFactor(1, 1)
        root.addWidget(self.pw, 1)

        self.curve_orig = self.p_orig.plot(pen=pg.mkPen(COLOR_ORIG, width=1))
        self.curve_proc = self.p_proc.plot(pen=pg.mkPen(COLOR_PROC, width=1))
        for p, color, name in ((self.p_orig, COLOR_ORIG, "处理前（原始）"),
                               (self.p_proc, COLOR_PROC, "处理后")):
            p.setTitle(name)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getAxis("left").setPen(pg.mkPen(FG))
            p.getAxis("bottom").setPen(pg.mkPen(FG))
        self.p_orig.setLabel("left", "Amplitude")
        self.p_orig.setLabel("bottom", "Time (s)")
        self.p_proc.setLabel("left", "Amplitude")
        self.p_proc.setLabel("bottom", "Time (s)")
        # 纵轴手动控制：滚轮缩放/拖动平移
        self.p_orig.setYRange(-1.0, 1.0, padding=0)
        self.p_proc.setYRange(-1.0, 1.0, padding=0)

        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {BG}; color: {FG}; }}
            QLabel {{ color: {FG}; }}
            QPushButton, QComboBox, QDoubleSpinBox, QListWidget {{
                background-color: #2a2a3a; color: {FG};
                border: 1px solid #3d3d55; border-radius: 5px; padding: 3px 8px;
            }}
            QPushButton:hover, QComboBox:hover {{ border-color: #5a5a80; background-color: #34344a; }}
            QListWidget::item:selected {{ background-color: #3d3d60; }}
            QComboBox QAbstractItemView {{
                background-color: #26263a; color: {FG};
                selection-background-color: #3d3d60;
            }}
            QStatusBar {{ background: #16161f; color: {FG}; }}
        """)

    # ---- 文件 ----
    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 CSV 数据文件", os.getcwd(), "CSV 文件 (*.csv);;所有文件 (*)")
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path):
        try:
            data, cols = load_signal_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "读取失败", str(e))
            return
        self.path = path
        self.data = data
        self.cols = cols
        self.proc = None

        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        ncol = data.shape[1]
        if cols:
            for i, name in enumerate(cols[:ncol]):
                self.col_combo.addItem(f"{name}  (列{i})", i)
        else:
            for i in range(ncol):
                self.col_combo.addItem(f"列{i}", i)
        # 默认选 raw / 第1列
        idx = 0
        if cols:
            for i, name in enumerate(cols):
                if name.strip().lower() in ("raw", "signal", "data"):
                    idx = i
                    break
        self.col_combo.setCurrentIndex(min(idx, self.col_combo.count() - 1))
        self.col_combo.blockSignals(False)
        self._on_col_change(self.col_combo.currentIndex())

        self.path_label.setText(os.path.basename(path))
        self.path_label.setStyleSheet(f"color:{FG};")
        self.statusBar().showMessage(f"已加载 {path}  （{data.shape[0]} 行 × {data.shape[1]} 列）")

    def _on_col_change(self, idx):
        if self.data is None or idx < 0:
            return
        self.sig = self.data[:, idx]
        self.proc = None
        self._plot()

    # ---- 步骤管理 ----
    def _step_text(self, step):
        p = step["params"]; t = step["type"]
        if t == "bandpass":
            return f"{STEP_TYPES[t]['label']} {p['low']:g}-{p['high']:g}Hz  阶{p['order']}"
        if t == "notch":
            return f"{STEP_TYPES[t]['label']} {p['freq']:g}Hz  Q{p['q']:g}"
        if t == "highpass":
            return f"{STEP_TYPES[t]['label']} >{p['cutoff']:g}Hz  阶{p['order']}"
        if t == "lowpass":
            return f"{STEP_TYPES[t]['label']} <{p['cutoff']:g}Hz  阶{p['order']}"
        if t == "wavelet":
            return f"{STEP_TYPES[t]['label']} {p['wavelet']} L{p['level']} {p['mode']}"
        if t == "median":
            return f"{STEP_TYPES[t]['label']} 窗{p['size']}"
        if t == "impulse":
            return f"{STEP_TYPES[t]['label']} 窗{p['window']}·k{p['k']:g}·限{p['max_len']}点"
        if t == "mains":
            info = step.get("info", "")
            base = f"Q{p['q']:g}·{p['harmonics']}阶谐波"
            return f"{STEP_TYPES[t]['label']} {base}" + (f"（{info}）" if info else "")
        return STEP_TYPES[t]["label"]

    def _refresh_steps(self):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for i, st in enumerate(self.steps):
            self.step_list.addItem(f"{i + 1}. {self._step_text(st)}")
        self.step_list.blockSignals(False)
        if self.step_list.count() > 0:
            self.step_list.setCurrentRow(self.step_list.count() - 1)
        else:
            self._rebuild_param_panel()

    def _add_step(self):
        t = self.add_type_combo.currentData()
        self.steps.append({"type": t, "params": dict(STEP_TYPES[t]["params"])})
        self._refresh_steps()
        self._update_step_item(0)

    def _del_step(self):
        i = self.step_list.currentRow()
        if 0 <= i < len(self.steps):
            self.steps.pop(i)
            self._refresh_steps()

    def _move_step(self, d):
        i = self.step_list.currentRow()
        j = i + d
        if 0 <= i < len(self.steps) and 0 <= j < len(self.steps):
            self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
            self._refresh_steps()
            self.step_list.setCurrentRow(j)

    def _update_step_item(self, idx):
        if 0 <= idx < len(self.steps):
            self.step_list.item(idx).setText(f"{idx + 1}. {self._step_text(self.steps[idx])}")

    # ---- 参数面板 ----
    def _rebuild_param_panel(self, *_):
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        i = self.step_list.currentRow()
        if i < 0 or i >= len(self.steps):
            return
        step = self.steps[i]
        for key, val in step["params"].items():
            if key in COMBO_OPTS:
                w = QtWidgets.QComboBox()
                w.addItems(COMBO_OPTS[key])
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(
                    lambda s, k=key: self._param_changed(i, k, s))
            elif isinstance(val, float):
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(0.01, 5000.0)
                w.setDecimals(1)
                w.setValue(float(val))
                w.valueChanged.connect(
                    lambda v, k=key: self._param_changed(i, k, float(v)))
            else:
                w = QtWidgets.QSpinBox()
                if key == "level":
                    w.setRange(0, 16)
                    w.setToolTip("0 = 自动选择层数")
                elif key == "order":
                    w.setRange(1, 8)
                elif key == "harmonics":
                    w.setRange(1, 6)
                    w.setToolTip("级联陷波阶数：1=仅基波；3=同时陷 f0/2f0/3f0（约50/100/150Hz）")
                elif key == "window":
                    w.setRange(3, 201)
                    w.setSingleStep(2)
                    w.setToolTip("中值检测窗宽（奇数），须明显大于脉冲宽度，默认41≈20ms")
                elif key == "max_len":
                    w.setRange(1, 200)
                    w.setToolTip("连续超过阈值的样本数≤此值才判定为脉冲，更大则视为真实信号")
                else:  # size 中值窗宽
                    w.setRange(3, 201)
                    w.setSingleStep(2)
                    w.setToolTip("窗宽（奇数）")
                w.setValue(int(val))
                w.valueChanged.connect(
                    lambda v, k=key: self._param_changed(i, k, int(v)))
            self.param_form.addRow(f"  {key}:", w)

    def _param_changed(self, idx, key, value):
        if 0 <= idx < len(self.steps):
            self.steps[idx]["params"][key] = value
            self._update_step_item(idx)

    # ---- 处理与显示 ----
    def _apply(self):
        if self.sig is None:
            QtWidgets.QMessageBox.warning(self, "未加载数据", "请先打开 CSV 文件。")
            return
        if not self.steps:
            QtWidgets.QMessageBox.information(self, "提示", "还没有处理步骤，请先添加滤波方式。")
            return
        try:
            self.proc = apply_pipeline(self.sig, self.fs, self.steps)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "处理失败", str(e))
            return
        self._plot()
        infos = [s.get("info") for s in self.steps if s.get("info")]
        msg = f"处理完成：{len(self.sig)} 个样本，{len(self.steps)} 步"
        if infos:
            msg += "　" + "；".join(infos)
        self.statusBar().showMessage(msg)

    def _export(self):
        if self.proc is None:
            QtWidgets.QMessageBox.warning(self, "无处理后信号", "请先点击「应用处理」。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出 WAV", os.path.join(os.getcwd(), "processed.wav"), "WAV 音频 (*.wav)")
        if not path:
            return
        try:
            export_wav(self.proc, self.fs, path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导出失败", str(e))
            return
        self.statusBar().showMessage(f"已导出 {path}，正在用系统播放器打开…")
        try:
            os.startfile(path)   # Windows 默认播放器
        except Exception:
            pass

    def _on_range_toggle(self, *_):
        self.t0_spin.setEnabled(self.time_chk.isChecked())
        self.t1_spin.setEnabled(self.time_chk.isChecked())
        self.ymin_spin.setEnabled(self.y_chk.isChecked())
        self.ymax_spin.setEnabled(self.y_chk.isChecked())

    def _export_source(self):
        """根据「内容」选择返回 (主信号, 原始信号或None, 模式)。模式: proc/orig/both"""
        mode = self.src_combo.currentData()
        if mode == "orig":
            return self.sig, None, "orig"
        if mode == "both":
            return self.proc, self.sig, "both"
        return self.proc, None, "proc"

    def _sync_from_view(self):
        if self.proc is None:
            QtWidgets.QMessageBox.warning(self, "无处理后信号", "请先点击「应用处理」再同步视图。")
            return
        # 导出内容为「原始」时同步上方原始波形，否则同步下方处理后波形
        plot = self.p_orig if self.src_combo.currentData() == "orig" else self.p_proc
        [[x0, x1], [y0, y1]] = plot.viewRange()
        self.time_chk.setChecked(True)
        self.y_chk.setChecked(True)
        self.t0_spin.setValue(round(float(x0), 3))
        self.t1_spin.setValue(round(float(x1), 3))
        self.ymin_spin.setValue(round(float(y0), 1))
        self.ymax_spin.setValue(round(float(y1), 1))
        self.statusBar().showMessage(
            "已同步视图范围：t %.3f~%.3f s，y %.1f~%.1f" % (x0, x1, y0, y1))

    def _export_window(self):
        """返回 (t0, t1, ymin, ymax)；None 表示用全长 / 自动幅值"""
        if self.time_chk.isChecked():
            t0, t1 = self.t0_spin.value(), self.t1_spin.value()
            if t1 <= t0:
                t1 = None          # 只给起点：终点取数据末尾
        else:
            t0 = t1 = None
        if self.y_chk.isChecked():
            ymin, ymax = self.ymin_spin.value(), self.ymax_spin.value()
        else:
            ymin = ymax = None
        return t0, t1, ymin, ymax

    def _export_csv(self):
        if self.proc is None:
            QtWidgets.QMessageBox.warning(self, "无处理后信号", "请先点击「应用处理」。")
            return
        sig2, orig2, mode = self._export_source()
        t0, t1, _, _ = self._export_window()
        default = {"both": "raw_and_processed.csv", "proc": "processed.csv",
                   "orig": "raw.csv"}[mode]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出 CSV", os.path.join(os.getcwd(), default), "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            export_csv(sig2, self.fs, path, t0, t1, original=orig2, source=mode)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导出失败", str(e))
            return
        rng = "全段" if t0 is None else "%.3f~%.3fs" % (t0, t1)
        self.statusBar().showMessage(f"已导出 CSV（{rng}，内容:{self.src_combo.currentText()}）：{path}")

    def _export_png(self):
        if self.proc is None:
            QtWidgets.QMessageBox.warning(self, "无处理后信号", "请先点击「应用处理」。")
            return
        sig2, orig2, mode = self._export_source()
        t0, t1, ymin, ymax = self._export_window()
        title = {"both": "Raw + Processed", "proc": "Processed Signal",
                 "orig": "Raw Signal"}[mode]
        default = {"both": "raw_and_processed_waveform.png", "proc": "processed_waveform.png",
                   "orig": "raw_waveform.png"}[mode]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出波形图", os.path.join(os.getcwd(), default), "PNG 图片 (*.png)")
        if not path:
            return
        try:
            export_waveform_png(sig2, self.fs, path, t0, t1, ymin, ymax,
                                title=title, original=orig2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导出失败", str(e))
            return
        rng = "全段" if t0 is None else "%.3f~%.3fs" % (t0, t1)
        yrng = "自动" if ymin is None else "%.0f~%.0f" % (ymin, ymax)
        self.statusBar().showMessage(
            f"已导出波形图（{rng}，Y:{yrng}，内容:{self.src_combo.currentText()}）：{path}")

    def _plot(self):
        if self.sig is None:
            return
        n = len(self.sig)
        t = np.arange(n) / self.fs
        self.curve_orig.setData(t, self.sig)
        if self.proc is not None:
            self.curve_proc.setData(t, self.proc)
        else:
            self.curve_proc.setData([], [])
        self.p_orig.setXRange(0, n / self.fs, padding=0)


def main():
    ap = argparse.ArgumentParser(description="离线信号处理与回放工具")
    ap.add_argument("file", nargs="?", help="可直接在命令行指定 CSV 文件")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Offline Processor")
    win = MainWindow()
    win.show()
    if args.file:
        win._load_file(args.file)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
