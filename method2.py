"""
File: method2.py
Author: Chuncheng Zhang
Date: 2026-01-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and display its tfr.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-09 ------------------------
# Requirements and constants
import cv2
from scipy import signal as sp_signal
from util.easy_import import *
from util.signal_processing import bandpass_filter

# %% ---- 2026-01-09 ------------------------
# Function and class


class EEG:
    fs: int = 250
    n_channels: int = 2

    interval = 1 / fs

    def __init__(self, signal: np.ndarray):
        n_times, n_channels = signal.shape
        times = np.arange(0, self.interval * n_times, self.interval)[:n_times]
        self.signal = signal
        self.times = times
        self.n_times = n_times

    def preprocessing(self):
        filtered_signal = bandpass_filter(
            self.signal, lowcut=1.0, highcut=45.0, fs=self.fs)

        self.filtered_signal = filtered_signal

        return self.filtered_signal

    def setup_fetch_data(self, start: int = 0):
        self.fetch_start = 0

    def fetch_data(self, length: int = fs):
        a, b = self.fetch_start, self.fetch_start+length

        fetched = self.filtered_signal[a: b]

        self.fetch_start = 0 if b > self.n_times else b

        return fetched


def read_example_data():
    df = pd.read_csv('./data/data.csv')
    # eeg shape is (n_times, n_channels)
    eeg = df[['eeg_1', 'eeg_2']].to_numpy()

    return eeg


raw_eeg = read_example_data()
eeg = EEG(raw_eeg)
eeg.preprocessing()
eeg.setup_fetch_data()


# %% ---- 2026-01-09 ------------------------
# Play ground

def tfr_morlet(signal, fs, freqs=None, n_cycles=None, use_fft=True, return_complex=False):
    """
    使用Morlet小波进行时频分析

    参数:
    ----------
    signal : ndarray, shape (n_times,)
        输入信号（1D时间序列）
    fs : float
        采样频率（Hz）
    freqs : ndarray or None, shape (n_freqs,)
        要分析的频率数组，默认1~45Hz，对数间隔
    n_cycles : float or ndarray
        每个频率的小波周期数，可以是标量或与freqs长度相同的数组
    use_fft : bool
        是否使用FFT加速计算（推荐True）
    return_complex : bool
        是否返回复数结果（默认返回幅度）

    返回:
    ----------
    tfr_power : ndarray, shape (n_freqs, n_times)
        时频能量矩阵（功率）
    tfr_phase : ndarray, shape (n_freqs, n_times), optional
        时频相位矩阵（仅当return_complex=True时返回）
    freqs : ndarray
        频率数组
    times : ndarray
        时间数组
    """

    # 参数检查和默认值设置
    signal = np.asarray(signal, dtype=np.float64)
    n_times = len(signal)

    # 默认频率范围：1-45Hz，对数间隔（符合感知特性）
    if freqs is None:
        # 对数间隔更适合脑电分析
        freqs = np.logspace(np.log10(1), np.log10(45), 40)
        # 或者线性间隔：freqs = np.linspace(1, 45, 40)

    n_freqs = len(freqs)

    # 默认周期数：低频用更少周期，高频用更多周期
    if n_cycles is None:
        # 典型设置：低频3-4周期，高频7-10周期
        n_cycles = freqs / 3.0  # 频率越高，周期数越多
        n_cycles = np.clip(n_cycles, 3, 10)
    elif np.isscalar(n_cycles):
        n_cycles = np.ones(n_freqs) * n_cycles

    # 时间数组
    times = np.arange(n_times) / fs

    # 预分配结果数组
    if return_complex:
        tfr_result = np.zeros((n_freqs, n_times), dtype=np.complex128)
    else:
        tfr_power = np.zeros((n_freqs, n_times))

    # 预处理信号：去均值
    signal = signal - np.mean(signal)

    if use_fft:
        # ====================
        # 使用FFT加速的版本
        # ====================

        # 计算信号的FFT
        n_fft = next_pow2(n_times * 2)  # 使用2的幂次长度，避免循环卷积
        signal_fft = np.fft.fft(signal, n_fft)
        freqs_fft = np.fft.fftfreq(n_fft, 1/fs)

        # 对每个频率进行卷积
        for i, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
            # 生成Morlet小波（时域）
            wavelet = morlet_wavelet(freq, n_cycle, fs)
            n_wavelet = len(wavelet)

            # 小波的FFT
            wavelet_fft = np.fft.fft(wavelet, n_fft)

            # 频域相乘（卷积定理）
            conv_result = np.fft.ifft(signal_fft * wavelet_fft)

            # 取中间部分（有效卷积结果）
            start = (n_fft - n_times) // 2
            end = start + n_times

            # 处理边界效应：只取中间的有效部分
            if n_wavelet < n_times:
                valid_start = n_wavelet // 2
                valid_end = n_times - n_wavelet // 2

                if return_complex:
                    tfr_result[i, valid_start:valid_end] = conv_result[start +
                                                                       valid_start:start+valid_end]
                    # 边界用最近的有效值填充
                    tfr_result[i, :valid_start] = tfr_result[i, valid_start]
                    tfr_result[i, valid_end:] = tfr_result[i, valid_end-1]
                else:
                    tfr_power[i, valid_start:valid_end] = np.abs(
                        conv_result[start+valid_start:start+valid_end])**2
                    tfr_power[i, :valid_start] = tfr_power[i, valid_start]
                    tfr_power[i, valid_end:] = tfr_power[i, valid_end-1]
            else:
                if return_complex:
                    tfr_result[i] = conv_result[start:end]
                else:
                    tfr_power[i] = np.abs(conv_result[start:end])**2

    else:
        # ====================
        # 时域卷积版本（更慢但直观）
        # ====================

        for i, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
            # 生成Morlet小波
            wavelet = morlet_wavelet(freq, n_cycle, fs)

            # 与小波进行卷积
            conv_result = np.convolve(signal, wavelet, mode='same')

            if return_complex:
                tfr_result[i] = conv_result
            else:
                # 计算功率（幅度的平方）
                tfr_power[i] = np.abs(conv_result)**2

    # 归一化：每个频率的能量除以小波的能量
    if not return_complex:
        for i, n_cycle in enumerate(n_cycles):
            # 小波能量归一化
            wavelet = morlet_wavelet(freqs[i], n_cycle, fs)
            wavelet_energy = np.sum(np.abs(wavelet)**2)
            if wavelet_energy > 0:
                tfr_power[i] /= wavelet_energy

        # 转换为分贝（dB）尺度，更符合感知
        tfr_power = 10 * np.log10(tfr_power + 1e-10)  # 加小量避免log(0)

    if return_complex:
        return tfr_result, freqs, times
    else:
        return tfr_power, freqs, times


def morlet_wavelet(freq, n_cycles, fs):
    """
    生成Morlet小波

    参数:
    ----------
    freq : float
        中心频率（Hz）
    n_cycles : float
        小波周期数
    fs : float
        采样频率（Hz）

    返回:
    ----------
    wavelet : ndarray
        复数Morlet小波
    """
    # 标准差（与周期数相关）
    sigma = n_cycles / (2 * np.pi * freq)

    # 时间范围：±3个标准差
    t_range = 3 * sigma
    t = np.arange(-t_range, t_range, 1/fs)

    # Morlet小波公式
    # 复数小波：exp(i*2πft) * exp(-t²/(2σ²))
    normalization = (sigma * np.sqrt(np.pi)) ** -0.5
    wavelet = normalization * \
        np.exp(1j * 2 * np.pi * freq * t) * np.exp(-t**2 / (2 * sigma**2))

    return wavelet


def next_pow2(n):
    """返回大于等于n的最小的2的幂次"""
    return 1 << (n-1).bit_length()


# %% ---- 2026-01-09 ------------------------
# Pending

# %% ---- 2026-01-09 ------------------------
# Pending


class TFRScroller:
    """时频图长卷式滚动器"""

    def __init__(self, tfr_power, freqs, times, fs=250,
                 window_seconds=2.0, scroll_speed=1.0):
        """
        参数:
        tfr_power: shape (n_freqs, n_times) 时频功率矩阵
        freqs: 频率数组
        times: 时间数组
        fs: 采样率
        window_seconds: 显示窗口长度（秒）
        scroll_speed: 滚动速度倍数（1.0=实时）
        """
        self.tfr_power = tfr_power
        self.freqs = freqs
        self.times = times
        self.fs = fs
        self.window_seconds = window_seconds
        self.scroll_speed = scroll_speed

        # 基础参数
        self.n_freqs, self.n_times = tfr_power.shape
        self.total_seconds = times[-1] - times[0]

        # 计算要显示的时间点数
        self.window_points = int(
            window_seconds * self.n_times / self.total_seconds)

        # 归一化并翻转（低频在下）
        self._preprocess()

        # 显示窗口
        self.display_height = 900
        self.display_width = 1440

        # 当前显示位置
        self.current_idx = 0
        self.frame_count = 0

    def _preprocess(self):
        """预处理：归一化并翻转频率轴"""
        # 归一化到0-255
        tfr_min = np.percentile(self.tfr_power, 5)
        tfr_max = np.percentile(self.tfr_power, 95)
        if tfr_max > tfr_min:
            self.tfr_norm = ((self.tfr_power - tfr_min) /
                             (tfr_max - tfr_min) * 255).astype(np.uint8)
        else:
            self.tfr_norm = np.zeros_like(self.tfr_power, dtype=np.uint8)

        # 翻转频率轴（低频在下）
        self.tfr_norm = np.flipud(self.tfr_norm)

    def get_current_frame(self):
        """获取当前帧"""
        # 清空画布
        frame = np.zeros(
            (self.display_height, self.display_width, 3), dtype=np.uint8)

        # 获取当前要显示的数据段
        start_idx = self.current_idx
        end_idx = min(start_idx + self.window_points, self.n_times)

        if start_idx >= end_idx:
            return frame

        # 提取数据段
        data_segment = self.tfr_norm[:, start_idx:end_idx]

        # 调整大小到显示高度
        if data_segment.shape[0] != self.display_height:
            data_segment = cv2.resize(data_segment.astype(np.float32),
                                      (data_segment.shape[1],
                                       self.display_height),
                                      interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        # 调整宽度到显示宽度
        segment_width = data_segment.shape[1]
        if segment_width > 0:
            # 应用颜色映射
            # colored_segment = cv2.applyColorMap(data_segment, cv2.COLORMAP_JET)
            colored_segment = cv2.applyColorMap(
                data_segment, cv2.COLORMAP_CIVIDIS)

            # 计算显示位置（从右侧开始）
            display_width = min(self.display_width, segment_width)
            if segment_width < self.display_width:
                # 数据不够宽，从右侧开始显示
                x_start = self.display_width - segment_width
                frame[:, x_start:x_start+segment_width] = colored_segment
            else:
                # 数据比显示区域宽，取最右侧部分
                x_start = 0
                frame[:, :] = colored_segment[:, -self.display_width:]

        # 添加时间标签
        current_time = self.times[start_idx]
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(frame, time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 更新位置（每帧移动点数）
        points_per_frame = int(
            self.scroll_speed * self.n_times / self.total_seconds / 30)  # 假设30fps
        self.current_idx += max(1, points_per_frame)

        # 循环播放
        if self.current_idx >= self.n_times:
            self.current_idx = 0

        self.frame_count += 1
        return frame


def start_tfr_scrolling(tfr_power, freqs, times, fs):
    """
    启动时频图滚动显示

    参数:
    tfr_power: 时频功率矩阵 (n_freqs, n_times)
    freqs: 频率数组
    times: 时间数组
    fs: 采样率
    """
    scroller = TFRScroller(
        tfr_power=tfr_power,
        freqs=freqs,
        times=times,
        fs=fs,
        window_seconds=4.0,   # 2秒窗口
        scroll_speed=1.0      # 实时速度
    )

    cv2.namedWindow('EEG TFR Scrolling', cv2.WINDOW_NORMAL)
    cv2.namedWindow('EEG TFR Scrolling', cv2.WINDOW_AUTOSIZE)

    while True:
        frame = scroller.get_current_frame()
        cv2.imshow('EEG TFR Scrolling', frame)

        # 按键控制
        key = cv2.waitKey(33) & 0xFF

        if key == 27:  # ESC退出
            break
        elif key == ord('+'):  # 加速
            scroller.scroll_speed *= 1.2
            print(f"速度: {scroller.scroll_speed:.1f}x")
        elif key == ord('-'):  # 减速
            scroller.scroll_speed *= 0.8
            print(f"速度: {scroller.scroll_speed:.1f}x")
        elif key == ord('r'):  # 重置位置
            scroller.current_idx = 0
            print("位置重置")

    cv2.destroyAllWindows()


def create_scrolling_video(tfr_power, freqs, times, fs, output_file='tfr_scroll.mp4'):
    """
    生成滚动视频文件
    """
    scroller = TFRScroller(
        tfr_power=tfr_power,
        freqs=freqs,
        times=times,
        fs=fs,
        window_seconds=2.0,
        scroll_speed=1.5
    )

    # 设置视频参数
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps,
                          (scroller.display_width, scroller.display_height))

    # 计算总帧数（播放完整数据）
    total_seconds = times[-1] - times[0]
    total_frames = int(fps * total_seconds / scroller.scroll_speed)

    print(f"生成视频: {output_file}")
    print(f"总帧数: {total_frames}")

    for i in range(total_frames):
        frame = scroller.get_current_frame()
        out.write(frame)

        # 显示进度
        if i % 100 == 0:
            progress = (i + 1) / total_frames * 100
            print(f"进度: {progress:.1f}%")

    out.release()
    print("视频生成完成")

# ============================================================================
# 主程序
# ============================================================================


# %%
# tfr_power shape is (n_freqs, n_times)
tfr_power, freqs, times = tfr_morlet(
    eeg.filtered_signal[:, 0], eeg.fs)

print(f'{tfr_power.shape}')

if __name__ == "__main__":
    print("时频图长卷式滚动显示")
    print("=" * 40)

    # 直接使用示例
    start_tfr_scrolling(tfr_power, freqs, times, eeg.fs)

# %%
