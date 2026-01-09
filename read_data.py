# %%
from util.easy_import import *
from util.signal_processing import bandpass_filter


# %%
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


# %%
raw_eeg = read_example_data()
eeg = EEG(raw_eeg)
eeg.preprocessing()
eeg.setup_fetch_data()

tfr_power, freqs, times = tfr_morlet(
    eeg.filtered_signal[:, 0], eeg.fs)
