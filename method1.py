"""
File: method1.py
Author: Chuncheng Zhang
Date: 2026-01-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and display? it.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-09 ------------------------
# Requirements and constants
from collections import deque
from scipy.stats import entropy
from scipy.signal import welch, butter, filtfilt
import cv2

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


class EEGVisualizer:
    def __init__(self, width=1200, height=800):
        """
        初始化OpenCV脑电可视化器
        """
        self.width = width
        self.height = height

        # 创建画布
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # 粒子系统
        self.particles = []
        self.max_particles = 500

        # 历史数据存储（用于趋势显示）
        self.history_length = 100
        self.valence_history = deque(maxlen=self.history_length)
        self.arousal_history = deque(maxlen=self.history_length)
        self.alpha_history = deque(maxlen=self.history_length)

        # 颜色映射
        self.colormap = self.create_colormap()

        # 状态变量
        self.time_counter = 0

    def create_colormap(self):
        """创建情绪颜色映射：从蓝色（负）到红色（正）"""
        colormap = []
        for i in range(256):
            # 负情绪：蓝色系
            if i < 128:
                b = 255
                g = int(i * 2)
                r = 0
            # 正情绪：红色系
            else:
                b = 0
                g = int(255 - (i-128)*2)
                r = 255
            colormap.append((b, g, r))
        return colormap

    def extract_features(self, left_signal, right_signal, fs=250):
        """
        从2导联额叶EEG提取特征
        """
        features = {}

        # 1. 预处理：带通滤波 1-45 Hz
        b, a = butter(4, [1/(fs/2), 45/(fs/2)], btype='band')
        left_filtered = filtfilt(b, a, left_signal)
        right_filtered = filtfilt(b, a, right_signal)

        # 2. 计算功率谱
        freqs, Pxx_left = welch(left_filtered, fs, nperseg=fs)
        freqs, Pxx_right = welch(right_filtered, fs, nperseg=fs)

        # 3. 频带定义
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # 4. 计算各频带功率
        for band_name, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            features[f'left_{band_name}'] = np.sum(Pxx_left[idx])
            features[f'right_{band_name}'] = np.sum(Pxx_right[idx])

        # 5. 关键特征计算
        # FAA: 情绪效价
        features['FAA'] = np.log(
            features['right_alpha'] + 1e-10) - np.log(features['left_alpha'] + 1e-10)

        # 唤醒度: Beta总功率
        features['arousal'] = (features['left_beta'] +
                               features['right_beta']) / 2

        # 平静度: Alpha总功率
        features['calm'] = (features['left_alpha'] +
                            features['right_alpha']) / 2

        # 想象水平: Theta功率
        features['imagination'] = (
            features['left_theta'] + features['right_theta']) / 2

        # 左右平衡
        total_left = sum([features[f'left_{b}'] for b in bands.keys()])
        total_right = sum([features[f'right_{b}'] for b in bands.keys()])
        features['balance'] = (total_right - total_left) / \
            (total_right + total_left + 1e-10)

        # 复杂度: 样本熵（简化版）
        features['complexity'] = np.std(left_filtered) * np.std(right_filtered)

        # 归一化到[0, 1]
        for key in ['arousal', 'calm', 'imagination', 'complexity']:
            features[key] = np.tanh(features[key] * 0.1)  # 压缩函数

        # FAA归一化到[-1, 1]
        features['FAA'] = np.clip(features['FAA'] * 2, -1, 1)

        return features

    def create_particle(self, features):
        """根据特征创建新粒子"""
        # 情绪决定颜色
        valence_norm = int((features['FAA'] + 1) * 127.5)  # [-1,1] -> [0,255]
        color = self.colormap[valence_norm]

        # 唤醒度决定初始速度
        speed = 2 + features['arousal'] * 8

        # 想象水平决定粒子类型
        if features['imagination'] > 0.7:
            # 灵感粒子：特殊轨迹
            particle_type = 'inspiration'
            size = 3
            life = 150
        else:
            # 普通粒子
            particle_type = 'normal'
            size = 2
            life = 100

        # 左右平衡决定水平位置
        x = int(self.width * (0.5 + features['balance'] * 0.4))

        particle = {
            'x': x,
            'y': self.height - 50,  # 从底部产生
            'vx': (np.random.random() - 0.5) * speed,
            'vy': -speed * (0.8 + np.random.random() * 0.4),
            'color': color,
            'size': size,
            'life': life,
            'max_life': life,
            'type': particle_type,
            'trail': []  # 轨迹点存储
        }
        return particle

    def update_particles(self, features):
        """更新所有粒子状态"""
        # 根据特征创建新粒子
        particles_to_create = int(
            features['arousal'] * 10 + features['imagination'] * 5)

        for _ in range(min(particles_to_create, 5)):  # 限制每帧最大生成数
            if len(self.particles) < self.max_particles:
                self.particles.append(self.create_particle(features))

        # 更新现有粒子
        new_particles = []
        for p in self.particles:
            # 生命值减少
            p['life'] -= 1

            # 不同类型粒子的不同行为
            if p['type'] == 'inspiration':
                # 灵感粒子：螺旋轨迹
                angle = self.time_counter * 0.1 + p['life'] * 0.05
                p['vx'] += np.sin(angle) * 0.5
                p['vy'] += np.cos(angle) * 0.3
            else:
                # 普通粒子：受脑电特征影响的力场
                p['vx'] += (features['balance'] * 0.2 +
                            (np.random.random() - 0.5) * 0.1)
                p['vy'] += (features['calm'] * -0.1 -  # 平静度产生向上力
                            features['arousal'] * 0.05)  # 唤醒度产生向下力

            # 速度衰减
            p['vx'] *= 0.99
            p['vy'] *= 0.99

            # 更新位置
            p['x'] += p['vx']
            p['y'] += p['vy']

            # 存储轨迹点（用于绘制尾迹）
            p['trail'].append((int(p['x']), int(p['y'])))
            if len(p['trail']) > 10:  # 只保留最近10个点
                p['trail'].pop(0)

            # 边界处理
            if p['x'] < 0:
                p['x'] = 0
                p['vx'] = abs(p['vx']) * 0.5
            if p['x'] >= self.width:
                p['x'] = self.width-1
                p['vx'] = -abs(p['vx']) * 0.5
            if p['y'] < 0:
                p['y'] = 0
                p['vy'] = abs(p['vy']) * 0.5
            if p['y'] >= self.height:
                p['y'] = self.height-1
                p['vy'] = -abs(p['vy']) * 0.5

            # 保留还有生命的粒子
            if p['life'] > 0:
                new_particles.append(p)

        self.particles = new_particles

    def draw_frequency_bars(self, features):
        """绘制频带功率条形图"""
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_colors = [(100, 100, 200), (100, 200, 100), (200, 100, 100),
                       (200, 200, 100), (200, 100, 200)]

        bar_width = 40
        start_x = 50

        for i, band in enumerate(bands):
            # 左右脑功率
            left_power = features[f'left_{band}']
            right_power = features[f'right_{band}']

            # 归一化（相对显示）
            total_power = sum(
                [features[f'left_{b}'] + features[f'right_{b}'] for b in bands])
            left_height = int((left_power / total_power) * 200)
            right_height = int((right_power / total_power) * 200)

            # 绘制左脑条形
            x = start_x + i * (bar_width * 2 + 10)
            cv2.rectangle(self.canvas,
                          (x, self.height - 50),
                          (x + bar_width, self.height - 50 - left_height),
                          band_colors[i], -1)

            # 绘制右脑条形
            cv2.rectangle(self.canvas,
                          (x + bar_width + 5, self.height - 50),
                          (x + bar_width * 2 + 5, self.height - 50 - right_height),
                          band_colors[i], -1)

            # 频带标签
            cv2.putText(self.canvas, band.upper(),
                        (x, self.height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_center_aura(self, features):
        """绘制中心光晕（平静度）"""
        center_x, center_y = self.width // 2, self.height // 2
        aura_radius = int(50 + features['calm'] * 100)

        # 多层光晕效果
        for i in range(3):
            radius = aura_radius - i * 10
            alpha = 0.3 - i * 0.1
            color = (0, int(200 * features['calm']), 255)

            # 创建透明层
            overlay = self.canvas.copy()
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.addWeighted(overlay, alpha, self.canvas,
                            1 - alpha, 0, self.canvas)

    def draw_particles(self):
        """绘制所有粒子"""
        for p in self.particles:
            # 绘制尾迹
            if len(p['trail']) > 1:
                for i in range(1, len(p['trail'])):
                    alpha = i / len(p['trail'])  # 尾迹透明度衰减
                    color = [int(c * alpha) for c in p['color']]
                    thickness = max(1, int(p['size'] * alpha))

                    cv2.line(self.canvas,
                             p['trail'][i-1], p['trail'][i],
                             color, thickness)

            # 绘制粒子本体
            alpha = p['life'] / p['max_life']  # 生命值透明度
            color = [int(c * alpha) for c in p['color']]

            cv2.circle(self.canvas,
                       (int(p['x']), int(p['y'])),
                       p['size'], color, -1)

            # 灵感粒子特殊标记
            if p['type'] == 'inspiration':
                cv2.circle(self.canvas,
                           (int(p['x']), int(p['y'])),
                           p['size'] + 2, (255, 255, 255), 1)

    def draw_history_graphs(self):
        """绘制特征历史曲线"""
        if len(self.valence_history) < 2:
            return

        # 情绪效价历史（顶部）
        graph_height = 80
        graph_width = 400

        # 1. 情绪曲线
        cv2.rectangle(self.canvas,
                      (self.width - graph_width - 20, 20),
                      (self.width - 20, 20 + graph_height),
                      (40, 40, 40), -1)

        points = []
        for i, val in enumerate(self.valence_history):
            x = self.width - 20 - graph_width + \
                int(i * graph_width / self.history_length)
            y = 20 + graph_height // 2 - int(val * graph_height // 2)
            points.append((x, y))

        if len(points) > 1:
            cv2.polylines(self.canvas, [np.array(
                points)], False, (255, 100, 100), 2)

        cv2.putText(self.canvas, "Valence",
                    (self.width - graph_width - 15, 15 + graph_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2. 唤醒度曲线
        cv2.rectangle(self.canvas,
                      (self.width - graph_width - 20, 20 + graph_height + 10),
                      (self.width - 20, 20 + graph_height * 2 + 10),
                      (40, 40, 40), -1)

        points = []
        for i, val in enumerate(self.arousal_history):
            x = self.width - 20 - graph_width + \
                int(i * graph_width / self.history_length)
            y = 20 + graph_height + 10 + graph_height - int(val * graph_height)
            points.append((x, y))

        if len(points) > 1:
            cv2.polylines(self.canvas, [np.array(
                points)], False, (100, 255, 100), 2)

        cv2.putText(self.canvas, "Arousal",
                    (self.width - graph_width - 15, 15 + graph_height * 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_info_text(self, features):
        """绘制特征信息文本"""
        y_offset = 30
        line_height = 25

        info = [
            f"Valence (FAA): {features['FAA']:.3f}",
            f"Arousal (Beta): {features['arousal']:.3f}",
            f"Calm (Alpha): {features['calm']:.3f}",
            f"Imagination (Theta): {features['imagination']:.3f}",
            f"Balance: {features['balance']:.3f}",
            f"Particles: {len(self.particles)}/{self.max_particles}",
            f"Time: {self.time_counter}"
        ]

        for i, text in enumerate(info):
            cv2.putText(self.canvas, text,
                        (20, y_offset + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def render(self, features):
        """主渲染函数"""
        # 清空画布（带渐变效果）
        self.canvas = cv2.GaussianBlur(self.canvas, (5, 5), 0)
        self.canvas = cv2.addWeighted(self.canvas, 0.95,
                                      np.zeros(
                                          (self.height, self.width, 3), dtype=np.uint8),
                                      0.05, 0)

        # 更新粒子系统
        self.update_particles(features)

        # 绘制各个组件
        self.draw_center_aura(features)
        self.draw_particles()
        self.draw_frequency_bars(features)
        self.draw_history_graphs()
        self.draw_info_text(features)

        # 更新时间计数器
        self.time_counter += 1

        return self.canvas

# ============================================
# 使用示例：模拟实时EEG数据可视化
# ============================================


def simulate_eeg_data(fs=250, duration=2):
    """生成模拟的2导联EEG数据"""
    t = np.arange(0, duration, 1/fs)

    # 基础脑电成分
    alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
    beta_wave = 0.3 * np.sin(2 * np.pi * 20 * t)   # 20Hz beta
    theta_wave = 0.4 * np.sin(2 * np.pi * 6 * t)   # 6Hz theta

    # 添加情绪变化：FAA随时间变化
    emotion_modulation = np.sin(t * 0.5)  # 缓慢的情绪波动

    # 左脑信号（模拟FAA变化）
    left_signal = (alpha_wave * (1 - emotion_modulation) +
                   beta_wave * (0.5 + 0.3 * np.sin(t * 2)) +
                   theta_wave * (0.3 + 0.2 * np.cos(t * 1.5)) +
                   0.1 * np.random.randn(len(t)))

    # 右脑信号
    right_signal = (alpha_wave * (1 + emotion_modulation) +
                    beta_wave * (0.5 + 0.3 * np.cos(t * 2)) +
                    theta_wave * (0.3 + 0.2 * np.sin(t * 1.5)) +
                    0.1 * np.random.randn(len(t)))

    return left_signal, right_signal


def main():
    """主函数：实时可视化演示"""
    # 初始化可视化器
    visualizer = EEGVisualizer(width=1200, height=800)

    # 创建OpenCV窗口
    cv2.namedWindow('EEG Brain Art', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EEG Brain Art', 1200, 800)

    print("开始脑电艺术生成...")
    print("按ESC键退出")
    print("情绪：红色(积极) - 蓝色(消极)")
    print("唤醒度：粒子速度和数量")
    print("平静度：中心蓝色光晕大小")

    # Read eeg data
    raw_eeg = read_example_data()
    eeg = EEG(raw_eeg)
    eeg.preprocessing()
    eeg.setup_fetch_data()

    # 模拟实时数据流
    fs = eeg.fs
    window_size = 2 * fs  # 2秒窗口

    frame_count = 0
    while True:
        # 模拟获取新的EEG数据（每帧0.1秒，25个采样点）
        # chunk_size = 25
        # left_chunk, right_chunk = simulate_eeg_data(fs, chunk_size/fs)

        # 在实际应用中，这里应该是累积缓冲区
        # 为简化，我们每次重新生成2秒数据并加入随机变化
        # left_signal, right_signal = simulate_eeg_data(fs, window_size/fs)
        signals = eeg.fetch_data(window_size)
        left_signal = signals[:, 0]
        right_signal = signals[:, 1]

        # 加入时间变化
        phase_shift = frame_count * 0.1
        left_signal *= (1 + 0.2 * np.sin(phase_shift))
        right_signal *= (1 + 0.2 * np.cos(phase_shift))

        # 提取特征
        features = visualizer.extract_features(left_signal, right_signal, fs)

        # 更新历史数据
        visualizer.valence_history.append(features['FAA'])
        visualizer.arousal_history.append(features['arousal'])
        visualizer.alpha_history.append(features['calm'])

        # 渲染
        canvas = visualizer.render(features)

        # 显示
        cv2.imshow('EEG Brain Art', canvas)

        frame_count += 1

        # 检查按键
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC键
            break

    cv2.destroyAllWindows()
    print("程序结束")


if __name__ == "__main__":
    main()

# %% ---- 2026-01-09 ------------------------
# Play ground
# raw_eeg = read_example_data()
# eeg = EEG(raw_eeg)
# eeg.preprocessing()

# plt.plot(eeg.filtered_signal[:, 0])
# plt.plot(eeg.filtered_signal[:, 1])
# plt.show()


# %% ---- 2026-01-09 ------------------------
# Pending


# %% ---- 2026-01-09 ------------------------
# Pending
