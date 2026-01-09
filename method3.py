# %%
from read_data import tfr_power, freqs, times
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
from read_data import tfr_power, freqs, times, eeg
print(f'{tfr_power.shape=}, {freqs.shape=}, {times.shape=}')

# %%


class GradientParticles:
    def __init__(self, tfr_data, start_time_idx=0, window_size=250):
        """初始化粒子系统

        Args:
            tfr_data: 时频数据，形状为(频率数, 时间点数)
            start_time_idx: 起始时间点索引
            window_size: 时间窗口大小
        """
        # 获取数据窗口
        self.data_window = tfr_data[:,
                                    start_time_idx:start_time_idx+window_size]
        self.window_size = window_size
        self.current_time_idx = start_time_idx
        self.max_time_idx = tfr_data.shape[1] - window_size

        self.max_value = np.max(tfr_data)
        self.min_value = np.min(tfr_data)

        # 目标显示尺寸
        self.target_height = 900
        self.target_width = 1440

        # 插值数据到目标尺寸
        self.interpolated_field = self.interpolate_field()

        # 计算梯度场
        self.compute_gradients()

        # 粒子参数
        self.num_particles = 2000
        self.particles = self.initialize_particles()
        self.particle_trails = []

        # 可视化参数
        self.particle_radius = 2
        self.trail_length = 20
        self.max_speed = 5
        self.friction = 0.95

        # 创建颜色映射
        self.create_colormap()

    def interpolate_field(self):
        """三次插值将数据缩放到目标尺寸"""
        # 原始数据尺寸
        orig_height, orig_width = self.data_window.shape

        # 第一次插值：放大宽度
        interp1 = cv2.resize(self.data_window, (self.target_width, orig_height),
                             interpolation=cv2.INTER_CUBIC)

        # 第二次插值：放大高度
        interp2 = cv2.resize(interp1, (self.target_width, self.target_height),
                             interpolation=cv2.INTER_CUBIC)

        # 第三次插值：轻微平滑（可选）
        interp3 = cv2.GaussianBlur(interp2, (3, 3), 0.5)

        return interp3

    def compute_gradients(self):
        """计算标量场的梯度"""
        # 使用Sobel算子计算梯度
        grad_x = cv2.Sobel(self.interpolated_field, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.interpolated_field, cv2.CV_64F, 0, 1, ksize=3)

        # 归一化梯度
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude[magnitude == 0] = 1  # 避免除零

        self.grad_x = grad_x / magnitude * -1  # 负梯度用于下降
        self.grad_y = grad_y / magnitude * -1

        # 计算能量（用于可视化）
        self.energy = self.normalize_field(self.interpolated_field)

    def normalize_field(self, field):
        """归一化场到[0, 1]范围"""
        field_min = field.min()
        field_max = field.max()
        field_min = self.min_value
        field_max = self.max_value
        if field_max > field_min:
            return (field - field_min) / (field_max - field_min)
        return field

    def initialize_particles(self):
        """随机初始化粒子位置"""
        particles = []
        for _ in range(self.num_particles):
            # 随机位置，避免边界
            x = random.randint(50, self.target_width - 50)
            y = random.randint(50, self.target_height - 50)
            # 随机速度
            vx = random.uniform(-1, 1)
            vy = random.uniform(-1, 1)
            particles.append({
                'x': x, 'y': y,
                'vx': vx, 'vy': vy,
                'color': self.get_random_color(),
                'trail': []
            })
        return particles

    def get_random_color(self):
        """生成随机鲜艳颜色"""
        # 使用HSV颜色空间获取更鲜艳的颜色
        hue = random.randint(0, 179)
        return cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

    def create_colormap(self):
        """创建用于背景可视化的颜色映射"""
        # 使用matplotlib的颜色映射
        self.cmap = plt.get_cmap('viridis')
        self.cmap = plt.get_cmap('RdBu')

    def update_particles(self):
        """更新粒子位置基于梯度下降"""
        for particle in self.particles:
            # 获取粒子当前位置（转换为整数索引）
            x_int = int(particle['x'])
            y_int = int(particle['y'])

            # 确保在边界内
            x_int = max(0, min(self.target_width - 1, x_int))
            y_int = max(0, min(self.target_height - 1, y_int))

            # 获取梯度方向
            gx = self.grad_x[y_int, x_int]
            gy = self.grad_y[y_int, x_int]

            # 更新速度（梯度下降 + 动量）
            particle['vx'] = particle['vx'] * self.friction + gx * 0.5
            particle['vy'] = particle['vy'] * self.friction + gy * 0.5

            # 限制最大速度
            speed = np.sqrt(particle['vx']**2 + particle['vy']**2)
            if speed > self.max_speed:
                particle['vx'] = particle['vx'] / speed * self.max_speed
                particle['vy'] = particle['vy'] / speed * self.max_speed

            # 更新位置
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']

            particle['x'] += np.random.random() * 0.1 * particle['vx']
            particle['y'] += np.random.random() * 0.1 * particle['vy']

            # 边界处理
            if particle['x'] < 0 or particle['x'] >= self.target_width:
                particle['vx'] *= -0.5
                particle['x'] = np.random.random() * self.target_width
                particle['y'] = np.random.random() * self.target_height
                # particle['x'] = max(
                #     0, min(self.target_width - 1, particle['x']))
            if particle['y'] < 0 or particle['y'] >= self.target_height:
                particle['vy'] *= -0.5
                particle['x'] = np.random.random() * self.target_width
                particle['y'] = np.random.random() * self.target_height
                # particle['y'] = max(
                #     0, min(self.target_height - 1, particle['y']))

            # 更新轨迹
            # particle['trail'].append((int(particle['x']), int(particle['y'])))
            # if len(particle['trail']) > self.trail_length:
            #     particle['trail'].pop(0)

    def update_time_window(self, new_time_idx):
        """更新时间窗口数据"""
        if 0 <= new_time_idx <= self.max_time_idx:
            self.current_time_idx = new_time_idx
            # 获取新数据窗口
            self.data_window = tfr_power[:,
                                         new_time_idx:new_time_idx+self.window_size]
            # 重新插值和计算梯度
            self.interpolated_field = self.interpolate_field()
            self.compute_gradients()
            return True
        return False

    def create_visualization(self):
        """创建可视化图像"""
        # 创建背景（使用颜色映射）
        background = (self.cmap(self.energy) * 255 *
                      0.5).astype(np.uint8)[:, :, :3]
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

        # 添加梯度方向的可视化（可选）
        # 每20个像素画一个梯度箭头
        step = 20
        for y in range(0, self.target_height, step):
            for x in range(0, self.target_width, step):
                gx = self.grad_x[y, x]
                gy = self.grad_y[y, x]
                # 只画明显的梯度
                if abs(gx) > 0.1 or abs(gy) > 0.1:
                    length = 10
                    end_x = int(x + gx * length)
                    end_y = int(y + gy * length)
                    cv2.arrowedLine(background, (x, y), (end_x, end_y),
                                    (100, 100, 150), 1, tipLength=0.3)

        # 绘制粒子轨迹
        for particle in self.particles:
            trail = particle['trail']
            color = particle['color'].tolist()

            # 绘制轨迹线（渐变色）
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                trail_color = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(2 * alpha))
                cv2.line(background, trail[i-1],
                         trail[i], trail_color, thickness)

            # 绘制粒子当前位置
            x, y = int(particle['x']), int(particle['y'])
            cv2.circle(background, (x, y), self.particle_radius,
                       color, -1)
            # 添加光晕效果
            cv2.circle(background, (x, y), self.particle_radius + 1,
                       (255, 255, 255), 1)

        # 添加信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(background, f'Time: {self.current_time_idx}',
                    (20, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(background, f'Particles: {self.num_particles}',
                    (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(background, 'Press: Q=Quit, R=Reset, N=Next Window',
                    (20, self.target_height - 20), font, 0.6, (200, 200, 200), 1)

        return background


def main():
    # 检查数据
    print(f"TFR数据形状: {tfr_power.shape}")
    print(f"频率点数: {freqs.shape[0]}")
    print(f"时间点数: {times.shape[0]}")

    # 创建粒子系统
    window_size = 250
    start_idx = 0

    particle_system = GradientParticles(tfr_power, start_idx, window_size)

    # 创建显示窗口
    window_name = "TFR Gradient Descent Particles"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 900)

    frame_count = 0
    time_step = 0
    auto_advance_counter = 0

    print("控制说明:")
    print("  Q: 退出")
    print("  R: 重置粒子")
    print("  N: 下一个时间窗口")
    print("  P: 上一个时间窗口")
    print("  +/-: 增加/减少粒子数量")

    while True:
        # 更新粒子
        particle_system.update_particles()

        # 每隔一定帧数自动推进时间窗口（可选）
        auto_advance_counter += 1
        if auto_advance_counter >= 300:  # 每300帧推进一次
            time_step = (time_step + 50) % particle_system.max_time_idx
            particle_system.update_time_window(time_step)
            auto_advance_counter = 0

        # 创建可视化
        visualization = particle_system.create_visualization()

        # 显示
        cv2.imshow(window_name, visualization)

        # 键盘控制
        key = cv2.waitKey(3) & 0xFF

        time_step = min(particle_system.max_time_idx, time_step + 1)
        particle_system.update_time_window(time_step)

        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重置粒子
            particle_system.particles = particle_system.initialize_particles()
        elif key == ord('n'):
            # 下一个时间窗口
            time_step = min(particle_system.max_time_idx, time_step + 50)
            particle_system.update_time_window(time_step)
            auto_advance_counter = 0
        elif key == ord('p'):
            # 上一个时间窗口
            time_step = max(0, time_step - 50)
            particle_system.update_time_window(time_step)
            auto_advance_counter = 0
        elif key == ord('+') or key == ord('='):
            # 增加粒子
            particle_system.num_particles = min(
                5000, particle_system.num_particles + 100)
            particle_system.particles = particle_system.initialize_particles()
        elif key == ord('-'):
            # 减少粒子
            particle_system.num_particles = max(
                100, particle_system.num_particles - 100)
            particle_system.particles = particle_system.initialize_particles()

        frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
