# -*- coding: utf-8 -*-
"""
V1.0.0稳定版本
针对特定元件实现准确识别与分析
"""

import cv2  # 导入 OpenCV 库，用于视频处理
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
import plotly.graph_objects as go  # 导入 Plotly 库，用于可视化
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
from scipy.ndimage import gaussian_filter1d  # 导入高斯滤波函数
import os  # 导入 os 库，用于文件操作
from concurrent.futures import ThreadPoolExecutor  # 导入多线程执行器

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, roi_x: int, roi_y: int, roi_width: int, roi_height: int):
        '''初始化视频处理器，设置视频路径、输出目录和感兴趣区域（ROI）的坐标和大小'''
        self.video_path = video_path  # 视频文件路径
        self.output_dir = output_dir  # 输出目录
        self.roi_x = roi_x  # ROI 左上角 x 坐标
        self.roi_y = roi_y  # ROI 左上角 y 坐标
        self.roi_width = roi_width  # ROI 宽度
        self.roi_height = roi_height  # ROI 高度
        self.displacements = []  # 存储位移数据的列表
        self.cap = cv2.VideoCapture(self.video_path)  # 打开视频文件

        # 检查视频文件是否成功打开
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频的帧率和总帧数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 设置输出视频文件路径
        self.output_video_path = os.path.join(self.output_dir, "processed_output.mp4")
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 确认 mp4v 编码器
        # 创建视频写入对象
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        '''处理单帧图像，返回处理后的图像'''
        # 将图像转换为灰度
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊以减少噪声
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # 创建二进制图像以识别黑色链条
        _, binary_frame = cv2.threshold(blurred_frame, 50, 255, cv2.THRESH_BINARY_INV)

        # 进行形态学操作来去除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

        # 只提取ROI区域
        roi = morphed_frame[self.roi_y:self.roi_y + self.roi_height, self.roi_x:self.roi_x + self.roi_width]

        # 查找轮廓
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓
        max_contour = self.find_largest_contour(contours)

        # 如果找到了最大的轮廓，跟踪位移
        if max_contour is not None:
            self.track_displacement(frame, max_contour)

        return frame  # 返回处理后的帧

    def find_largest_contour(self, contours):
        '''寻找最大的轮廓'''
        max_contour = None  # 存储最大轮廓
        max_area = 0  # 存储最大面积

        # 遍历所有轮廓，找出最大的
        for contour in contours:
            area = cv2.contourArea(contour)  # 计算当前轮廓的面积
            if area > max_area:  # 如果当前面积大于最大面积
                max_area = area  # 更新最大面积
                max_contour = contour  # 更新最大轮廓
        return max_contour  # 返回最大轮廓

    def track_displacement(self, frame, max_contour):
        '''计算轮廓的重心并跟踪位移'''
        M = cv2.moments(max_contour)  # 计算轮廓的矩
        if M["m00"] != 0:  # 检查轮廓面积是否为零
            # 计算重心坐标
            cx = int(M["m10"] / M["m00"]) + self.roi_x
            cy = int(M["m01"] / M["m00"]) + self.roi_y

            # 确保只记录右边弧形区域的变化
            contour_points = max_contour[:, 0, :] + np.array([self.roi_x, self.roi_y])  # 将ROI坐标加回
            right_side_points = contour_points[contour_points[:, 0] > cx]  # 提取x坐标大于中心的点（右侧）

            if len(right_side_points) > 0:  # 如果右侧有点
                right_displacement = np.mean(right_side_points[:, 0])  # 计算右边点的平均x位置
                displacement = right_displacement - (self.roi_x + self.roi_width // 2)  # 计算位移
                self.displacements.append(displacement)  # 记录位移
                cv2.drawContours(frame, [max_contour + (self.roi_x, self.roi_y)], -1, (0, 255, 0), 2)  # 绘制轮廓
            else:
                self.displacements.append(0)  # 如果没有右侧点，记录位移为0
        else:
            self.displacements.append(0)  # 如果面积为零，记录位移为0

    def process_video(self):
        '''处理视频中的每一帧'''
        with ThreadPoolExecutor() as executor:  # 创建线程池执行器
            for i in tqdm(range(self.total_frames), desc="Processing video frames"):
                ret, frame = self.cap.read()  # 读取一帧
                if not ret:  # 如果没有读取到帧，退出
                    break
                processed_frame = executor.submit(self.process_frame, frame, i).result()  # 提交帧处理任务并获取结果
                self.video_writer.write(processed_frame)  # 将处理后的帧写入输出视频

        self.cap.release()  # 释放视频文件
        self.video_writer.release()  # 释放视频写入对象

    def analyze_waveform(self):
        '''分析位移波形，计算频率和振幅'''
        smoothed_displacements = gaussian_filter1d(self.displacements, sigma=5)  # 对位移数据进行平滑处理
        peaks = []  # 存储峰值的索引
        amplitudes = []  # 存储振幅
        frequencies = []  # 存储频率

        # 寻找波峰
        for i in range(1, len(smoothed_displacements) - 1):
            if smoothed_displacements[i] > smoothed_displacements[i - 1] and smoothed_displacements[i] > smoothed_displacements[i + 1]:
                peaks.append(i)  # 记录波峰索引
                if len(peaks) > 1:  # 如果至少有两个峰
                    freq = self.fps / (peaks[-1] - peaks[-2])  # 计算频率
                    amplitude = smoothed_displacements[peaks[-1]] - min(smoothed_displacements[peaks[-2]:peaks[-1]])  # 计算振幅
                    frequencies.append(freq)  # 记录频率
                    amplitudes.append(amplitude)  # 记录振幅
        return frequencies, amplitudes, peaks  # 返回频率、振幅和峰值索引

    def detect_frequency_changes(self, frequencies, window_size=5, threshold=0.5):
        '''检测频率变化'''
        avg_frequencies = np.convolve(frequencies, np.ones(window_size) / window_size, mode='valid')  # 计算移动平均
        changes = []  # 存储变化点的索引
        for i in range(1, len(avg_frequencies)):
            if abs(avg_frequencies[i] - avg_frequencies[i - 1]) > threshold:  # 判断是否存在显著变化
                changes.append(i + window_size - 1)  # 标记突变点
        return changes, avg_frequencies  # 返回变化点索引和平均频率

    def calculate_average_before_change(self, frequencies, changes, window_size=20):
        '''计算变化点前的平均频率'''
        averages = []  # 存储均值
        for change in changes:
            if change >= window_size:  # 确保变化点之前有足够的数据
                avg = np.mean(frequencies[change - window_size:change])  # 计算前20个点的均值
                averages.append(avg)  # 记录均值
            else:
                averages.append(None)  # 如果没有足够的数据，记录为 None
        return averages  # 返回均值列表

    def save_results(self, frequencies, amplitudes, peaks, changes, avg_frequencies, averages_before_change):
        '''将位移数据和频率数据保存到 CSV 文件'''
        csv_output_path = os.path.join(self.output_dir, "displacement_data.csv")  # CSV 文件路径
        df = pd.DataFrame({
            'Frame': range(len(self.displacements)),  # 帧索引
            'Displacement': self.displacements,  # 位移数据
            'Time (s)': [i / self.fps for i in range(len(self.displacements))],  # 时间数据
            'Frequency (Hz)': np.pad(frequencies, (len(self.displacements) - len(frequencies), 0), 'constant')  # 填充频率数据
        })
        df.to_csv(csv_output_path, index=False)  # 保存为 CSV 文件

        # 使用 plotly 创建交互式图表
        fig = go.Figure()  # 创建图表对象

        smoothed_displacements = gaussian_filter1d(self.displacements, sigma=5)  # 平滑位移数据
        fig.add_trace(go.Scatter(
            x=[i / self.fps for i in range(len(smoothed_displacements))],  # 时间轴
            y=smoothed_displacements,  # 位移数据
            mode='lines',
            name='Smoothed Displacement',
            line=dict(color='blue', width=1)  # 线条颜色和宽度
        ))

        # 添加变化点的标记
        for i, change in enumerate(changes):
            change_time = peaks[change] / self.fps  # 计算变化点时间
            max_y = max(smoothed_displacements)  # 获取最大位移

            # 添加变化点的垂直线
            fig.add_trace(go.Scatter(
                x=[change_time, change_time],
                y=[0, max_y],
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=True,
                name='Change Point Line'  # 变化点线的名称
            ))

            # 添加变化点前的平均频率标记
            if averages_before_change[i] is not None:
                avg_y = averages_before_change[i]
                fig.add_trace(go.Scatter(
                    x=[change_time],
                    y=[max_y + 0.1],  # 在最大位移上方绘制文本
                    mode='text',
                    text=[f'Avg_Max: {avg_y:.2f} Hz'],  # 显示平均频率
                    textposition="bottom center",
                    textfont=dict(size=20),  # 字体大小
                    showlegend=False
                ))

        # 添加频率的散点图
        fig.add_trace(go.Scatter(
            x=[peaks[i] / self.fps for i in range(len(peaks))],  # 时间轴
            y=frequencies,  # 频率数据
            mode='markers',
            name='Frequency',
            marker=dict(color='green', size=4, symbol='cross')  # 标记样式
        ))

        # 更新图表布局
        fig.update_layout(
            title='Displacement and Frequency Analysis',  # 图表标题
            xaxis_title='Time (s)',  # x 轴标题
            yaxis_title='Frequency (Hz) / Displacement (px)',  # y 轴标题
            showlegend=True,  # 显示图例
            template='plotly_white',  # 使用白色模板
            font=dict(size=24)  # 设置字体大小
        )

        # 保存图表为 HTML 文件
        html_output_path = os.path.join(self.output_dir, "displacement_frequency_analysis.html")  # HTML 文件路径
        fig.write_html(html_output_path)  # 写入 HTML 文件
        print("Processing complete. Output saved.")  # 输出处理完成的消息


if __name__ == "__main__":
    # 主程序入口
    video_path = r'C:\Users\kamiy\Documents\day8 stimulation\X1.25_#14E_5s_background+stimulation_2573.avi'  # 视频文件路径
    output_dir = r"C:\Users\kamiy\Documents\output"  # 输出目录
    roi_x, roi_y, roi_width, roi_height = 31, 214, 100, 110  # ROI 的坐标和大小

    # 创建视频处理器实例并开始处理视频
    processor = VideoProcessor(video_path, output_dir, roi_x, roi_y, roi_width, roi_height)
    processor.process_video()  # 处理视频

    # 分析波形并检测频率变化
    frequencies, amplitudes, peaks = processor.analyze_waveform()  # 分析位移波形
    changes, avg_frequencies = processor.detect_frequency_changes(frequencies)  # 检测频率变化
    averages_before_change = processor.calculate_average_before_change(frequencies, changes)  # 计算变化前的均值

    # 保存结果
    processor.save_results(frequencies, amplitudes, peaks, changes, avg_frequencies, averages_before_change)  # 保存结果