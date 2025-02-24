# -*- coding: utf-8 -*-
"""
面向对象版本，保留所有功能并修正位移计算、振幅分析，以及检测频率方差的突然增大。
"""

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import os
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, roi_x: int, roi_y: int, roi_width: int, roi_height: int):
        self.video_path = video_path
        self.output_dir = output_dir
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.displacements = []
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not os.path.exists(self.output_dir):  # 确保输出目录存在
            os.makedirs(self.output_dir)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_video_path = os.path.join(self.output_dir, "processed_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 确认 mp4v 编码器
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
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
    
        max_contour = self.find_largest_contour(contours)
    
        # 在输出帧上绘制ROI区域
        cv2.rectangle(frame, (self.roi_x, self.roi_y), 
                      (self.roi_x + self.roi_width, self.roi_y + self.roi_height), 
                      (255, 0, 0), 2)  # 绘制ROI区域为蓝色矩形
    
        if max_contour is not None:
            self.track_displacement(frame, max_contour)
            # 在输出帧上绘制最大轮廓
            cv2.drawContours(frame, [max_contour + (self.roi_x, self.roi_y)], -1, (0, 255, 0), 2)  # 绘制轮廓为绿色
    
        return frame

    def find_largest_contour(self, contours):
        max_contour = None
        max_area = 0

        # 寻找最大的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        return max_contour

    def track_displacement(self, frame, max_contour):
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            # 使用浮点数计算质心的x和y坐标
            cx = M["m10"] / M["m00"] + self.roi_x
            cy = M["m01"] / M["m00"] + self.roi_y
    
            # 提取轮廓的点并加上roi的偏移量
            contour_points = max_contour[:, 0, :] + np.array([self.roi_x, self.roi_y])

            # 计算x和y方向的位移，不再取整
            displacement_x = cx - (self.roi_x + self.roi_width / 2)
            displacement_y = cy - (self.roi_y + self.roi_height / 2)

            # 记录位移向量（浮点数位移）
            self.displacements.append((displacement_x, displacement_y))
                
            # 在帧上绘制质心
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        else:
            self.displacements.append((0, 0))

    def process_video(self):
        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(self.total_frames), desc="Processing video frames"):
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed_frame = executor.submit(self.process_frame, frame, i).result()
                self.video_writer.write(processed_frame)

        self.cap.release()
        self.video_writer.release()

    def analyze_waveform(self):
        # 分别平滑x和y方向的位移
        displacements_x = [d[0] for d in self.displacements]
        displacements_y = [d[1] for d in self.displacements]
        
        smoothed_displacements_x = gaussian_filter1d(displacements_x, sigma=5)
        smoothed_displacements_y = gaussian_filter1d(displacements_y, sigma=5)
        
        peaks = []
        frequencies = []
        amplitudes = []
        
        # 分析总位移的幅度
        total_displacements = np.sqrt(np.array(smoothed_displacements_x) ** 2 + np.array(smoothed_displacements_y) ** 2)

        for i in range(1, len(total_displacements) - 1):
            if total_displacements[i] > total_displacements[i - 1] and total_displacements[i] > total_displacements[i + 1]:
                peaks.append(i)
                if len(peaks) > 1:
                    freq = self.fps / (peaks[-1] - peaks[-2])  # 计算频率
                    amplitude = total_displacements[peaks[-1]] - min(total_displacements[peaks[-2]:peaks[-1]])  # 计算振幅
                    frequencies.append(freq)
                    amplitudes.append(amplitude)
        return frequencies, amplitudes, peaks

    def detect_frequency_changes(self, frequencies, window_size=5, threshold=0.4):
        avg_frequencies = np.convolve(frequencies, np.ones(window_size) / window_size, mode='valid')
        changes = []
        for i in range(1, len(avg_frequencies)):
            if abs(avg_frequencies[i] - avg_frequencies[i - 1]) > threshold:
                changes.append(i + window_size - 1)  # 标记突变点
        return changes, avg_frequencies

    def calculate_average_before_change(self, frequencies, changes, window_size=20):
        averages = []
        for change in changes:
            if change >= window_size:
                avg = np.mean(frequencies[change - window_size:change])  # 计算前20个点的均值
                averages.append(avg)
            else:
                averages.append(None)
        return averages

    def detect_variance_spike(self, frequencies, window_size=5, variance_threshold=1.5):
        """检测总体方差的突然增加，并标记时刻"""
        # 计算整个频率序列的全局方差
        global_variance = np.var(frequencies)
    
        # 计算滑动窗口内的局部方差
        local_variances = [np.var(frequencies[max(0, i - window_size):i]) for i in range(window_size, len(frequencies))]
    
        # 检测局部方差与全局方差的比率，如果比率超过阈值则认为出现显著跳跃
        variance_spikes = [i for i in range(len(local_variances)) if local_variances[i] / global_variance > variance_threshold]
    
        # 只保留前5个方差突变点(更改为[:5]即可)
        variance_spikes = variance_spikes[:]
    
        return variance_spikes, local_variances
    
    def calculate_average_before_spike(self, frequencies, variance_spikes, window_size=20):
        averages = []
        for spike in variance_spikes:
            if spike >= window_size:
                avg = np.mean(frequencies[spike - window_size:spike])  # 计算前20个点的均值
                averages.append(avg)
            else:
                averages.append(None)
        return averages
    
    def save_results(self, frequencies, amplitudes, peaks, changes, variance_spikes, avg_frequencies, averages_before_change, averages_before_spike):
        # 将位移数据和频率数据保存到 CSV 文件
        csv_output_path = os.path.join(self.output_dir, "displacement_data.csv")
        df = pd.DataFrame({
            'Frame': range(len(self.displacements)),
            'Displacement X': [d[0] for d in self.displacements],
            'Displacement Y': [d[1] for d in self.displacements],
            'Total Displacement': np.sqrt(np.array([d[0] for d in self.displacements]) ** 2 + np.array([d[1] for d in self.displacements]) ** 2),
            'Time (s)': [i / self.fps for i in range(len(self.displacements))],
            'Frequency (Hz)': np.pad(frequencies, (len(self.displacements) - len(frequencies), 0), 'constant'),
            'Amplitude': np.pad(amplitudes, (len(self.displacements) - len(amplitudes), 0), 'constant')
        })
        df.to_csv(csv_output_path, index=False)

        # 使用 plotly 创建交互式图表
        fig = go.Figure()

        smoothed_displacements = gaussian_filter1d(np.sqrt(np.array([d[0] for d in self.displacements]) ** 2 + np.array([d[1] for d in self.displacements]) ** 2), sigma=5)
        fig.add_trace(go.Scatter(
            x=[i / self.fps for i in range(len(smoothed_displacements))],
            y=smoothed_displacements,
            mode='lines',
            name='Smoothed Displacement',
            line=dict(color='blue', width=1)
        ))

        # 标记频率突变点（红色）
        for i, change in enumerate(changes):
            change_time = peaks[change] / self.fps
            max_y = max(smoothed_displacements)

            fig.add_trace(go.Scatter(
                x=[change_time, change_time],
                y=[0, max_y],
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=True,
                name='Change Point Line'
            ))

            if averages_before_change[i] is not None:
                avg_y = averages_before_change[i]
                fig.add_trace(go.Scatter(
                    x=[change_time],
                    y=[max_y + 0.1],
                    mode='text',
                    text=[f'Avg_Max: {avg_y:.2f} Hz'],
                    textposition="bottom center",
                    showlegend=False
                ))

        # 标记频率方差突增点（黄色）
        for i, spike in enumerate(variance_spikes):
            spike_time = peaks[spike] / self.fps
            max_y = max(smoothed_displacements)
            
            fig.add_trace(go.Scatter(
                x=[spike_time, spike_time],
                y=[0, max_y],
                mode='lines',
                line=dict(color='yellow', width=1),
                showlegend=True,
                name='Variance Spike'
            ))
            
            if averages_before_spike[i] is not None:
                avg_y = averages_before_spike[i]
                fig.add_trace(go.Scatter(
                    x=[spike_time],
                    y=[max_y + 0.1],
                    mode='text',
                    text=[f'Avg_Max: {avg_y:.2f} Hz'],
                    textposition="bottom center",
                    showlegend=False
                ))

        # 添加频率散点
        fig.add_trace(go.Scatter(
            x=[peaks[i] / self.fps for i in range(len(peaks))],
            y=frequencies,
            mode='markers',
            name='Frequency',
            marker=dict(color='green', size=4, symbol='cross')
        ))

        fig.update_layout(
            title='Displacement and Frequency Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz) / Displacement (px)',
            showlegend=True,
            template='plotly_white'
        )

        html_output_path = os.path.join(self.output_dir, "displacement_frequency_analysis.html")
        fig.write_html(html_output_path)
        print("Processing complete. Output saved.")


if __name__ == "__main__":
    video_path = r'C:\Users\kamiy\Documents\day8 stimulation\X1.25_14M_5s_background+stimulation_2578.avi'
    output_dir = r"C:\Users\kamiy\Documents\results\2578_R"
    roi_x, roi_y, roi_width, roi_height = 883, 316, 65, 33

    processor = VideoProcessor(video_path, output_dir, roi_x, roi_y, roi_width, roi_height)
    processor.process_video()

    frequencies, amplitudes, peaks = processor.analyze_waveform()
    changes, avg_frequencies = processor.detect_frequency_changes(frequencies)
    averages_before_change = processor.calculate_average_before_change(frequencies, changes)
    variance_spikes, _ = processor.detect_variance_spike(frequencies)
    averages_before_spike = processor.calculate_average_before_spike(frequencies, variance_spikes)

    processor.save_results(frequencies, amplitudes, peaks, changes, variance_spikes, avg_frequencies, averages_before_change, averages_before_spike)