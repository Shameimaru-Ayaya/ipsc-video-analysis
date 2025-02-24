#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:48:18 2025

@author: GitHub@KirisameMarisa-DAZE (master.spark.kirisame.marisa.daze@gmail.com). All rights reserved. © 2021~2025

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
            raise ValueError(f"Failed to open input video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_video_path = os.path.join(self.output_dir, "processed_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 确认 mp4v 编码器
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_writers = {}
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 为每个步骤创建视频写入对象
        for step in ['1-hsv', 'circles']:
            self.video_writers[step] = cv2.VideoWriter(os.path.join(output_dir, f'{step}.mp4'), fourcc, self.fps, frame_size) # 彩色图不用额外设置
        for step in ['2-mask', '3-opening', '4-bila', '5-edges']:
            self.video_writers[step] = cv2.VideoWriter(os.path.join(output_dir, f'{step}.mp4'), fourcc, self.fps, frame_size, isColor=False) # 由于灰度图是单通道，确保这里设置为False


        # 卡尔曼滤波器初始化（用于平滑圆心的运动）
        self.kf = cv2.KalmanFilter(4, 2)  # 4个状态变量，2个观测变量
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1.0  # 增加测量噪声
        self.kf.statePre = np.array([0, 0, 0, 0], np.float32)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        # 转换为HSV空间
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.video_writers['1-hsv'].write(hsv_frame)

        # 创建掩膜，识别黑色区域
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)
        self.video_writers['2-mask'].write(mask)

        # 形态学开运算，去除小的噪声点
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self.video_writers['3-opening'].write(opening)

        # 双边滤波消除噪声
        bila = cv2.bilateralFilter(opening, 10, 200, 200)
        self.video_writers['4-bila'].write(bila)

        # 边缘识别
        edges = cv2.Canny(bila, 50, 100)
        self.video_writers['5-edges'].write(edges)
        # 创建一个彩色副本用于绘制圆形
        edges_with_circles = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 提取ROI区域
        roi = edges[self.roi_y:self.roi_y + self.roi_height, self.roi_x:self.roi_x + self.roi_width]

        # 在输出帧上绘制ROI区域
        cv2.rectangle(frame, (self.roi_x, self.roi_y),
                      (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                      (255, 0, 0), 2)  # 绘制ROI区域为蓝色矩形
        cv2.rectangle(edges_with_circles, (self.roi_x, self.roi_y),
                      (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                      (255, 0, 0), 2)  # 绘制ROI区域为蓝色矩形

        # 寻找ROI内最大轮廓
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)  # 找到最大轮廓

        if max_contour is not None:
            cv2.drawContours(frame, [max_contour + (self.roi_x, self.roi_y)], -1, (0, 255, 0), 2)  # 绘制轮廓为绿色
            cv2.drawContours(edges_with_circles, [max_contour + (self.roi_x, self.roi_y)], -1, (0, 255, 0), 2)  # 绘制轮廓为绿色

        # 计算最大轮廓的重心
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            centroid = (M["m10"] / M["m00"] + self.roi_x, M["m01"] / M["m00"] + self.roi_y)
        else:
            centroid = None

        # 使用霍夫圆变换检测ROI区域内的圆形
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                    param1=50, param2=30, minRadius=10, maxRadius=100)

        predicted_center = centroid # 用于存储圆心或重心位置

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for circle in circles:
                # 提取圆心和半径
                center = (circle[0] + self.roi_x, circle[1] + self.roi_y)  # 圆心坐标相对于整个图像
                radius = circle[2]

                # 检查半径是否在合理范围内，这里假设合理范围是[20, 40]
                if 30 <= radius <= 40:
                    # 应用卡尔曼滤波来平滑圆心坐标
                    self.kf.correct(np.array([center[0], center[1]], np.float32))  # 更新卡尔曼滤波器的测量值
                    predicted = self.kf.predict()  # 预测下一个位置
                    '''
                    predicted_center = (predicted[0], predicted[1])

                    # 结合当前检测结果和卡尔曼预测值
                    predicted_center = self.smooth_prediction(center, predicted_center)
                    '''

                    # 在原始图像中绘制识别到的圆
                    cv2.circle(frame, center, radius, (255, 255, 255), 2)
                    cv2.circle(frame, center, 2, (0, 0, 255), 3)

                    # 在原始图像中绘制平滑处理的圆
                    cv2.circle(frame, (int(predicted_center[0]), int(predicted_center[1])), radius, (0, 255, 0), 2)
                    cv2.circle(frame, (int(predicted_center[0]), int(predicted_center[1])), 2, (255, 0, 0), 3)  # 绘制圆心

                    # 在edges副本上绘制识别到的圆
                    cv2.circle(edges_with_circles, center, radius, (255, 255, 255), 2)
                    cv2.circle(edges_with_circles, center, 2, (0, 0, 255), 3)

                    # 记录圆心的位移
                    self.track_displacement(predicted_center)
                else:
                    print(f"Frame {frame_index}: Ignored circle with radius {radius} as it's out of expected range.")

        else:
            # 如果没有检测到圆形，使用重心数据
            if centroid is not None:
                print(f"Frame {frame_index}: No circle")
                self.kf.correct(np.array([centroid[0], centroid[1]], np.float32))
                predicted = self.kf.predict()
                predicted_center = (predicted[0], predicted[1])
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (255, 0, 0), 3)  # 绘制重心
                cv2.circle(edges_with_circles, (int(centroid[0]), int(centroid[1])), 2, (255, 0, 0), 3)  # 绘制重心

        self.video_writers['circles'].write(edges_with_circles)

        return frame

    def smooth_prediction(self, detected_center, predicted_center):
        # 结合当前检测值和卡尔曼滤波器预测值
        # 权重可以根据实际情况进行调整，这里使用0.6:0.4的比例
        weight_detected = 0.85
        weight_predicted = 0.15

        smoothed_center = (
            weight_detected * np.array(detected_center) + weight_predicted * np.array(predicted_center)
        )
        return tuple(smoothed_center)

    def track_displacement(self, center):
        # 记录圆心的位移
        displacement_x = center[0] - (self.roi_x + self.roi_width / 2)
        displacement_y = center[1] - (self.roi_y + self.roi_height / 2)
        self.displacements.append((displacement_x, displacement_y))

    def process_video(self):
        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(self.total_frames), desc="Processing video frames"):
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed_frame = executor.submit(self.process_frame, frame, i).result()
                if processed_frame is not None:
                    self.video_writer.write(processed_frame)

        self.cap.release()
        self.video_writer.release()
        for writer in self.video_writers.values():
            writer.release()

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

        # 平滑后的位移图
        smoothed_displacements = gaussian_filter1d(np.sqrt(np.array([d[0] for d in self.displacements]) ** 2 + np.array([d[1] for d in self.displacements]) ** 2), sigma=5)
        fig.add_trace(go.Scatter(
            x=[i / self.fps for i in range(len(smoothed_displacements))],
            y=smoothed_displacements,
            mode='lines',
            name='Smoothed Displacement',
            line=dict(color='blue', width=1)
        ))

        # 绘制使用直接识别到的位移数据（未平滑）
        raw_displacements = np.sqrt(np.array([d[0] for d in self.displacements]) ** 2 + np.array([d[1] for d in self.displacements]) ** 2)
        fig.add_trace(go.Scatter(
            x=[i / self.fps for i in range(len(raw_displacements))],
            y=raw_displacements,
            mode='lines',
            name='Raw Displacement',
            line=dict(color='orange', width=1, dash='dash')
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
    video_path = os.path.normpath(os.path.expanduser(input('Input video path >>>').strip().strip('"\'')))
    output_dir = os.path.normpath(os.path.expanduser(input('Output directory >>>').strip().strip('"\'')))
    roi_x, roi_y, roi_width, roi_height = map(int, input('ROI parameters: x y w h (separated by spaces) >>> ').split())

    processor = VideoProcessor(video_path, output_dir, roi_x, roi_y, roi_width, roi_height)
    processor.process_video()

    frequencies, amplitudes, peaks = processor.analyze_waveform()
    changes, avg_frequencies = processor.detect_frequency_changes(frequencies)
    averages_before_change = processor.calculate_average_before_change(frequencies, changes)
    variance_spikes, _ = processor.detect_variance_spike(frequencies)
    averages_before_spike = processor.calculate_average_before_spike(frequencies, variance_spikes)

    processor.save_results(frequencies, amplitudes, peaks, changes, variance_spikes, avg_frequencies, averages_before_change, averages_before_spike)