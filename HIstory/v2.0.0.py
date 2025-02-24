# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:06:41 2024

@author: kamiyama shiki
"""

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, roi_x: int, roi_y: int, roi_width: int, roi_height: int, threshold=0.5, variance_threshold=1.5, progress_callback=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.threshold = threshold
        self.variance_threshold = variance_threshold + 1
        self.displacements = []
        self.progress_callback = progress_callback  # 添加进度回调
        self.cap = cv2.VideoCapture(self.video_path)
        self.processing_speed = 96178

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

            # 计算x和y方向的位移，不再取整
            displacement_x = cx - (self.roi_x + self.roi_width / 2)
            displacement_y = cy - (self.roi_y + self.roi_height / 2)

            # 记录位移向量（浮点数位移）
            self.displacements.append((displacement_x, displacement_y))
                
            # 在帧上绘制质心
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        else:
            self.displacements.append((0, 0))

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
    
    def detect_frequency_changes(self, frequencies, window_size=5):
        avg_frequencies = np.convolve(frequencies, np.ones(window_size) / window_size, mode='valid')
        changes = []
        for i in range(1, len(avg_frequencies)):
            if abs(avg_frequencies[i] - avg_frequencies[i - 1]) > self.threshold:
                changes.append(i + window_size - 1)  # Mark spike point
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
    
    def detect_variance_spike(self, frequencies, window_size=5):
        global_variance = np.var(frequencies)
        local_variances = [np.var(frequencies[max(0, i - window_size):i]) for i in range(window_size, len(frequencies))]
        variance_spikes = [i for i in range(len(local_variances)) if local_variances[i] / global_variance > self.variance_threshold]
        return variance_spikes[:], local_variances
    
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
            title=f'Displacement and Frequency Analysis ROI:x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz) / Displacement (px)',
            showlegend=True,
            template='plotly_white'
        )

        html_output_path = os.path.join(self.output_dir, "displacement_frequency_analysis.html")
        fig.write_html(html_output_path)

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频处理器（遇到卡死退出重启即可）")

        # 视频路径选择
        tk.Label(root, text="视频路径:").grid(row=0, column=0)
        self.video_path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.video_path_var, width=50).grid(row=0, column=1)
        tk.Button(root, text="选择视频路径", command=self.select_video).grid(row=0, column=2)

        # 输出路径选择
        tk.Label(root, text="输出路径:").grid(row=1, column=0)
        self.output_path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.output_path_var, width=50).grid(row=1, column=1)
        tk.Button(root, text="选择输出路径", command=self.select_output).grid(row=1, column=2)

        # ROI 输入
        tk.Label(root, text="ROI X (默认 0):").grid(row=2, column=0)
        self.roi_x_var = tk.IntVar()
        tk.Entry(root, textvariable=self.roi_x_var, width=10).grid(row=2, column=1, sticky="W")

        tk.Label(root, text="ROI Y (默认 0):").grid(row=2, column=2)
        self.roi_y_var = tk.IntVar()
        tk.Entry(root, textvariable=self.roi_y_var, width=10).grid(row=2, column=3, sticky="W")

        tk.Label(root, text="ROI Width (默认视频宽度):").grid(row=3, column=0)
        self.roi_width_var = tk.IntVar()
        tk.Entry(root, textvariable=self.roi_width_var, width=10).grid(row=3, column=1, sticky="W")

        tk.Label(root, text="ROI Height (默认视频高度):").grid(row=3, column=2)
        self.roi_height_var = tk.IntVar()
        tk.Entry(root, textvariable=self.roi_height_var, width=10).grid(row=3, column=3, sticky="W")

        # 频率变化阈值
        tk.Label(root, text="频率变化阈值 (默认 50%):").grid(row=4, column=0)
        self.threshold_var = tk.DoubleVar(value=0.5)
        tk.Entry(root, textvariable=self.threshold_var, width=10).grid(row=4, column=1, sticky="W")

        # 方差突增阈值
        tk.Label(root, text="方差突增阈值 (默认 50%):").grid(row=4, column=2)
        self.variance_threshold_var = tk.DoubleVar(value=0.5)
        tk.Entry(root, textvariable=self.variance_threshold_var, width=10).grid(row=4, column=3, sticky="W")

        # 进度条
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.start_time = None
        self.time_label = tk.Label(root, text="Elapsed Time: 00:00 | Remaining Time: 00:00")
        self.time_label.grid(row=5, column=2, columnspan=2, pady=10)
        self.progress_label = tk.Label(root, text="处理进度: 0/0 (0.00%) - 平均速度: 0.00 帧/秒")
        self.progress_label.grid(row=6, column=0, columnspan=2, pady=10)

        # 控制按钮
        self.start_button = tk.Button(root, text="开始处理", command=self.run_processing_thread)
        self.start_button.grid(row=7, column=1, pady=10)

        self.pause_button = tk.Button(root, text="暂停处理", state="disabled", command=self.pause_processing)
        self.pause_button.grid(row=7, column=2, pady=10)

        self.stop_button = tk.Button(root, text="终止处理", state="disabled", command=self.stop_processing)
        self.stop_button.grid(row=7, column=3, pady=10)

        # 控制变量
        self.processing_paused = False
        self.processing_stopped = False
        self.processor_thread = None

    def select_video(self):
        path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4;*.avi")])
        if path:
            self.video_path_var.set(path)
            # 打开视频文件以获取其尺寸
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()  # 关闭视频文件
                # 设置ROI为视频的最大大小
                self.roi_x_var.set(0)
                self.roi_y_var.set(0)
                self.roi_width_var.set(width)
                self.roi_height_var.set(height)
            else:
                messagebox.showerror("错误", "无法打开视频文件。")

    def select_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_path_var.set(path)

    def update_progress(self, current, total, processing_speed):
        """更新进度条，确保在主线程中执行"""
        self.root.after(0, self._update_progress_internal, current, total, processing_speed)
    
    def _update_progress_internal(self, current, total, processing_speed):
        if self.start_time is None:
            self.start_time = time.time()  
        
        self.progress["maximum"] = total
        self.progress["value"] = current

        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / current) * (total - current) if current > 0 else 0

        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        remaining_str = time.strftime("%M:%S", time.gmtime(remaining_time))
        self.time_label.config(text=f"Elapsed Time: {elapsed_str} | Remaining Time: {remaining_str}")
        
        progress_percentage = (current / total) * 100
        elapsed_time = time.time() - self.start_time  # 已过去的时间
        processing_speed = (current + 1) / elapsed_time if elapsed_time > 0 else 0  # 处理速度（帧/秒）
        self.progress_label.config(text=f"处理进度: {current}/{total} ({progress_percentage:.2f}%) - 平均速度: {processing_speed:.2f} 帧/秒")

    def run_processing_thread(self):
        self.start_time = None  
        self.processing_stopped = False
        self.processing_paused = False
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")

        self.processor_thread = threading.Thread(target=self.run_processing, daemon=True)
        self.processor_thread.start()

    def pause_processing(self):
        """暂停处理"""
        if not self.processing_paused:
            self.processing_paused = True
            self.pause_button.config(text="继续处理")
        else:
            self.processing_paused = False
            self.pause_button.config(text="暂停处理")

    def stop_processing(self):
        """终止处理"""
        self.processing_stopped = True
        self.processing_paused = False
        self.pause_button.config(text="暂停处理")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.start_button.config(state="normal")
        
        # 归零进度条和计时器
        self.progress["value"] = 0
        self.start_time = None
        self.time_label.config(text="Elapsed Time: 00:00 | Remaining Time: 00:00")
        
        messagebox.showinfo("终止", "视频处理已终止。")

    def run_processing(self):
        try:
            processor = VideoProcessor(
                video_path=self.video_path_var.get(),
                output_dir=self.output_path_var.get(),
                roi_x=self.roi_x_var.get(),
                roi_y=self.roi_y_var.get(),
                roi_width=self.roi_width_var.get(),
                roi_height=self.roi_height_var.get(),
                threshold=self.threshold_var.get(),
                variance_threshold=self.variance_threshold_var.get(),
                progress_callback=self.update_progress
            )
    
            frame_number = 0
            while frame_number < processor.total_frames:
                if self.processing_stopped:
                    break
    
                while self.processing_paused:
                    time.sleep(0.1)
    
                ret, frame = processor.cap.read()  # 读取当前帧
                if not ret:
                    break
    
                processed_frame = processor.process_frame(frame, frame_number)  # 传入 frame 和 frame_number
                processor.video_writer.write(processed_frame)
    
                self.update_progress(frame_number + 1, processor.total_frames, processor.processing_speed)
                frame_number += 1
    
            if not self.processing_stopped:
                # 处理视频和分析波形
                frequencies, amplitudes, peaks = processor.analyze_waveform()
                changes, avg_frequencies = processor.detect_frequency_changes(frequencies)
                averages_before_change = processor.calculate_average_before_change(frequencies, changes)
                variance_spikes, _ = processor.detect_variance_spike(frequencies)
                averages_before_spike = processor.calculate_average_before_spike(frequencies, variance_spikes)
                processor.save_results(frequencies, amplitudes, peaks, changes, variance_spikes, avg_frequencies, averages_before_change, averages_before_spike)
    
                messagebox.showinfo("完成", "视频处理已完成并保存结果。")
        except Exception as e:
            messagebox.showerror("错误", f"处理时出错: {str(e)}")
        finally:
            self.start_button.config(state="normal")
            self.pause_button.config(state="disabled")
            self.stop_button.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()