#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:16:50 2025

@author: Github@Shameimaru-Ayaya
"""

import sys
import os
import cv2
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                           QFileDialog, QVBoxLayout, QWidget, QProgressBar, QSlider,
                           QHBoxLayout, QComboBox, QAction, QMessageBox, QDialog,
                           QSpinBox, QGridLayout, QDialogButtonBox)

# --- 新增：ROI微调对话框 ---
class RoiTuningDialog(QDialog):
    def __init__(self, initial_roi, max_width, max_height, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ROI 微调')
        
        layout = QGridLayout(self)
        
        # 创建标签和SpinBox控件
        self.x_spinbox = QSpinBox()
        self.y_spinbox = QSpinBox()
        self.w_spinbox = QSpinBox()
        self.h_spinbox = QSpinBox()
        
        # 设置SpinBox的范围
        self.x_spinbox.setRange(0, max_width)
        self.y_spinbox.setRange(0, max_height)
        self.w_spinbox.setRange(0, max_width)
        self.h_spinbox.setRange(0, max_height)
        
        # 设置初始值
        self.x_spinbox.setValue(initial_roi.x())
        self.y_spinbox.setValue(initial_roi.y())
        self.w_spinbox.setValue(initial_roi.width())
        self.h_spinbox.setValue(initial_roi.height())

        # 添加到布局
        layout.addWidget(QLabel('X 坐标:'), 0, 0)
        layout.addWidget(self.x_spinbox, 0, 1)
        layout.addWidget(QLabel('Y 坐标:'), 1, 0)
        layout.addWidget(self.y_spinbox, 1, 1)
        layout.addWidget(QLabel('宽度 (W):'), 2, 0)
        layout.addWidget(self.w_spinbox, 2, 1)
        layout.addWidget(QLabel('高度 (H):'), 3, 0)
        layout.addWidget(self.h_spinbox, 3, 1)

        # 添加OK和Cancel按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box, 4, 0, 1, 2)

    def get_roi(self):
        # 从SpinBox获取值并返回一个新的QRect
        return QRect(
            self.x_spinbox.value(),
            self.y_spinbox.value(),
            self.w_spinbox.value(),
            self.h_spinbox.value()
        )

class VideoLabel(QLabel):
    clicked = pyqtSignal()
    roi_selected = pyqtSignal(QRect)
    HANDLE_SIZE = 10

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText('拖放视频文件到这里，或点击选择文件')
        self.setMinimumSize(640, 480)
        self.drag_mode = None
        self.start_point = QPoint()
        self.current_roi = QRect()
        self.permanent_roi = QRect()
        self.original_size = QSize()
        self.display_rect = QRect()
        self.pen = QPen(Qt.red, 2, Qt.SolidLine)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)  # 启用拖放功能

    def set_video_frame(self, pixmap, original_size):
        self.original_size = original_size
        scaled_pix = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pix)
        pw = scaled_pix.width()
        ph = scaled_pix.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        self.display_rect = QRect(x, y, pw, ph)
        self.update() # 确保在加载新视频时重绘

    def get_scaled_roi(self):
        if self.original_size.width() == 0 or self.original_size.height() == 0 or self.permanent_roi.isNull():
            return QRect()
        scale_x = self.display_rect.width() / self.original_size.width()
        scale_y = self.display_rect.height() / self.original_size.height()
        return QRect(
            int(self.permanent_roi.x() * scale_x) + self.display_rect.x(),
            int(self.permanent_roi.y() * scale_y) + self.display_rect.y(),
            int(self.permanent_roi.width() * scale_x),
            int(self.permanent_roi.height() * scale_y)
        )

    def get_handle_rect(self, center):
        size = self.HANDLE_SIZE
        return QRect(center.x() - size//2, center.y() - size//2, size, size)

    def get_handles(self):
        scaled_roi = self.get_scaled_roi()
        if scaled_roi.isNull():
            return {}
        handles = {
            'topleft': scaled_roi.topLeft(),
            'top': QPoint(scaled_roi.center().x(), scaled_roi.top()),
            'topright': scaled_roi.topRight(),
            'right': QPoint(scaled_roi.right(), scaled_roi.center().y()),
            'bottomright': scaled_roi.bottomRight(),
            'bottom': QPoint(scaled_roi.center().x(), scaled_roi.bottom()),
            'bottomleft': scaled_roi.bottomLeft(),
            'left': QPoint(scaled_roi.left(), scaled_roi.center().y())
        }
        return handles

    def handle_at(self, pos):
        handles = self.get_handles()
        for key, center in handles.items():
            if self.get_handle_rect(center).contains(pos):
                return key
        scaled_roi = self.get_scaled_roi()
        if not scaled_roi.isNull() and scaled_roi.adjusted(-5, -5, 5, 5).contains(pos) and not scaled_roi.adjusted(5, 5, -5, -5).contains(pos):
            return 'move'
        if not scaled_roi.isNull() and scaled_roi.contains(pos):
            return 'move'
        return None

    def mousePressEvent(self, event):
        if not self.pixmap():
            self.clicked.emit()
            return
        if not self.pixmap() or not self.display_rect.contains(event.pos()):
            return
        handle = self.handle_at(event.pos())
        if handle:
            self.drag_mode = handle
            self.start_point = event.pos()
            self.original_roi = QRect(self.permanent_roi)
        else:
            self.drag_mode = 'create'
            self.start_point = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(self.start_point, self.start_point)
            # 在开始创建时清除旧的ROI
            self.permanent_roi = QRect()
            self.roi_selected.emit(self.permanent_roi)


    def mouseMoveEvent(self, event):
        if not self.pixmap():
            return
        if self.drag_mode == 'create':
            current_pos = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(self.start_point, current_pos).normalized()
            self.update()
        elif self.drag_mode:
            self.handle_roi_resize(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_mode == 'create':
            scale_x = self.original_size.width() / self.display_rect.width()
            scale_y = self.original_size.height() / self.display_rect.height()
            self.permanent_roi = QRect(
                int(self.current_roi.x() * scale_x),
                int(self.current_roi.y() * scale_y),
                int(self.current_roi.width() * scale_x),
                int(self.current_roi.height() * scale_y)
            ).normalized()
            self.roi_selected.emit(self.permanent_roi)
        self.drag_mode = None
        self.update()

    def handle_roi_resize(self, pos):
        if not self.drag_mode:
            return
        
        scale_x = self.original_size.width() / self.display_rect.width()
        scale_y = self.original_size.height() / self.display_rect.height()
        
        dx = (pos.x() - self.start_point.x()) * scale_x
        dy = (pos.y() - self.start_point.y()) * scale_y
        
        new_roi = QRect(self.original_roi)
        
        if self.drag_mode == 'move':
            new_roi.translate(int(dx), int(dy))
        else:
            if 'left' in self.drag_mode:
                new_roi.setLeft(new_roi.left() + int(dx))
            elif 'right' in self.drag_mode:
                new_roi.setRight(new_roi.right() + int(dx))
            if 'top' in self.drag_mode:
                new_roi.setTop(new_roi.top() + int(dy))
            elif 'bottom' in self.drag_mode:
                new_roi.setBottom(new_roi.bottom() + int(dy))
        
        if new_roi.width() > 0 and new_roi.height() > 0:
            self.permanent_roi = new_roi.normalized()
            self.roi_selected.emit(self.permanent_roi)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap():
            return
        
        painter = QPainter(self)
        painter.setPen(self.pen)
        
        if self.drag_mode == 'create':
            painter.drawRect(self.current_roi.translated(self.display_rect.topLeft()))
        
        scaled_roi = self.get_scaled_roi()
        if not scaled_roi.isNull():
            painter.drawRect(scaled_roi)
            handles = self.get_handles()
            for center in handles.values():
                painter.drawRect(self.get_handle_rect(center))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = str(urls[0].toLocalFile())
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.parent().parent().load_video(file_path)
                event.acceptProposedAction()

class VideoProcessor(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)
    speed_updated = pyqtSignal(float)

    def __init__(self, video_path, output_dir, roi, threshold=0.5, variance_threshold=5, is_overflow=False, debug_output=False, output_intermediate_frames=False):
        super().__init__()
        self.video_path = video_path
        self.base_output_dir = output_dir  # 保存原始输出目录作为基础目录
        self.roi_x, self.roi_y, self.roi_width, self.roi_height = roi
        self.threshold = threshold
        self.variance_threshold = variance_threshold
        self.cap = cv2.VideoCapture(video_path)
        self.displacements = []
        self.processing_speed = 96178
        self.is_overflow = is_overflow
        self.debug_output = debug_output
        self.output_intermediate_frames = output_intermediate_frames

        # 获取视频文件名（不含扩展名）作为输出目录名
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = os.path.join(self.base_output_dir, video_filename)

        # 创建输出目录结构
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 先初始化视频属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_video_path = os.path.join(self.output_dir, "processed_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                          (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.processing_times = []  # 用于存储每帧处理时间
        
        # 记录初始化参数
        self._log_command(f"INIT {self.video_path}", f"ROI: {roi}, 阈值: {threshold}, 溢出模式: {is_overflow}")

        if self.debug_output:
            # 创建调试输出目录（作为二级子文件夹）
            self.debug_dir = os.path.join(self.output_dir, "debug_output")
            self.debug_frames_dir = os.path.join(self.debug_dir, "frames")
            self.debug_videos_dir = os.path.join(self.debug_dir, "videos")
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(self.debug_frames_dir, exist_ok=True)
            os.makedirs(self.debug_videos_dir, exist_ok=True)
            
            # 创建日志文件
            log_path = os.path.join(self.debug_dir, "process.log")
            cmd_path = os.path.join(self.debug_dir, "commands.log")
            
            # 检查日志文件是否已存在，如果存在则使用追加模式
            if os.path.exists(log_path):
                self.log_file = open(log_path, "a")
                self._log_info("\n" + "="*50 + "\n")
                self._log_info("继续记录日志")
            else:
                self.log_file = open(log_path, "w")
                
            # 检查命令文件是否已存在，如果存在则使用追加模式
            if os.path.exists(cmd_path):
                self.cmd_file = open(cmd_path, "a")
                self.cmd_file.write("\n" + "="*50 + "\n")
            else:
                self.cmd_file = open(cmd_path, "w")

            # 记录初始化参数
            self._log_command("INIT", f"视频: {self.video_path}, ROI: {roi}, 阈值: {threshold}, 溢出模式: {is_overflow}")

            # 输出系统信息
            self._log_info("系统信息:")
            self._log_info(f"Python版本: {sys.version}")
            self._log_info(f"OpenCV版本: {cv2.__version__}")
            self._log_info(f"NumPy版本: {np.__version__}")
            self._log_info(f"Pandas版本: {pd.__version__}")
            
            # 输出视频信息
            self._log_info("视频信息:")
            self._log_info(f"路径: {self.video_path}")
            self._log_info(f"帧率: {self.fps}")
            self._log_info(f"总帧数: {self.total_frames}")
            self._log_info(f"分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            # 输出处理参数
            self._log_info("处理参数:")
            self._log_info(f"ROI: ({self.roi_x}, {self.roi_y}, {self.roi_width}, {self.roi_height})")
            self._log_info(f"频率变化阈值: {self.threshold}")
            self._log_info(f"方差阈值: {self.variance_threshold}")
            self._log_info(f"溢出模式: {self.is_overflow}")

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        # 记录开始时间
        start_time = time.time()
        
        # 创建调试帧列表（如果启用调试输出）
        debug_frames = [] if self.debug_output else None
        
        # 记录处理开始
        self._log_info(f"开始处理第 {frame_index} 帧")
            
        # 检查帧是否为空
        if frame is None or frame.size == 0:
            self._log_error(f"第 {frame_index} 帧为空或无效")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        if self.debug_output:
            debug_frames.append(("original", frame.copy()))
            self._log_debug(f"帧尺寸: {frame.shape}")
            
        try:
            # 转换为HSV空间
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.debug_output:
                debug_frames.append(("HSV", hsv_frame))
                self._log_debug("已转换为HSV空间")
            
            # 创建掩膜，识别黑色区域
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask = cv2.inRange(hsv_frame, lower_black, upper_black)
            if self.debug_output:
                debug_frames.append(("mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
                self._log_debug(f"掩膜统计: 黑色像素数量: {np.sum(mask > 0)}")
            
            # 形态学操作改进：先进行开运算去噪，再进行闭运算填充小孔
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if self.debug_output:
                debug_frames.append(("opening", cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            if self.debug_output:
                debug_frames.append(("closing", cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)))
            
            # 使用中值滤波
            median = cv2.medianBlur(closing, 5)
            if self.debug_output:
                debug_frames.append(("median", cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)))
            
            # 根据溢出标志决定是否进行边缘检测
            if not self.is_overflow:
                # 边缘检测
                edges = cv2.Canny(median, 50, 150)
                roi_source = edges
                if self.debug_output:
                    debug_frames.append(("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
            else:
                roi_source = median
                
            # 创建彩色显示图像
            display_frame = frame.copy()
            
            # 改进后的ROI处理部分
            try:
                # 计算扩展后的ROI边界（不扩展超过图像范围）
                expanded_x1 = max(0, self.roi_x - 1)
                expanded_y1 = max(0, self.roi_y - 1)
                expanded_x2 = min(frame.shape[1], self.roi_x + self.roi_width + 1)
                expanded_y2 = min(frame.shape[0], self.roi_y + self.roi_height + 1)
        
                # 提取扩展后的ROI区域
                roi_expanded = roi_source[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
                
                if roi_expanded.size == 0:
                    self._log_warning(f"警告：扩展ROI区域为空，帧索引：{frame_index}")
                    return frame
        
                # 创建原始ROI对应的掩膜（不包含扩展区域）
                mask = np.zeros_like(roi_expanded)
                original_in_expanded_x = self.roi_x - expanded_x1
                original_in_expanded_y = self.roi_y - expanded_y1
                mask[original_in_expanded_y:original_in_expanded_y+self.roi_height,
                     original_in_expanded_x:original_in_expanded_x+self.roi_width] = 255
        
                # 应用掩膜（仅保留原始ROI范围内的区域）
                roi_masked = cv2.bitwise_and(roi_expanded, roi_expanded, mask=mask.astype(np.uint8))
                
                # 进行边界填充以保持轮廓闭合
                padded_roi = cv2.copyMakeBorder(roi_masked, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                
            except Exception as e:
                self._log_warning(f"ROI提取错误：{str(e)}，帧索引：{frame_index}")
                return frame
                
            # 改进后的轮廓处理部分
            contours, _ = cv2.findContours(padded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # 调整轮廓坐标时考虑边界填充和扩展区域
            adjusted_contours = []
            for contour in contours:
                adjusted_contour = contour.copy()
                adjusted_contour[:,:,0] += (expanded_x1 - 1)
                adjusted_contour[:,:,1] += (expanded_y1 - 1)
                adjusted_contours.append(adjusted_contour)
            
            # 添加更多的错误检查和处理
            try:
                if adjusted_contours:
                    # 先检查是否有轮廓
                    contour_areas = [cv2.contourArea(cnt) for cnt in adjusted_contours]
                    valid_contours = [cnt for i, cnt in enumerate(adjusted_contours) if contour_areas[i] > 100]
                    
                    if valid_contours:
                        max_contour = max(valid_contours, key=cv2.contourArea)
                        
                        # 绘制最大轮廓
                        cv2.drawContours(display_frame, [max_contour], -1, (0, 255, 0), 2)
                        
                        M = cv2.moments(max_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                            self.displacements.append((cx - (self.roi_x + self.roi_width/2),
                                                   cy - (self.roi_y + self.roi_height/2)))
                        else:
                            self._log_warning(f"警告：轮廓重心计算失败，帧索引：{frame_index}")
                            self.displacements.append((0, 0))
                    else:
                        self._log_warning(f"警告：没有面积大于阈值的有效轮廓，帧索引：{frame_index}")
                        self.displacements.append((0, 0))
                else:
                    self._log_warning(f"警告：未检测到轮廓，帧索引：{frame_index}")
                    self.displacements.append((0, 0))
            except Exception as e:
                self._log_warning(f"轮廓处理错误：{str(e)}，帧索引：{frame_index}")
                self.displacements.append((0, 0))
            
            # 如果启用调试输出，保存调试图像
            if self.debug_output and debug_frames:  # 添加空列表检查
                self._save_debug_frame(debug_frames, frame_index)
                
                # 创建轮廓显示图像
                contour_image = np.zeros_like(frame)
                if adjusted_contours:
                    cv2.drawContours(contour_image, adjusted_contours, -1, (0, 255, 0), 2)
                debug_frames.append(("contours", contour_image))
                
                # 添加最终处理结果图像
                debug_frames.append(("result", display_frame.copy()))
                
                # 再次保存包含所有图像的调试帧
                self._save_debug_frame(debug_frames, frame_index)
            
            # 计算处理时间并更新平均速度
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            avg_time = sum(self.processing_times) / len(self.processing_times)
            self.speed_updated.emit(avg_time * 1000)
            
            self._log_info(f"第 {frame_index} 帧处理完成，耗时: {process_time:.4f}秒")
            
            return display_frame
        except Exception as e:
            self._log_error(f"处理第 {frame_index} 帧时发生错误: {str(e)}")
            import traceback
            self._log_error(traceback.format_exc())
            return frame
        
        # 如果启用调试输出，保存调试图像
        if self.debug_output and debug_frames:
            self._save_debug_frame(debug_frames, frame_index)
        
        # 计算处理时间并更新平均速度
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        avg_time = sum(self.processing_times) / len(self.processing_times)
        self.speed_updated.emit(avg_time * 1000)
        
        self._log_info(f"第 {frame_index} 帧处理完成，耗时: {process_time:.4f}秒")
        
        return display_frame

    def find_largest_contour(self, contours):
        max_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        return max_contour

    def track_displacement(self, frame, max_contour):
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"] + self.roi_x
            cy = M["m01"] / M["m00"] + self.roi_y

            displacement_x = cx - (self.roi_x + self.roi_width / 2)
            displacement_y = cy - (self.roi_y + self.roi_height / 2)

            self.displacements.append((displacement_x, displacement_y))
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        else:
            self.displacements.append((0, 0))

    def analyze_waveform(self):
        displacements_x = [d[0] for d in self.displacements]
        displacements_y = [d[1] for d in self.displacements]
        
        smoothed_displacements_x = gaussian_filter1d(displacements_x, sigma=5)
        smoothed_displacements_y = gaussian_filter1d(displacements_y, sigma=5)
        
        peaks = []
        frequencies = []
        amplitudes = []
        
        total_displacements = np.sqrt(np.array(smoothed_displacements_x) ** 2 + np.array(smoothed_displacements_y) ** 2)

        for i in range(1, len(total_displacements) - 1):
            if total_displacements[i] > total_displacements[i - 1] and total_displacements[i] > total_displacements[i + 1]:
                peaks.append(i)
                if len(peaks) > 1:
                    freq = self.fps / (peaks[-1] - peaks[-2])
                    amplitude = total_displacements[peaks[-1]] - min(total_displacements[peaks[-2]:peaks[-1]])
                    frequencies.append(freq)
                    amplitudes.append(amplitude)
        return frequencies, amplitudes, peaks

    def detect_frequency_changes(self, frequencies, window_size=5):
        avg_frequencies = np.convolve(frequencies, np.ones(window_size) / window_size, mode='valid')
        changes = []
        for i in range(1, len(avg_frequencies)):
            if abs(avg_frequencies[i] - avg_frequencies[i - 1]) > self.threshold:
                changes.append(i + window_size - 1)
        return changes, avg_frequencies

    def calculate_average_before_change(self, frequencies, changes, window_size=20):
        averages = []
        for change in changes:
            if change >= window_size:
                avg = np.mean(frequencies[change - window_size:change])
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
                avg = np.mean(frequencies[spike - window_size:spike])
                averages.append(avg)
            else:
                averages.append(None)
        return averages

    def save_results(self, frequencies, amplitudes, peaks, changes, variance_spikes, avg_frequencies, averages_before_change, averages_before_spike):
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

        fig = go.Figure()
        smoothed_displacements = gaussian_filter1d(np.sqrt(np.array([d[0] for d in self.displacements]) ** 2 + np.array([d[1] for d in self.displacements]) ** 2), sigma=5)
        
        fig.add_trace(go.Scatter(
            x=[i / self.fps for i in range(len(smoothed_displacements))],
            y=smoothed_displacements,
            mode='lines',
            name='平滑位移',
            line=dict(color='blue', width=1)
        ))

        for i, change in enumerate(changes):
            change_time = peaks[change] / self.fps
            fig.add_trace(go.Scatter(
                x=[change_time],
                y=[smoothed_displacements[peaks[change]]],
                mode='markers',
                name=f'频率变化点 {i+1}',
                marker=dict(color='red', size=10)
            ))

        fig.update_layout(
            title='位移和频率变化分析',
            xaxis_title='时间 (秒)',
            yaxis_title='位移',
            showlegend=True
        )
        
        fig.write_html(os.path.join(self.output_dir, 'displacement_analysis.html'))

    def run(self):
        self._log_info(f"开始处理视频: {self.video_path}")
        self._log_command("START_PROCESSING", f"开始处理视频: {os.path.basename(self.video_path)}")
        self._log_info(f"总帧数: {self.total_frames}, FPS: {self.fps}")
        self._log_info(f"ROI区域: ({self.roi_x}, {self.roi_y}, {self.roi_width}, {self.roi_height})")
        self._log_info(f"阈值设置: 频率变化阈值={self.threshold}, 方差阈值={self.variance_threshold}")
        self._log_info(f"溢出模式: {self.is_overflow}, 调试输出: {self.debug_output}")
        
        try:
            frame_index = 0
            while True:
                # 检查是否被终止
                if self.isInterruptionRequested():
                    self._log_info("处理被用户终止")
                    self._log_command("STOP_PROCESSING", "用户手动终止")
                    self.finished.emit("处理被用户终止")
                    return
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                try:
                    processed_frame = self.process_frame(frame, frame_index)
                    self.video_writer.write(processed_frame)
                except Exception as e:
                    self._log_error(f"处理第 {frame_index} 帧时发生错误: {str(e)}")
                    import traceback
                    self._log_error(traceback.format_exc())
                    # 继续处理下一帧，而不是中断整个处理
                    self.video_writer.write(frame)  # 写入原始帧
                
                # 更新进度
                progress = int((frame_index + 1) / self.total_frames * 100)
                self.progress_updated.emit(progress)
                
                frame_index += 1
                
            # 处理完成后的分析
            self._log_info("视频处理完成，开始分析波形...")
            self._log_command("ANALYZE_WAVEFORM", "开始波形分析")
            
            try:
                frequencies, amplitudes, peaks = self.analyze_waveform()
                changes, avg_frequencies = self.detect_frequency_changes(frequencies)
                variance_spikes, local_variances = self.detect_variance_spike(frequencies)
                
                averages_before_change = self.calculate_average_before_change(frequencies, changes)
                averages_before_spike = self.calculate_average_before_spike(frequencies, variance_spikes)
                
                self._log_command(f"SAVE_RESULTS {self.output_dir}", "保存分析结果")
                self.save_results(frequencies, amplitudes, peaks, changes, variance_spikes,
                                avg_frequencies, averages_before_change, averages_before_spike)
                
                self._log_info(f"分析完成，结果已保存到: {self.output_dir}")
                self._log_command("SAVE_RESULTS", f"保存分析结果到 {self.output_dir}")
                self.finished.emit(self.output_dir)
                self._log_command("FINISH_PROCESSING", "处理流程正常完成")
            except Exception as e:
                self._log_error(f"波形分析时发生错误: {str(e)}")
                import traceback
                self._log_error(traceback.format_exc())
                self.finished.emit(f"错误: 波形分析失败 - {str(e)}")
            
        except Exception as e:
            self._log_error(f"处理视频时发生错误: {str(e)}")
            import traceback
            self._log_error(traceback.format_exc())
            self.finished.emit(f"错误: {str(e)}")
        finally:
            # 释放资源
            try:
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                if hasattr(self, 'video_writer') and self.video_writer:
                    self.video_writer.release()
                if self.debug_output:
                    if hasattr(self, 'log_file') and self.log_file:
                        self.log_file.close()
                    if hasattr(self, 'cmd_file') and self.cmd_file:
                        self.cmd_file.close()
            except Exception as e:
                print(f"释放资源时发生错误: {str(e)}")

    def _log_info(self, message):
        """记录信息级别的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[INFO] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()
            
    def _log_command(self, command, description=""):
        """记录命令到命令日志文件"""
        if not self.debug_output or not hasattr(self, 'cmd_file'):
            return
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {command}"
        if description:
            log_entry += f" # {description}"
        
        self.cmd_file.write(log_entry + "\n")
        self.cmd_file.flush()

    def _log_warning(self, message):
        """记录警告级别的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[WARNING] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()

    def _log_error(self, message):
        """记录错误级别的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[ERROR] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()

    def _log_debug(self, message):
        """记录调试级别的日志，仅在调试模式下输出"""
        if self.debug_output:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_message = f"[DEBUG] {timestamp}: {message}"
            print(log_message)
            if hasattr(self, 'log_file'):
                self.log_file.write(log_message + "\n")
                self.log_file.flush()

    def _save_debug_frame(self, debug_frames, frame_index):
        """保存调试帧"""
        if not self.debug_output or not debug_frames:
            return
            
        # 如果不输出中间帧，则完全不保存任何单帧图像
        if not self.output_intermediate_frames:
            return
            
        try:
            # 创建调试帧目录
            frame_dir = os.path.join(self.debug_frames_dir, f"frame_{frame_index:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # 保存每个调试图像
            for name, img in debug_frames:
                # 确保图像是彩色的
                if len(img.shape) == 2:  # 灰度图像
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # 保存图像
                output_path = os.path.join(frame_dir, f"{name}.png")
                cv2.imwrite(output_path, img)
                
            # 记录到日志
            self._log_warning(f"已保存帧 {frame_index} 的调试图像到 {frame_dir}")
        except Exception as e:
            self._log_warning(f"保存调试帧时出错: {str(e)}")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'debug_output') and self.debug_output:
            if hasattr(self, 'log_file'):
                self.log_file.close()
            if hasattr(self, 'cmd_file'):
                self.cmd_file.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('视频处理工具')
        self.video_path = ''
        self.output_dir = os.path.expanduser('~/Desktop')
        self.video_processor = None  # 添加video_processor属性初始化
        self.speed_label = QLabel('平均处理时间: 0.00 ms/帧')
        self.permanent_roi = None  # 添加ROI属性
        self.init_ui()
        self._create_menu_bar() # 创建菜单栏

    def init_ui(self):
        self.video_label = VideoLabel()
        self.video_label.roi_selected.connect(self.update_roi)
        self.video_label.clicked.connect(self.select_video_file)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_label = QLabel()
        self.progress_label.hide()

        self.status_label = QLabel('就绪')
        self.time_label = QLabel('00:00/00:00')

        # 创建水平布局来放置文件操作按钮
        file_buttons_layout = QHBoxLayout()
        
        self.select_dir_btn = QPushButton('选择输出目录')
        self.select_dir_btn.clicked.connect(self.select_output_dir)
        
        self.select_video_btn = QPushButton('重新选择视频')
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.select_video_btn.setEnabled(False)
        
        file_buttons_layout.addWidget(self.select_dir_btn)
        file_buttons_layout.addStretch()  # 添加弹性空间，将重新选择视频按钮推到最右侧
        file_buttons_layout.addWidget(self.select_video_btn)

        # 创建水平布局来放置选项按钮
        options_layout = QHBoxLayout()
        
        # 添加目标溢出选择按钮
        self.overflow_combo = QComboBox()
        self.overflow_combo.addItems(['目标不溢出边界', '目标溢出边界'])
        self.overflow_combo.setFixedWidth(150)
        self.overflow_combo.setStyleSheet("QComboBox { text-align: center; }")
        
        # 添加调试输出选择按钮
        self.debug_output_combo = QComboBox()
        self.debug_output_combo.addItems(['不输出调试信息', '输出调试信息'])
        self.debug_output_combo.setFixedWidth(150)
        self.debug_output_combo.setStyleSheet("QComboBox { text-align: center; }")
        self.debug_output_combo.currentIndexChanged.connect(self.update_intermediate_frames_combo_state)
        
        # 添加中间帧输出选择按钮
        self.intermediate_frames_combo = QComboBox()
        self.intermediate_frames_combo.addItems(['不输出中间帧', '输出中间帧'])
        self.intermediate_frames_combo.setFixedWidth(150)
        self.intermediate_frames_combo.setStyleSheet("QComboBox { text-align: center; }")
        self.intermediate_frames_combo.setEnabled(False)  # 默认不可用
        
        options_layout.addWidget(self.overflow_combo)
        options_layout.addStretch()  # 添加弹性空间，将调试输出按钮推到右侧
        options_layout.addWidget(self.debug_output_combo)
        options_layout.addWidget(self.intermediate_frames_combo)
        
        # 创建水平布局来放置处理速度和预计剩余时间
        speed_layout = QHBoxLayout()
        self.speed_label = QLabel('处理速度: 0.00 秒/帧')
        self.remaining_time_label = QLabel('预计剩余时间: --:--')
        speed_layout.addWidget(self.speed_label)
        speed_layout.addStretch()
        speed_layout.addWidget(self.remaining_time_label)
        
        self.process_btn = QPushButton('开始处理')
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.toggle_processing)
        
        control_layout = QVBoxLayout()
        control_layout.addLayout(file_buttons_layout)  # 文件操作按钮布局
        control_layout.addLayout(options_layout)  # 选项按钮布局
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.progress_label)
        control_layout.addLayout(speed_layout)  # 处理速度和剩余时间布局
        control_layout.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # 文件菜单
        file_menu = menu_bar.addMenu('文件')

        open_action = QAction('打开视频', self)
        open_action.triggered.connect(self.select_video_file)
        file_menu.addAction(open_action)

        output_dir_action = QAction('选择输出目录', self)
        output_dir_action.triggered.connect(self.select_output_dir)
        file_menu.addAction(output_dir_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menu_bar.addMenu('编辑')

        # --- 新增：ROI 微调菜单项 ---
        self.fine_tune_roi_action = QAction('ROI 微调', self)
        self.fine_tune_roi_action.triggered.connect(self.open_roi_tuning_dialog)
        self.fine_tune_roi_action.setEnabled(False) # 默认禁用
        edit_menu.addAction(self.fine_tune_roi_action)

        # 帮助菜单
        help_menu = menu_bar.addMenu('帮助')

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        QMessageBox.about(self, '关于', '视频处理工具\n版本 2.2.0\n作者: Github@Shameimaru-Ayaya')

    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '',
                                                 '视频文件 (*.mp4 *.avi *.mov *.mkv)')
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        self.video_path = file_path
        self.permanent_roi = None # 加载新视频时重置ROI
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.set_video_frame(pixmap, QSize(w, h))
                
                self.select_video_btn.setEnabled(True)
                self.process_btn.setEnabled(False) # 加载视频后需要先选择ROI才能处理
                self.status_label.setText('请选择ROI区域')
                self.fine_tune_roi_action.setEnabled(True) # --- 修改：加载视频后启用菜单 ---
            cap.release()

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, '选择输出目录', self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.status_label.setText(f'输出目录: {dir_path}')

    def update_roi(self, roi):
        self.permanent_roi = roi
        self.video_label.permanent_roi = roi # 确保VideoLabel中的ROI也更新
        self.video_label.update() # 强制重绘
        if roi and not roi.isNull() and roi.isValid():
             self.process_btn.setEnabled(True)  # 选择有效ROI后启用处理按钮
             self.status_label.setText('ROI已选择，可以开始处理')
        else:
             self.process_btn.setEnabled(False)
             self.status_label.setText('请选择有效的ROI区域')

    # --- 新增：打开ROI微调对话框的方法 ---
    def open_roi_tuning_dialog(self):
        video_size = self.video_label.original_size
        if not video_size.isValid():
            return # 如果没有视频，则不执行任何操作

        # 确定初始ROI
        if self.permanent_roi and not self.permanent_roi.isNull():
            initial_roi = self.permanent_roi
        else:
            # 如果没有设置ROI，则默认为整个视频画面
            initial_roi = QRect(0, 0, video_size.width(), video_size.height())

        # 创建并执行对话框
        dialog = RoiTuningDialog(initial_roi, video_size.width(), video_size.height(), self)
        if dialog.exec_() == QDialog.Accepted:
            new_roi = dialog.get_roi()
            self.update_roi(new_roi) # 使用新ROI更新主窗口
            
    def update_intermediate_frames_combo_state(self, index):
        # 当选择输出调试信息时(index=1)，启用中间帧下拉菜单
        self.intermediate_frames_combo.setEnabled(index == 1)

    def toggle_processing(self):
        # 如果当前正在处理，则终止处理
        if hasattr(self, 'video_processor') and self.video_processor and self.video_processor.isRunning():
            self.stop_processing()
        else:
            self.start_processing()
    
    def stop_processing(self):
        try:
            if hasattr(self, 'video_processor') and self.video_processor:
                # 终止处理线程
                self.video_processor.terminate()
                self.video_processor.wait()
                
                # 恢复UI状态
                self.process_btn.setText('开始处理')
                self.process_btn.setEnabled(True)
                self.select_video_btn.setEnabled(True)
                self.select_dir_btn.setEnabled(True)
                self.progress_bar.hide()
                self.progress_label.hide()
                self.status_label.setText('处理已终止')
                self.remaining_time_label.setText('预计剩余时间: --:--')
        except Exception as e:
            self.handle_error(f"终止处理时发生错误: {str(e)}")

    def start_processing(self):
        if not self.video_path or not self.permanent_roi or self.permanent_roi.isNull():
            self.status_label.setText("错误：开始处理前必须选择一个有效的ROI区域！")
            return
            
        try:
            is_overflow = self.overflow_combo.currentText() == '目标溢出边界'
            debug_output = self.debug_output_combo.currentText() == '输出调试信息'
            output_intermediate_frames = self.intermediate_frames_combo.currentText() == '输出中间帧' and debug_output
            
            # 将 QRect 对象转换为元组 (x, y, width, height)
            roi_tuple = (self.permanent_roi.x(), self.permanent_roi.y(), 
                         self.permanent_roi.width(), self.permanent_roi.height())
            
            self.video_processor = VideoProcessor(
                self.video_path, 
                self.output_dir, 
                roi_tuple,  # 使用转换后的元组
                is_overflow=is_overflow,
                debug_output=debug_output,
                output_intermediate_frames=output_intermediate_frames
            )
            self.video_processor.progress_updated.connect(self.update_progress)
            self.video_processor.finished.connect(self.processing_finished)
            self.video_processor.speed_updated.connect(self.update_speed)  # 连接速度更新信号
            
            # 更改按钮文本和状态
            self.process_btn.setText('终止处理')
            self.select_video_btn.setEnabled(False)
            self.select_dir_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.show()
            self.status_label.setText('正在处理...')
            
            # 重置处理时间和剩余时间显示
            self.speed_label.setText('处理速度: 计算中...')
            self.remaining_time_label.setText('预计剩余时间: 计算中...')
            
            # 记录开始时间和已处理帧数，用于计算剩余时间
            self.processing_start_time = time.time()
            self.processed_frames = 0
            
            self.video_processor.start()
        except Exception as e:
            self.handle_error(f"启动处理时发生错误: {str(e)}")

    def update_progress(self, value):
        """更新进度条、进度标签和预计剩余时间"""
        try:
            self.progress_bar.setValue(value)
            self.progress_label.setText(f'处理进度: {value}%')
            
            # 计算预计剩余时间
            if hasattr(self, 'video_processor') and self.video_processor:
                total_frames = self.video_processor.total_frames
                self.processed_frames = int(total_frames * value / 100)
                
                if self.processed_frames > 0:
                    elapsed_time = time.time() - self.processing_start_time
                    frames_per_second = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
                    
                    if frames_per_second > 0:
                        remaining_frames = total_frames - self.processed_frames
                        remaining_seconds = remaining_frames / frames_per_second
                        
                        # 格式化为分:秒
                        minutes = int(remaining_seconds // 60)
                        seconds = int(remaining_seconds % 60)
                        self.remaining_time_label.setText(f'预计剩余时间: {minutes:02d}:{seconds:02d}')
            
            if not self.progress_bar.isVisible():
                self.progress_bar.show()
                self.progress_label.show()
        except Exception as e:
            self.handle_error(f"更新进度时发生错误: {str(e)}")

    def update_speed(self, avg_time):
        try:
            # 将毫秒转换为秒
            avg_time_sec = avg_time / 1000.0
            
            # 计算帧率（帧/秒）
            fps = 1.0 / avg_time_sec if avg_time_sec > 0 else 0
            
            # 根据帧率选择合适的显示单位
            if fps > 1.0:
                # 帧率大于1，显示为帧/秒
                self.speed_label.setText(f'处理速度: {fps:.2f} 帧/秒')
            else:
                # 帧率小于1，显示为秒/帧
                self.speed_label.setText(f'处理速度: {avg_time_sec:.2f} 秒/帧')
        except Exception as e:
            self.handle_error(f"更新速度时发生错误: {str(e)}")
    
    def processing_finished(self, result):
        try:
            self.progress_bar.hide()
            self.progress_label.hide()
            self.process_btn.setText('开始处理')
            self.process_btn.setEnabled(True)
            self.select_video_btn.setEnabled(True)
            self.select_dir_btn.setEnabled(True)
            self.remaining_time_label.setText('预计剩余时间: --:--')

            if result.startswith('错误'):
                self.status_label.setText(result)
            else:
                self.status_label.setText(f'处理完成！已保存到: {result}')
        except Exception as e:
            self.handle_error(f"处理完成回调时发生错误: {str(e)}")

    def handle_error(self, error_message):
        """全局错误处理函数"""
        print(f"[ERROR] {error_message}")
        self.status_label.setText(f"错误: {error_message}")
        
        # 如果启用了调试输出，记录到日志文件
        if hasattr(self, 'video_processor') and self.video_processor and self.video_processor.debug_output:
            self.video_processor._log_error(error_message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())