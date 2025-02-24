#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:16:50 2025

@author:
    GitHub@KirisameMarisa-DAZE (master.spark.kirisame.marisa.daze@gmail.com)
    All rights reserved. © 2021~2025
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QPoint, QSize, QTime
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QProgressBar, QSlider,
                             QHBoxLayout, QComboBox)

# 处理线程，增加了 output_format 参数支持
class VideoProcessor(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_dir, roi_rect, output_format="MP4"):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.roi_rect = roi_rect
        self.output_format = output_format
        self.cancel_flag = False
        self.fourcc_map = {
            'MP4': 'mp4v',
            'AVI': 'XVID',
            'MOV': 'mp4v',
            'MKV': 'X264'
        }

    def run(self):
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise ValueError("无法打开输入视频")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("视频帧数为0")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_cropped.{self.output_format.lower()}")

            fourcc = cv2.VideoWriter_fourcc(*self.fourcc_map[self.output_format])
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (self.roi_rect.width(), self.roi_rect.height()))

            for frame_num in range(total_frames):
                if self.cancel_flag:
                    break

                ret, frame = cap.read()
                if not ret:
                    print(f"读取帧失败，退出于第 {frame_num} 帧")
                    break

                x, y, w, h = self.roi_rect.x(), self.roi_rect.y(), self.roi_rect.width(), self.roi_rect.height()
                roi_frame = frame[y:y+h, x:x+w]
                out.write(roi_frame)

                self.progress_updated.emit(int((frame_num+1)/total_frames*100))

            cap.release()
            out.release()
            self.finished.emit(output_path if not self.cancel_flag else "")

        except Exception as e:
            print(f"Processing error: {str(e)}")
            self.finished.emit(f"错误: {str(e)}")

    def cancel(self):
        self.cancel_flag = True

# 视频显示区域，同时支持ROI框的创建、移动和缩放（带8个手柄）
class VideoLabel(QLabel):
    roi_selected = pyqtSignal(QRect)
    HANDLE_SIZE = 10  # 调整手柄尺寸

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("拖放视频文件到这里")
        self.setMinimumSize(640, 480)
        # ROI变量
        self.drag_mode = None   # "create"、"move" 或具体手柄名称，如"topleft", "top", "topright", etc.
        self.start_point = QPoint()
        self.current_roi = QRect()       # 绘制新ROI时使用
        self.permanent_roi = QRect()     # 最终ROI（原图坐标）
        self.original_size = QSize()
        self.display_rect = QRect()
        self.pen = QPen(Qt.red, 2, Qt.SolidLine)
        self.setMouseTracking(True)

    def set_video_frame(self, pixmap, original_size):
        self.original_size = original_size
        scaled_pix = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pix)
        pw = scaled_pix.width()
        ph = scaled_pix.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        self.display_rect = QRect(x, y, pw, ph)

    def get_scaled_roi(self):
        # 将原图坐标下的 ROI 转换到显示坐标
        if self.original_size.width() == 0 or self.original_size.height() == 0:
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
        # 返回当前ROI在显示坐标下的8个手柄位置
        scaled_roi = self.get_scaled_roi()
        if scaled_roi.isNull():
            return {}
        handles = {
            "topleft": scaled_roi.topLeft(),
            "top": QPoint(scaled_roi.center().x(), scaled_roi.top()),
            "topright": scaled_roi.topRight(),
            "right": QPoint(scaled_roi.right(), scaled_roi.center().y()),
            "bottomright": scaled_roi.bottomRight(),
            "bottom": QPoint(scaled_roi.center().x(), scaled_roi.bottom()),
            "bottomleft": scaled_roi.bottomLeft(),
            "left": QPoint(scaled_roi.left(), scaled_roi.center().y())
        }
        return handles

    def handle_at(self, pos):
        # 检测鼠标pos是否位于任一手柄上
        handles = self.get_handles()
        for key, center in handles.items():
            if self.get_handle_rect(center).contains(pos):
                return key
        # 如果鼠标在ROI边缘（扩大一些检测区域）则返回"move"
        scaled_roi = self.get_scaled_roi()
        if not scaled_roi.isNull() and scaled_roi.adjusted(-5, -5, 5, 5).contains(pos) and not scaled_roi.adjusted(5, 5, -5, -5).contains(pos):
            return "move"
        if not scaled_roi.isNull() and scaled_roi.contains(pos):
            return "move"
        return None

    def mousePressEvent(self, event):
        if not self.pixmap() or not self.display_rect.contains(event.pos()):
            return
        handle = self.handle_at(event.pos())
        if handle:
            self.drag_mode = handle
            self.start_point = event.pos()
            self.original_roi = QRect(self.permanent_roi)
        else:
            # 开始绘制新ROI
            self.drag_mode = "create"
            self.start_point = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(self.start_point, self.start_point)

    def mouseMoveEvent(self, event):
        if not self.pixmap():
            return
        # 更新光标形状
        handle = self.handle_at(event.pos())
        if handle:
            if handle == "move":
                self.setCursor(Qt.SizeAllCursor)
            elif handle in ("topleft", "bottomright"):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ("topright", "bottomleft"):
                self.setCursor(Qt.SizeBDiagCursor)
            elif handle in ("top", "bottom"):
                self.setCursor(Qt.SizeVerCursor)
            elif handle in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.drag_mode == "create":
            current = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(
                QPoint(min(self.start_point.x(), current.x()),
                       min(self.start_point.y(), current.y())),
                QPoint(max(self.start_point.x(), current.x()),
                       max(self.start_point.y(), current.y()))
            )
            self.update()
        elif self.drag_mode in ("move", "topleft", "top", "topright", "right", "bottomright", "bottom", "bottomleft", "left"):
            delta = event.pos() - self.start_point
            scale_x = self.original_size.width() / self.display_rect.width()
            scale_y = self.original_size.height() / self.display_rect.height()
            dx = int(delta.x() * scale_x)
            dy = int(delta.y() * scale_y)
            new_roi = QRect(self.original_roi)
            if self.drag_mode == "move":
                new_roi.translate(dx, dy)
            else:
                if "left" in self.drag_mode:
                    new_roi.setLeft(new_roi.left() + dx)
                if "right" in self.drag_mode:
                    new_roi.setRight(new_roi.right() + dx)
                if "top" in self.drag_mode:
                    new_roi.setTop(new_roi.top() + dy)
                if "bottom" in self.drag_mode:
                    new_roi.setBottom(new_roi.bottom() + dy)
            new_roi = new_roi.intersected(QRect(QPoint(0, 0), self.original_size))
            if new_roi.width() > 0 and new_roi.height() > 0:
                self.permanent_roi = new_roi
                self.roi_selected.emit(new_roi)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_mode == "create":
            if self.current_roi.width() > 0 and self.current_roi.height() > 0:
                scale_x = self.original_size.width() / self.display_rect.width()
                scale_y = self.original_size.height() / self.display_rect.height()
                roi = QRect(
                    int(self.current_roi.x() * scale_x),
                    int(self.current_roi.y() * scale_y),
                    int(self.current_roi.width() * scale_x),
                    int(self.current_roi.height() * scale_y)
                )
                self.permanent_roi = roi
                if not self.permanent_roi.isNull() and self.permanent_roi.width() > 0 and self.permanent_roi.height() > 0:
                    self.roi_selected.emit(self.permanent_roi)
        self.drag_mode = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(self.pen)
        if not self.permanent_roi.isNull():
            scaled_roi = self.get_scaled_roi()
            painter.drawRect(scaled_roi)
            handles = self.get_handles()
            for center in handles.values():
                rect = self.get_handle_rect(center)
                painter.fillRect(rect, Qt.white)

# 主窗口，整合视频预览、播放、时间显示、处理进度和输出目录/格式选择
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频ROI裁剪工具")
        self.setAcceptDrops(True)
        self.video_path = ""
        self.roi_rect = QRect()
        self.processing_thread = None
        self.cap = None  # 视频预览使用
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.total_duration = 0
        self.fps = 25
        # 初始化输出目录，防止后续引用时出错
        self.output_dir = os.path.expanduser("~/Desktop")
        self.init_ui()

    def init_ui(self):
        # 视频显示区域
        self.video_label = VideoLabel()
        self.video_label.roi_selected.connect(self.update_roi)

        # 控制组件
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_label = QLabel()  # 显示处理进度百分比
        self.progress_label.hide()

        self.status_label = QLabel("就绪")
        self.time_label = QLabel("00:00/00:00")  # 显示视频当前时间/总时长

        # 按钮组件
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.clicked.connect(self.select_video_file)

        # 输出目录与格式下拉菜单放在同一水平布局中
        self.select_dir_btn = QPushButton("选择输出目录")
        self.select_dir_btn.clicked.connect(self.select_output_dir)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP4", "AVI", "MOV", "MKV"])
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.select_dir_btn)
        dir_layout.addWidget(self.format_combo)

        self.process_btn = QPushButton("开始处理")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)

        # 播放控制
        self.play_pause_btn = QPushButton("播放")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.slider_released)

        # 播放进度条与时间标签在同一水平布局中
        play_layout = QHBoxLayout()
        play_layout.addWidget(self.slider)
        play_layout.addWidget(self.time_label)

        # 处理进度条与百分比标签在同一水平布局中
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        # 控制区总布局
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.select_video_btn)
        control_layout.addLayout(dir_layout)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addLayout(play_layout)
        control_layout.addLayout(progress_layout)
        control_layout.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.mp4','.avi','.mov','.mkv')):
                self.load_video(file_path)
                break

    def load_video(self, path):
        self.video_path = path
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_label.setText("无法打开视频文件")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 25
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = total_frames / self.fps
        interval = int(1000 / self.fps)
        self.timer.setInterval(interval)
        self.slider.setRange(0, total_frames - 1)
        self.update_time_label(0)

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("无法读取视频帧")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.set_video_frame(pixmap, QSize(w, h))

        # 重置视频到第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.play_pause_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"已加载视频：{os.path.basename(path)}")

    def update_time_label(self, current_frame):
        if not self.cap:
            return
        fps = self.fps if self.fps > 0 else 25
        current_sec = int(current_frame / fps)
        total_sec = int(self.total_duration)
        current_time = QTime(0, 0).addSecs(current_sec).toString("mm:ss")
        total_time = QTime(0, 0).addSecs(total_sec).toString("mm:ss")
        self.time_label.setText(f"{current_time}/{total_time}")

    def toggle_play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText("播放")
        else:
            self.timer.start()
            self.play_pause_btn.setText("暂停")

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.set_video_frame(pixmap, QSize(w, h))
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.setValue(current_frame)
                self.update_time_label(current_frame)
            else:
                self.timer.stop()

    def slider_released(self):
        if self.cap:
            frame_number = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.update_frame()

    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                   "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.load_video(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.status_label.setText(f"输出目录：{dir_path}")

    def update_roi(self, roi):
        self.roi_rect = roi
        self.status_label.setText(f"已选择ROI区域：X={roi.x()}, Y={roi.y()}, {roi.width()}x{roi.height()}")

    def start_processing(self):
        if not self.roi_rect.isValid():
            self.status_label.setText("请先选择有效的ROI区域")
            return

        # 停止预览并释放资源
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)

        # 创建处理线程，传入当前选择的视频格式
        self.processing_thread = VideoProcessor(self.video_path, self.output_dir, self.roi_rect, self.format_combo.currentText())
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)

        self.process_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.show()
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"{value}%")

    def processing_finished(self, result):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.process_btn.setEnabled(True)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.clear()
        self.video_label.setText("拖放视频文件到这里")
        self.video_label.permanent_roi = QRect()

        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.slider.setValue(0)

        if result.startswith("错误:"):
            self.status_label.setText(result)
        elif result:
            self.status_label.setText(f"处理完成！保存至：{result}")
            QTimer.singleShot(3000, self.reset_initial_state)
        else:
            self.status_label.setText("处理已取消")

    def reset_initial_state(self):
        self.status_label.setText("就绪")
        self.video_path = ""
        self.roi_rect = QRect()
        self.video_label.update()

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.processing_thread.wait()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
