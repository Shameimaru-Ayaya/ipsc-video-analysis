#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jan 13 12:16:50 2025

@author:
    GitHub@KirisameMarisa-DAZE (master.spark.kirisame.marisa.daze@gmail.com)
    All rights reserved. © 2021~2025
'''

import sys
import os
import cv2
import subprocess
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QPoint, QSize, QTime
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QProgressBar, QSlider,
                             QHBoxLayout, QComboBox)

def resource_path(relative_path):
    ''' Solve the problem of resource path after packing '''
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)

# FFmpeg path handling
FFMPEG_BIN = resource_path('ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')

# OpenCV plugin path fixes
if hasattr(sys, '_MEIPASS'):
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['QT_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'qt5_plugins')

class VideoProcessor(QThread):
    ''' Video Processing Thread '''
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_dir, roi_rect, output_format='MP4'):
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
                raise ValueError('Failed to open input video')

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError('Video frame rate is 0')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            output_path = os.path.join(self.output_dir, f'{base_name}_cropped.{self.output_format.lower()}')

            fourcc = cv2.VideoWriter_fourcc(*self.fourcc_map[self.output_format])
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (self.roi_rect.width(), self.roi_rect.height()))

            for frame_num in range(total_frames):
                if self.cancel_flag:
                    break

                ret, frame = cap.read()
                if not ret:
                    print(f'Failed to read frame, exited at frame {frame_num}')
                    break

                x, y, w, h = self.roi_rect.x(), self.roi_rect.y(), self.roi_rect.width(), self.roi_rect.height()
                roi_frame = frame[y:y+h, x:x+w]
                out.write(roi_frame)

                self.progress_updated.emit(int((frame_num+1)/total_frames*100))

            cap.release()
            out.release()
            self.finished.emit(output_path if not self.cancel_flag else '')

        except Exception as e:
            print(f'Processing error: {str(e)}')
            self.finished.emit(f'错误: {str(e)}')

    def cancel(self):
        self.cancel_flag = True

class VideoLabel(QLabel):
    ''' Video display area with simultaneous support for ROI box creation, movement and scaling (with 8 handles) '''
    clicked = pyqtSignal()
    roi_selected = pyqtSignal(QRect)
    HANDLE_SIZE = 10  # Adjust handle size

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText('Drag and drop video files here, or click here to select a file')
        self.setMinimumSize(640, 480)
        # ROI variables
        self.drag_mode = None   # ‘create’, ‘move’ or a specific handle name such as ‘topleft’, ‘top’, ‘topright’, etc.
        self.start_point = QPoint()
        self.current_roi = QRect()       # Used when drawing new ROIs
        self.permanent_roi = QRect()     # Final ROI (original map coordinates)
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
        ''' Convert ROI in original map coordinates to display coordinates '''
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
        ''' Returns the current ROI's 8 handle positions in display coordinates '''
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
        ''' Detect if the mouse pos is on any of the handles '''
        handles = self.get_handles()
        for key, center in handles.items():
            if self.get_handle_rect(center).contains(pos):
                return key
        # Return ‘move’ if the mouse is at the edge of the ROI (expanding the detection area a bit)
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
            # Start drawing new ROIs
            self.drag_mode = 'create'
            self.start_point = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(self.start_point, self.start_point)

    def mouseMoveEvent(self, event):
        if not self.pixmap():
            return
        # Update cursor shape
        handle = self.handle_at(event.pos())
        if handle:
            if handle == 'move':
                self.setCursor(Qt.SizeAllCursor)
            elif handle in ('topleft', 'bottomright'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ('topright', 'bottomleft'):
                self.setCursor(Qt.SizeBDiagCursor)
            elif handle in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            elif handle in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.drag_mode == 'create':
            current = event.pos() - self.display_rect.topLeft()
            self.current_roi = QRect(
                QPoint(min(self.start_point.x(), current.x()),
                       min(self.start_point.y(), current.y())),
                QPoint(max(self.start_point.x(), current.x()),
                       max(self.start_point.y(), current.y()))
            )
            self.update()
        elif self.drag_mode in ('move', 'topleft', 'top', 'topright', 'right', 'bottomright', 'bottom', 'bottomleft', 'left'):
            delta = event.pos() - self.start_point
            scale_x = self.original_size.width() / self.display_rect.width()
            scale_y = self.original_size.height() / self.display_rect.height()
            dx = int(delta.x() * scale_x)
            dy = int(delta.y() * scale_y)
            new_roi = QRect(self.original_roi)
            if self.drag_mode == 'move':
                new_roi.translate(dx, dy)
            else:
                if 'left' in self.drag_mode:
                    new_roi.setLeft(new_roi.left() + dx)
                if 'right' in self.drag_mode:
                    new_roi.setRight(new_roi.right() + dx)
                if 'top' in self.drag_mode:
                    new_roi.setTop(new_roi.top() + dy)
                if 'bottom' in self.drag_mode:
                    new_roi.setBottom(new_roi.bottom() + dy)
            new_roi = new_roi.intersected(QRect(QPoint(0, 0), self.original_size))
            if new_roi.width() > 0 and new_roi.height() > 0:
                self.permanent_roi = new_roi
                self.roi_selected.emit(new_roi)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_mode == 'create':
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

class MainWindow(QMainWindow):
    ''' Main window with integrated video preview, playback, time display, processing progress and output directory/format selection '''
    def __init__(self):
        super().__init__()
        self.last_output_path = ''
        self.setWindowTitle('Video ROI Cropping Tool')
        self.setAcceptDrops(True)
        self.video_path = ''
        self.roi_rect = QRect()
        self.processing_thread = None
        self.cap = None  # Video preview use
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.total_duration = 0
        self.fps = 25
        self.output_dir = os.path.expanduser('~/Desktop') # Initialise the output directory to prevent errors in subsequent references
        self.init_ui()

    def init_ui(self):
        ''' Video display area '''
        self.video_label = VideoLabel()
        self.video_label.roi_selected.connect(self.update_roi)
        self.video_label.clicked.connect(self.select_video_file)

        # Control components
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_label = QLabel()  # Percentage of processing progress displayed
        self.progress_label.hide()

        self.status_label = QLabel('Ready')
        self.time_label = QLabel('00:00/00:00')

        # Button Components
        self.select_video_btn = QPushButton('Reselect the video')
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.select_video_btn.setEnabled(False)  # Initially unavailable

        # Reselect, output catalogue and format drop-down menus in the same horizontal layout
        self.select_dir_btn = QPushButton('Select output directory')
        self.select_dir_btn.clicked.connect(self.select_output_dir)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['MP4', 'AVI', 'MOV', 'MKV'])

        # Create sub-layouts containing tabs and formatting drop-down menus
        format_layout = QHBoxLayout()
        format_label = QLabel('Output format:')
        format_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Right-aligned
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.setSpacing(5)  # Set tab and dropdown box spacing

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.select_video_btn)
        dir_layout.addWidget(self.select_dir_btn)
        dir_layout.addLayout(format_layout)  # Add a sub-layout

        self.process_btn = QPushButton('Start processing')
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)

        self.open_output_btn = QPushButton('Open output directory')
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self.open_output_directory)

        # Start processing, open catalogue in the same horizontal layout
        process_layout = QHBoxLayout()
        process_layout.addWidget(self.process_btn)
        process_layout.addWidget(self.open_output_btn)

        # Playback control
        self.play_pause_btn = QPushButton('Play')
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.slider_released)

        # Play buttons, progress bars & timestamps in the same horizontal layout
        play_layout = QHBoxLayout()
        play_layout.addWidget(self.play_pause_btn)
        play_layout.addWidget(self.slider)
        play_layout.addWidget(self.time_label)

        # Handle progress bars and percentage labels in the same horizontal layout
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        # General layout of the control area
        control_layout = QVBoxLayout()
        control_layout.addLayout(play_layout)       # Playback (play button, timeline, timestamps)
        control_layout.addLayout(dir_layout)        # Path (input path, output directory, file format)
        control_layout.addLayout(process_layout)    # Processing (start processing, open catalogue)
        control_layout.addLayout(progress_layout)   # Process (progress bar, percentage of progress)
        control_layout.addWidget(self.status_label) # Status bar

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
            self.status_label.setText('Failure to open the video file')
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
            self.status_label.setText('Failed to read video frames')
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.set_video_frame(pixmap, QSize(w, h))

        # Reset the video to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.play_pause_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.select_video_btn.setEnabled(True)
        self.status_label.setText(f'Loaded video: {os.path.basename(path)}')

    def update_time_label(self, current_frame):
        if not self.cap:
            return
        fps = self.fps if self.fps > 0 else 25
        current_sec = int(current_frame / fps)
        total_sec = int(self.total_duration)
        current_time = QTime(0, 0).addSecs(current_sec).toString('mm:ss')
        total_time = QTime(0, 0).addSecs(total_sec).toString('mm:ss')
        self.time_label.setText(f'{current_time}/{total_time}')

    def toggle_play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText('Play')
        else:
            self.timer.start()
            self.play_pause_btn.setText('Pause')

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
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select video file', '',
                                                   '视频文件 (*.mp4 *.avi *.mov *.mkv)')
        if file_path:
            self.load_video(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select output directory', self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.status_label.setText(f'Output directory: {dir_path}')

    def update_roi(self, roi):
        self.roi_rect = roi
        self.status_label.setText(f'The ROI area has been selected as: X={roi.x()}, Y={roi.y()}, {roi.width()}x{roi.height()}')

    def start_processing(self):
        if not self.roi_rect.isValid():
            self.status_label.setText('Please select a valid ROI area first')
            return

        # Stop previewing and release resources
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)
        self.select_video_btn.setEnabled(False)
        self.select_dir_btn.setEnabled(False)
        self.format_combo.setEnabled(False)
        self.open_output_btn.setEnabled(False)

        # Create a processing thread, passing in the currently selected video format
        self.processing_thread = VideoProcessor(self.video_path, self.output_dir, self.roi_rect, self.format_combo.currentText())
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)

        self.process_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.show()
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f'{value}%')

    def processing_finished(self, result):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.process_btn.setEnabled(True)

        # Update output path and button status
        if result and not result.startswith('Error: '):
            self.last_output_path = result
            self.open_output_btn.setEnabled(True)
        else:
            self.open_output_btn.setEnabled(False)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.clear()
        self.video_label.setText('Drag and drop video files here, or click here to select a file')
        self.video_label.permanent_roi = QRect()

        self.select_dir_btn.setEnabled(True)
        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.select_video_btn.setEnabled(False)
        self.slider.setValue(0)

        if result.startswith('Error: '):
            self.status_label.setText(result)
        elif result:
            self.status_label.setText(f'Processing complete! Saved to: {result}')
            QTimer.singleShot(3000, self.reset_initial_state)
        else:
            self.status_label.setText('Processing cancelled')

    def reset_initial_state(self):
        self.status_label.setText('Ready')
        self.video_path = ''
        self.roi_rect = QRect()
        self.video_label.update()
        self.select_dir_btn.setEnabled(True)
        self.select_video_btn.setEnabled(False)

    def open_output_directory(self):
        if not self.last_output_path:
            return

        # Get file paths and directory paths
        file_path = os.path.normpath(self.last_output_path)
        dir_path = os.path.dirname(file_path)

        # Handle different operating systems
        if sys.platform == 'win32':
            # Windows: open directory and select files
            if os.path.exists(file_path):
                os.startfile(dir_path)
                # Additional processing required to select files using explorer
                subprocess.run(f'explorer /select,"{file_path}"', shell=True)
            else:
                os.startfile(dir_path)
        elif sys.platform == 'darwin':
            # macOS: open directory and select files
            if os.path.exists(file_path):
                subprocess.run(['open', '-R', file_path])
            else:
                subprocess.run(['open', dir_path])
        else:
            # Linux: Use File Manager to open
            if os.path.exists(file_path):
                subprocess.run(['xdg-open', dir_path])
            else:
                subprocess.run(['xdg-open', dir_path])

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.processing_thread.wait()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())