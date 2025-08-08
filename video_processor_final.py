#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:16:50 2025
Updated on Thu Aug 08 12:00:00 2025

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
                           QSpinBox, QGridLayout, QDialogButtonBox, QActionGroup)

# --- 新增：多语言翻译字典 ---
TRANSLATIONS = {
    'zh': {
        # 窗口标题
        'window_title': '视频处理工具',
        'roi_tuning_title': 'ROI 微调',
        'about_title': '关于',
        'lang_select_title': '选择语言',

        # 主界面文本
        'video_label_default': '拖放视频文件到这里，或点击选择文件',
        'select_output_dir_btn': '选择输出目录',
        'reselect_video_btn': '重新选择视频',
        'overflow_combo_no': '目标不溢出边界',
        'overflow_combo_yes': '目标溢出边界',
        'debug_output_combo_no': '不输出调试信息',
        'debug_output_combo_yes': '输出调试信息',
        'intermediate_frames_combo_no': '不输出中间帧',
        'intermediate_frames_combo_yes': '输出中间帧',
        'process_btn_start': '开始处理',
        'process_btn_stop': '终止处理',
        'speed_label_template': '处理速度: {speed_text}',
        'remaining_time_label_template': '预计剩余时间: {time_text}',
        'status_label_ready': '就绪',
        'status_label_select_roi': '请选择ROI区域',
        'status_label_roi_selected': 'ROI已选择，可以开始处理',
        'status_label_output_dir': '输出目录: {dir_path}',
        'status_label_processing': '正在处理...',
        'status_label_stopped': '处理已终止',
        'status_label_finished': '处理完成！已保存到: {result}',
        'status_label_error_roi': '错误：开始处理前必须选择一个有效的ROI区域！',
        'status_label_error_generic': '错误: {error_message}',
        'progress_label_template': '处理进度: {value}%',
        'speed_calculating': '计算中...',
        'time_calculating': '计算中...',
        'time_placeholder': '--:--',
        'fps_unit': '{fps:.2f} 帧/秒',
        'spf_unit': '{spf:.2f} 秒/帧',

        # 菜单
        'menu_file': '文件(&F)',
        'menu_file_open': '打开视频(&O)',
        'menu_file_output_dir': '选择输出目录(&D)',
        'menu_file_exit': '退出(&X)',
        'menu_edit': '编辑(&E)',
        'menu_edit_roi': 'ROI 微调(&R)',
        'menu_help': '帮助(&H)',
        'menu_help_about': '关于(&A)',
        'menu_language': '语言(&L)',

        # 对话框
        'about_dialog_text': '视频处理工具\n版本 2.3.0\n作者: Github@Shameimaru-Ayaya',
        'select_video_dialog_title': '选择视频文件',
        'video_files_filter': '视频文件 (*.mp4 *.avi *.mov *.mkv)',
        'select_output_dir_dialog_title': '选择输出目录',
        'roi_dialog_x': 'X 坐标:',
        'roi_dialog_y': 'Y 坐标:',
        'roi_dialog_w': '宽度 (W):',
        'roi_dialog_h': '高度 (H):',
    },
    'en': {
        # Window Titles
        'window_title': 'Video Processing Tool',
        'roi_tuning_title': 'ROI Fine-tuning',
        'about_title': 'About',
        'lang_select_title': 'Select Language',

        # Main UI Text
        'video_label_default': 'Drag & Drop video file here, or click to select file',
        'select_output_dir_btn': 'Select Output Directory',
        'reselect_video_btn': 'Reselect Video',
        'overflow_combo_no': 'Target within bounds',
        'overflow_combo_yes': 'Target overflows bounds',
        'debug_output_combo_no': 'Disable debug output',
        'debug_output_combo_yes': 'Enable debug output',
        'intermediate_frames_combo_no': 'No intermediate frames',
        'intermediate_frames_combo_yes': 'Output intermediate frames',
        'process_btn_start': 'Start Processing',
        'process_btn_stop': 'Stop Processing',
        'speed_label_template': 'Processing Speed: {speed_text}',
        'remaining_time_label_template': 'Est. time remaining: {time_text}',
        'status_label_ready': 'Ready',
        'status_label_select_roi': 'Please select an ROI',
        'status_label_roi_selected': 'ROI selected, ready to process',
        'status_label_output_dir': 'Output directory: {dir_path}',
        'status_label_processing': 'Processing...',
        'status_label_stopped': 'Processing stopped',
        'status_label_finished': 'Processing finished! Saved to: {result}',
        'status_label_error_roi': 'Error: A valid ROI must be selected before starting!',
        'status_label_error_generic': 'Error: {error_message}',
        'progress_label_template': 'Progress: {value}%',
        'speed_calculating': 'Calculating...',
        'time_calculating': 'Calculating...',
        'time_placeholder': '--:--',
        'fps_unit': '{fps:.2f} fps',
        'spf_unit': '{spf:.2f} s/frame',

        # Menus
        'menu_file': '&File',
        'menu_file_open': '&Open Video',
        'menu_file_output_dir': 'Select &Output Directory',
        'menu_file_exit': 'E&xit',
        'menu_edit': '&Edit',
        'menu_edit_roi': '&ROI Fine-tuning',
        'menu_help': '&Help',
        'menu_help_about': '&About',
        'menu_language': '&Language',

        # Dialogs
        'about_dialog_text': 'Video Processing Tool\nVersion 2.3.0\nAuthor: Github@Shameimaru-Ayaya',
        'select_video_dialog_title': 'Select Video File',
        'video_files_filter': 'Video Files (*.mp4 *.avi *.mov *.mkv)',
        'select_output_dir_dialog_title': 'Select Output Directory',
        'roi_dialog_x': 'X coordinate:',
        'roi_dialog_y': 'Y coordinate:',
        'roi_dialog_w': 'Width (W):',
        'roi_dialog_h': 'Height (H):',
    },
    'ja': {
        # Window Titles
        'window_title': 'ビデオ処理ツール',
        'roi_tuning_title': 'ROI 微調整',
        'about_title': 'バージョン情報',
        'lang_select_title': '言語を選択',

        # Main UI Text
        'video_label_default': 'ここにビデオファイルをドラッグ＆ドロップするか、クリックしてファイルを選択',
        'select_output_dir_btn': '出力先を選択',
        'reselect_video_btn': 'ビデオを再選択',
        'overflow_combo_no': 'ターゲットは境界内',
        'overflow_combo_yes': 'ターゲットは境界外',
        'debug_output_combo_no': 'デバッグ情報を出力しない',
        'debug_output_combo_yes': 'デバッグ情報を出力する',
        'intermediate_frames_combo_no': '中間フレームを保存しない',
        'intermediate_frames_combo_yes': '中間フレームを保存する',
        'process_btn_start': '処理開始',
        'process_btn_stop': '処理停止',
        'speed_label_template': '処理速度: {speed_text}',
        'remaining_time_label_template': '予想残り時間: {time_text}',
        'status_label_ready': '準備完了',
        'status_label_select_roi': 'ROI領域を選択してください',
        'status_label_roi_selected': 'ROI選択済み、処理を開始できます',
        'status_label_output_dir': '出力先: {dir_path}',
        'status_label_processing': '処理中...',
        'status_label_stopped': '処理が停止しました',
        'status_label_finished': '処理完了！保存先: {result}',
        'status_label_error_roi': 'エラー：処理を開始する前に、有効なROI領域を選択する必要があります！',
        'status_label_error_generic': 'エラー: {error_message}',
        'progress_label_template': '進捗: {value}%',
        'speed_calculating': '計算中...',
        'time_calculating': '計算中...',
        'time_placeholder': '--:--',
        'fps_unit': '{fps:.2f} フレーム/秒',
        'spf_unit': '{spf:.2f} 秒/フレーム',

        # Menus
        'menu_file': 'ファイル(&F)',
        'menu_file_open': 'ビデオを開く(&O)',
        'menu_file_output_dir': '出力先を選択(&D)',
        'menu_file_exit': '終了(&X)',
        'menu_edit': '編集(&E)',
        'menu_edit_roi': 'ROI 微調整(&R)',
        'menu_help': 'ヘルプ(&H)',
        'menu_help_about': 'バージョン情報(&A)',
        'menu_language': '言語(&L)',

        # Dialogs
        'about_dialog_text': 'ビデオ処理ツール\nバージョン 2.3.0\n作者: Github@Shameimaru-Ayaya',
        'select_video_dialog_title': 'ビデオファイルを選択',
        'video_files_filter': 'ビデオファイル (*.mp4 *.avi *.mov *.mkv)',
        'select_output_dir_dialog_title': '出力先を選択',
        'roi_dialog_x': 'X 座標:',
        'roi_dialog_y': 'Y 座標:',
        'roi_dialog_w': '幅 (W):',
        'roi_dialog_h': '高さ (H):',
    }
}


# --- 新增：启动时语言选择对话框 ---
class LanguageSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Language / 选择语言 / 言語を選択')
        self.selected_language = None
        
        layout = QVBoxLayout(self)
        
        label = QLabel('Please select your preferred language:')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        buttons_layout = QHBoxLayout()
        
        en_btn = QPushButton('English')
        en_btn.clicked.connect(lambda: self.select_lang('en'))
        
        zh_btn = QPushButton('简体中文 (Simplified Chinese)')

        zh_btn.clicked.connect(lambda: self.select_lang('zh'))
        
        ja_btn = QPushButton('日本語 (Japanese)')
        ja_btn.clicked.connect(lambda: self.select_lang('ja'))
        
        buttons_layout.addWidget(en_btn)
        buttons_layout.addWidget(zh_btn)
        buttons_layout.addWidget(ja_btn)
        
        layout.addLayout(buttons_layout)
        self.setFixedSize(self.sizeHint())

    def select_lang(self, lang):
        self.selected_language = lang
        self.accept()

    @staticmethod
    def get_language(parent=None):
        dialog = LanguageSelectionDialog(parent)
        dialog.exec_()
        return dialog.selected_language


# --- 修改：ROI微调对话框以支持多语言 ---
class RoiTuningDialog(QDialog):
    def __init__(self, initial_roi, max_width, max_height, parent=None, translations=None):
        super().__init__(parent)
        t = translations if translations else {}
        self.setWindowTitle(t.get('title', 'ROI Fine-tuning'))
        
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
        layout.addWidget(QLabel(t.get('x', 'X coordinate:')), 0, 0)
        layout.addWidget(self.x_spinbox, 0, 1)
        layout.addWidget(QLabel(t.get('y', 'Y coordinate:')), 1, 0)
        layout.addWidget(self.y_spinbox, 1, 1)
        layout.addWidget(QLabel(t.get('w', 'Width (W):')), 2, 0)
        layout.addWidget(self.w_spinbox, 2, 1)
        layout.addWidget(QLabel(t.get('h', 'Height (H):')), 3, 0)
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
        # 文本将在MainWindow的retranslate_ui中设置
        self.setMinimumSize(640, 480)
        self.drag_mode = None
        self.start_point = QPoint()
        self.current_roi = QRect()
        self.permanent_roi = QRect()
        self.original_size = QSize()
        self.display_rect = QRect()
        self.pen = QPen(Qt.red, 2, Qt.SolidLine)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    def set_video_frame(self, pixmap, original_size):
        self.original_size = original_size
        scaled_pix = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pix)
        pw = scaled_pix.width()
        ph = scaled_pix.height()
        x = (self.width() - pw) // 2
        y = (self.height() - ph) // 2
        self.display_rect = QRect(x, y, pw, ph)
        self.update()

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
        self.base_output_dir = output_dir
        self.roi_x, self.roi_y, self.roi_width, self.roi_height = roi
        self.threshold = threshold
        self.variance_threshold = variance_threshold
        self.cap = cv2.VideoCapture(video_path)
        self.displacements = []
        self.processing_speed = 96178
        self.is_overflow = is_overflow
        self.debug_output = debug_output
        self.output_intermediate_frames = output_intermediate_frames

        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = os.path.join(self.base_output_dir, video_filename)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_video_path = os.path.join(self.output_dir, "processed_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                          (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.processing_times = []
        
        self._log_command(f"INIT {self.video_path}", f"ROI: {roi}, 阈值: {threshold}, 溢出模式: {is_overflow}")

        if self.debug_output:
            self.debug_dir = os.path.join(self.output_dir, "debug_output")
            self.debug_frames_dir = os.path.join(self.debug_dir, "frames")
            self.debug_videos_dir = os.path.join(self.debug_dir, "videos")
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(self.debug_frames_dir, exist_ok=True)
            os.makedirs(self.debug_videos_dir, exist_ok=True)
            
            log_path = os.path.join(self.debug_dir, "process.log")
            cmd_path = os.path.join(self.debug_dir, "commands.log")
            
            if os.path.exists(log_path):
                self.log_file = open(log_path, "a")
                self._log_info("\n" + "="*50 + "\n")
                self._log_info("继续记录日志")
            else:
                self.log_file = open(log_path, "w")
                
            if os.path.exists(cmd_path):
                self.cmd_file = open(cmd_path, "a")
                self.cmd_file.write("\n" + "="*50 + "\n")
            else:
                self.cmd_file = open(cmd_path, "w")

            self._log_command("INIT", f"视频: {self.video_path}, ROI: {roi}, 阈值: {threshold}, 溢出模式: {is_overflow}")

            self._log_info("系统信息:")
            self._log_info(f"Python版本: {sys.version}")
            self._log_info(f"OpenCV版本: {cv2.__version__}")
            self._log_info(f"NumPy版本: {np.__version__}")
            self._log_info(f"Pandas版本: {pd.__version__}")
            
            self._log_info("视频信息:")
            self._log_info(f"路径: {self.video_path}")
            self._log_info(f"帧率: {self.fps}")
            self._log_info(f"总帧数: {self.total_frames}")
            self._log_info(f"分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            self._log_info("处理参数:")
            self._log_info(f"ROI: ({self.roi_x}, {self.roi_y}, {self.roi_width}, {self.roi_height})")
            self._log_info(f"频率变化阈值: {self.threshold}")
            self._log_info(f"方差阈值: {self.variance_threshold}")
            self._log_info(f"溢出模式: {self.is_overflow}")

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        start_time = time.time()
        
        debug_frames = [] if self.debug_output else None
        
        self._log_info(f"开始处理第 {frame_index} 帧")
            
        if frame is None or frame.size == 0:
            self._log_error(f"第 {frame_index} 帧为空或无效")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        if self.debug_output:
            debug_frames.append(("original", frame.copy()))
            self._log_debug(f"帧尺寸: {frame.shape}")
            
        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.debug_output:
                debug_frames.append(("HSV", hsv_frame))
                self._log_debug("已转换为HSV空间")
            
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask = cv2.inRange(hsv_frame, lower_black, upper_black)
            if self.debug_output:
                debug_frames.append(("mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
                self._log_debug(f"掩膜统计: 黑色像素数量: {np.sum(mask > 0)}")
            
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if self.debug_output:
                debug_frames.append(("opening", cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            if self.debug_output:
                debug_frames.append(("closing", cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)))
            
            median = cv2.medianBlur(closing, 5)
            if self.debug_output:
                debug_frames.append(("median", cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)))
            
            if not self.is_overflow:
                edges = cv2.Canny(median, 50, 150)
                roi_source = edges
                if self.debug_output:
                    debug_frames.append(("edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
            else:
                roi_source = median
                
            display_frame = frame.copy()
            
            try:
                expanded_x1 = max(0, self.roi_x - 1)
                expanded_y1 = max(0, self.roi_y - 1)
                expanded_x2 = min(frame.shape[1], self.roi_x + self.roi_width + 1)
                expanded_y2 = min(frame.shape[0], self.roi_y + self.roi_height + 1)
        
                roi_expanded = roi_source[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
                
                if roi_expanded.size == 0:
                    self._log_warning(f"警告：扩展ROI区域为空，帧索引：{frame_index}")
                    return frame
        
                mask = np.zeros_like(roi_expanded)
                original_in_expanded_x = self.roi_x - expanded_x1
                original_in_expanded_y = self.roi_y - expanded_y1
                mask[original_in_expanded_y:original_in_expanded_y+self.roi_height,
                     original_in_expanded_x:original_in_expanded_x+self.roi_width] = 255
        
                roi_masked = cv2.bitwise_and(roi_expanded, roi_expanded, mask=mask.astype(np.uint8))
                
                padded_roi = cv2.copyMakeBorder(roi_masked, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                
            except Exception as e:
                self._log_warning(f"ROI提取错误：{str(e)}，帧索引：{frame_index}")
                return frame
                
            contours, _ = cv2.findContours(padded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            adjusted_contours = []
            for contour in contours:
                adjusted_contour = contour.copy()
                adjusted_contour[:,:,0] += (expanded_x1 - 1)
                adjusted_contour[:,:,1] += (expanded_y1 - 1)
                adjusted_contours.append(adjusted_contour)
            
            try:
                if adjusted_contours:
                    contour_areas = [cv2.contourArea(cnt) for cnt in adjusted_contours]
                    valid_contours = [cnt for i, cnt in enumerate(adjusted_contours) if contour_areas[i] > 100]
                    
                    if valid_contours:
                        max_contour = max(valid_contours, key=cv2.contourArea)
                        
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
            
            if self.debug_output and debug_frames:
                self._save_debug_frame(debug_frames, frame_index)
                
                contour_image = np.zeros_like(frame)
                if adjusted_contours:
                    cv2.drawContours(contour_image, adjusted_contours, -1, (0, 255, 0), 2)
                debug_frames.append(("contours", contour_image))
                
                debug_frames.append(("result", display_frame.copy()))
                
                self._save_debug_frame(debug_frames, frame_index)
            
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
        
        if self.debug_output and debug_frames:
            self._save_debug_frame(debug_frames, frame_index)
        
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
            name='Smoothed Displacement', # Plotly text is not easily translated, keep as English
            line=dict(color='blue', width=1)
        ))

        for i, change in enumerate(changes):
            change_time = peaks[change] / self.fps
            fig.add_trace(go.Scatter(
                x=[change_time],
                y=[smoothed_displacements[peaks[change]]],
                mode='markers',
                name=f'Frequency Change Point {i+1}',
                marker=dict(color='red', size=10)
            ))

        fig.update_layout(
            title='Displacement and Frequency Change Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Displacement',
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
                if self.isInterruptionRequested():
                    self._log_info("处理被用户终止")
                    self._log_command("STOP_PROCESSING", "用户手动终止")
                    self.finished.emit("处理被用户终止") # Note: This message will be handled by MainWindow but not translated here.
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
                    self.video_writer.write(frame)
                
                progress = int((frame_index + 1) / self.total_frames * 100)
                self.progress_updated.emit(progress)
                
                frame_index += 1
                
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
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[INFO] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()
            
    def _log_command(self, command, description=""):
        if not self.debug_output or not hasattr(self, 'cmd_file'):
            return
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {command}"
        if description:
            log_entry += f" # {description}"
        
        self.cmd_file.write(log_entry + "\n")
        self.cmd_file.flush()

    def _log_warning(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[WARNING] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()

    def _log_error(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[ERROR] {timestamp}: {message}"
        print(log_message)
        if self.debug_output and hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()

    def _log_debug(self, message):
        if self.debug_output:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_message = f"[DEBUG] {timestamp}: {message}"
            print(log_message)
            if hasattr(self, 'log_file'):
                self.log_file.write(log_message + "\n")
                self.log_file.flush()

    def _save_debug_frame(self, debug_frames, frame_index):
        if not self.debug_output or not debug_frames:
            return
            
        if not self.output_intermediate_frames:
            return
            
        try:
            frame_dir = os.path.join(self.debug_frames_dir, f"frame_{frame_index:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            for name, img in debug_frames:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                output_path = os.path.join(frame_dir, f"{name}.png")
                cv2.imwrite(output_path, img)
                
            self._log_warning(f"已保存帧 {frame_index} 的调试图像到 {frame_dir}")
        except Exception as e:
            self._log_warning(f"保存调试帧时出错: {str(e)}")

    def __del__(self):
        if hasattr(self, 'debug_output') and self.debug_output:
            if hasattr(self, 'log_file'):
                self.log_file.close()
            if hasattr(self, 'cmd_file'):
                self.cmd_file.close()

class MainWindow(QMainWindow):
    def __init__(self, language='zh'):
        super().__init__()
        self.lang = language
        self.video_path = ''
        self.output_dir = os.path.expanduser('~/Desktop')
        self.video_processor = None
        self.permanent_roi = None
        
        self.init_ui()
        self.retranslate_ui() # 设置所有UI文本的初始语言

    def tr(self, key, **kwargs):
        """获取并格式化翻译文本"""
        text = TRANSLATIONS[self.lang].get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    def init_ui(self):
        # 仅创建控件，文本在 retranslate_ui 中设置
        self.video_label = VideoLabel()
        self.video_label.roi_selected.connect(self.update_roi)
        self.video_label.clicked.connect(self.select_video_file)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_label = QLabel()
        self.progress_label.hide()

        self.status_label = QLabel()
        self.time_label = QLabel('00:00/00:00')

        file_buttons_layout = QHBoxLayout()
        self.select_dir_btn = QPushButton()
        self.select_dir_btn.clicked.connect(self.select_output_dir)
        self.select_video_btn = QPushButton()
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.select_video_btn.setEnabled(False)
        file_buttons_layout.addWidget(self.select_dir_btn)
        file_buttons_layout.addStretch()
        file_buttons_layout.addWidget(self.select_video_btn)

        options_layout = QHBoxLayout()
        self.overflow_combo = QComboBox()
        self.overflow_combo.setFixedWidth(150)
        self.overflow_combo.setStyleSheet("QComboBox { text-align: center; }")
        
        self.debug_output_combo = QComboBox()
        self.debug_output_combo.setFixedWidth(150)
        self.debug_output_combo.setStyleSheet("QComboBox { text-align: center; }")
        self.debug_output_combo.currentIndexChanged.connect(self.update_intermediate_frames_combo_state)
        
        self.intermediate_frames_combo = QComboBox()
        self.intermediate_frames_combo.setFixedWidth(150)
        self.intermediate_frames_combo.setStyleSheet("QComboBox { text-align: center; }")
        self.intermediate_frames_combo.setEnabled(False)
        
        options_layout.addWidget(self.overflow_combo)
        options_layout.addStretch()
        options_layout.addWidget(self.debug_output_combo)
        options_layout.addWidget(self.intermediate_frames_combo)
        
        speed_layout = QHBoxLayout()
        self.speed_label = QLabel()
        self.remaining_time_label = QLabel()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addStretch()
        speed_layout.addWidget(self.remaining_time_label)
        
        self.process_btn = QPushButton()
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.toggle_processing)
        
        control_layout = QVBoxLayout()
        control_layout.addLayout(file_buttons_layout)
        control_layout.addLayout(options_layout)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.progress_label)
        control_layout.addLayout(speed_layout)
        control_layout.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # 创建菜单栏和动作，文本在 retranslate_ui 中设置
        self._create_menu_bar_actions()

    def _create_menu_bar_actions(self):
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu('')
        self.open_action = QAction('', self)
        self.open_action.triggered.connect(self.select_video_file)
        self.output_dir_action = QAction('', self)
        self.output_dir_action.triggered.connect(self.select_output_dir)
        self.exit_action = QAction('', self)
        self.exit_action.triggered.connect(self.close)

        self.edit_menu = self.menu_bar.addMenu('')
        self.fine_tune_roi_action = QAction('', self)
        self.fine_tune_roi_action.triggered.connect(self.open_roi_tuning_dialog)
        self.fine_tune_roi_action.setEnabled(False)

        self.help_menu = self.menu_bar.addMenu('')
        self.about_action = QAction('', self)
        self.about_action.triggered.connect(self.show_about_dialog)
        
        # 语言菜单
        self.lang_menu = self.menu_bar.addMenu('')
        self.lang_action_group = QActionGroup(self)
        self.lang_action_group.setExclusive(True)

        self.en_action = QAction('English', self, checkable=True)
        self.en_action.triggered.connect(lambda: self.change_language('en'))
        self.zh_action = QAction('简体中文', self, checkable=True)
        self.zh_action.triggered.connect(lambda: self.change_language('zh'))
        self.ja_action = QAction('日本語', self, checkable=True)
        self.ja_action.triggered.connect(lambda: self.change_language('ja'))

    def retranslate_ui(self):
        """更新所有UI元素的文本"""
        self.setWindowTitle(self.tr('window_title'))

        # 标签
        if not self.video_path:
             self.video_label.setText(self.tr('video_label_default'))
        speed_text = self.tr('time_placeholder')
        self.speed_label.setText(self.tr('speed_label_template', speed_text=speed_text))
        self.remaining_time_label.setText(self.tr('remaining_time_label_template', time_text=self.tr('time_placeholder')))
        self.status_label.setText(self.tr('status_label_ready'))
        self.progress_label.setText(self.tr('progress_label_template', value=0))

        # 按钮
        self.select_dir_btn.setText(self.tr('select_output_dir_btn'))
        self.select_video_btn.setText(self.tr('reselect_video_btn'))
        if self.video_processor and self.video_processor.isRunning():
            self.process_btn.setText(self.tr('process_btn_stop'))
        else:
            self.process_btn.setText(self.tr('process_btn_start'))
            
        # ComboBoxes - 保存并恢复当前索引
        for combo, keys in [
            (self.overflow_combo, ['overflow_combo_no', 'overflow_combo_yes']),
            (self.debug_output_combo, ['debug_output_combo_no', 'debug_output_combo_yes']),
            (self.intermediate_frames_combo, ['intermediate_frames_combo_no', 'intermediate_frames_combo_yes'])
        ]:
            current_index = combo.currentIndex()
            combo.clear()
            combo.addItems([self.tr(key) for key in keys])
            combo.setCurrentIndex(current_index)
            
        # 菜单栏
        self.file_menu.setTitle(self.tr('menu_file'))
        self.file_menu.clear()
        self.open_action.setText(self.tr('menu_file_open'))
        self.output_dir_action.setText(self.tr('menu_file_output_dir'))
        self.exit_action.setText(self.tr('menu_file_exit'))
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.output_dir_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu.setTitle(self.tr('menu_edit'))
        self.edit_menu.clear()
        self.fine_tune_roi_action.setText(self.tr('menu_edit_roi'))
        self.edit_menu.addAction(self.fine_tune_roi_action)

        self.help_menu.setTitle(self.tr('menu_help'))
        self.help_menu.clear()
        self.about_action.setText(self.tr('menu_help_about'))
        self.help_menu.addAction(self.about_action)

        self.lang_menu.setTitle(self.tr('menu_language'))
        self.lang_menu.clear()
        self.lang_action_group.removeAction(self.en_action)
        self.lang_action_group.removeAction(self.zh_action)
        self.lang_action_group.removeAction(self.ja_action)
        self.lang_menu.addAction(self.en_action)
        self.lang_menu.addAction(self.zh_action)
        self.lang_menu.addAction(self.ja_action)
        self.lang_action_group.addAction(self.en_action)
        self.lang_action_group.addAction(self.zh_action)
        self.lang_action_group.addAction(self.ja_action)
        
        if self.lang == 'en': self.en_action.setChecked(True)
        elif self.lang == 'zh': self.zh_action.setChecked(True)
        elif self.lang == 'ja': self.ja_action.setChecked(True)

    def change_language(self, lang_code):
        self.lang = lang_code
        self.retranslate_ui()

    def show_about_dialog(self):
        QMessageBox.about(self, self.tr('about_title'), self.tr('about_dialog_text'))

    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr('select_video_dialog_title'), '',
                                                 self.tr('video_files_filter'))
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        self.video_path = file_path
        self.permanent_roi = None
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.video_label.set_video_frame(QPixmap.fromImage(q_img), QSize(w, h))
                
                self.select_video_btn.setEnabled(True)
                self.process_btn.setEnabled(False)
                self.status_label.setText(self.tr('status_label_select_roi'))
                self.fine_tune_roi_action.setEnabled(True)
            cap.release()

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, self.tr('select_output_dir_dialog_title'), self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.status_label.setText(self.tr('status_label_output_dir', dir_path=dir_path))

    def update_roi(self, roi):
        self.permanent_roi = roi
        self.video_label.permanent_roi = roi
        self.video_label.update()
        if roi and not roi.isNull() and roi.isValid():
             self.process_btn.setEnabled(True)
             self.status_label.setText(self.tr('status_label_roi_selected'))
        else:
             self.process_btn.setEnabled(False)
             self.status_label.setText(self.tr('status_label_select_roi'))

    def open_roi_tuning_dialog(self):
        video_size = self.video_label.original_size
        if not video_size.isValid(): return

        initial_roi = self.permanent_roi if self.permanent_roi and not self.permanent_roi.isNull() else QRect(0, 0, video_size.width(), video_size.height())

        dialog_translations = {
            'title': self.tr('roi_tuning_title'),
            'x': self.tr('roi_dialog_x'),
            'y': self.tr('roi_dialog_y'),
            'w': self.tr('roi_dialog_w'),
            'h': self.tr('roi_dialog_h'),
        }
        dialog = RoiTuningDialog(initial_roi, video_size.width(), video_size.height(), self, translations=dialog_translations)
        if dialog.exec_() == QDialog.Accepted:
            self.update_roi(dialog.get_roi())
            
    def update_intermediate_frames_combo_state(self, index):
        self.intermediate_frames_combo.setEnabled(index == 1)

    def toggle_processing(self):
        if hasattr(self, 'video_processor') and self.video_processor and self.video_processor.isRunning():
            self.stop_processing()
        else:
            self.start_processing()
    
    def stop_processing(self):
        try:
            if hasattr(self, 'video_processor') and self.video_processor:
                self.video_processor.requestInterruption()
                self.video_processor.wait()
                
            self.process_btn.setText(self.tr('process_btn_start'))
            self.process_btn.setEnabled(True)
            self.select_video_btn.setEnabled(True)
            self.select_dir_btn.setEnabled(True)
            self.progress_bar.hide()
            self.progress_label.hide()
            self.status_label.setText(self.tr('status_label_stopped'))
            self.remaining_time_label.setText(self.tr('remaining_time_label_template', time_text=self.tr('time_placeholder')))
        except Exception as e:
            self.handle_error(f"终止处理时发生错误: {str(e)}")

    def start_processing(self):
        if not self.video_path or not self.permanent_roi or self.permanent_roi.isNull():
            self.status_label.setText(self.tr('status_label_error_roi'))
            return
            
        try:
            is_overflow = self.overflow_combo.currentIndex() == 1
            debug_output = self.debug_output_combo.currentIndex() == 1
            output_intermediate_frames = self.intermediate_frames_combo.currentIndex() == 1 and debug_output
            
            roi_tuple = (self.permanent_roi.x(), self.permanent_roi.y(), 
                         self.permanent_roi.width(), self.permanent_roi.height())
            
            self.video_processor = VideoProcessor(
                self.video_path, self.output_dir, roi_tuple, 
                is_overflow=is_overflow, debug_output=debug_output,
                output_intermediate_frames=output_intermediate_frames
            )
            self.video_processor.progress_updated.connect(self.update_progress)
            self.video_processor.finished.connect(self.processing_finished)
            self.video_processor.speed_updated.connect(self.update_speed)
            
            self.process_btn.setText(self.tr('process_btn_stop'))
            self.select_video_btn.setEnabled(False)
            self.select_dir_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.show()
            self.status_label.setText(self.tr('status_label_processing'))
            
            self.speed_label.setText(self.tr('speed_label_template', speed_text=self.tr('speed_calculating')))
            self.remaining_time_label.setText(self.tr('remaining_time_label_template', time_text=self.tr('time_calculating')))
            
            self.processing_start_time = time.time()
            self.processed_frames = 0
            
            self.video_processor.start()
        except Exception as e:
            self.handle_error(f"启动处理时发生错误: {str(e)}")

    def update_progress(self, value):
        try:
            self.progress_bar.setValue(value)
            self.progress_label.setText(self.tr('progress_label_template', value=value))
            
            if hasattr(self, 'video_processor') and self.video_processor:
                total_frames = self.video_processor.total_frames
                self.processed_frames = int(total_frames * value / 100)
                
                if self.processed_frames > 0:
                    elapsed_time = time.time() - self.processing_start_time
                    frames_per_second = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
                    
                    if frames_per_second > 0:
                        remaining_frames = total_frames - self.processed_frames
                        remaining_seconds = remaining_frames / frames_per_second
                        minutes = int(remaining_seconds // 60)
                        seconds = int(remaining_seconds % 60)
                        time_text = f'{minutes:02d}:{seconds:02d}'
                        self.remaining_time_label.setText(self.tr('remaining_time_label_template', time_text=time_text))
            
            if not self.progress_bar.isVisible():
                self.progress_bar.show()
                self.progress_label.show()
        except Exception as e:
            self.handle_error(f"更新进度时发生错误: {str(e)}")

    def update_speed(self, avg_time):
        try:
            avg_time_sec = avg_time / 1000.0
            fps = 1.0 / avg_time_sec if avg_time_sec > 0 else 0
            
            if fps > 1.0:
                speed_text = self.tr('fps_unit', fps=fps)
            else:
                speed_text = self.tr('spf_unit', spf=avg_time_sec)
            self.speed_label.setText(self.tr('speed_label_template', speed_text=speed_text))
        except Exception as e:
            self.handle_error(f"更新速度时发生错误: {str(e)}")
    
    def processing_finished(self, result):
        try:
            self.progress_bar.hide()
            self.progress_label.hide()
            self.process_btn.setText(self.tr('process_btn_start'))
            self.process_btn.setEnabled(True)
            self.select_video_btn.setEnabled(True)
            self.select_dir_btn.setEnabled(True)
            self.remaining_time_label.setText(self.tr('remaining_time_label_template', time_text=self.tr('time_placeholder')))

            if result.startswith('错误'):
                self.status_label.setText(result) # 保持原始错误信息
            elif result.startswith('Error'):
                self.status_label.setText(result) # 保持原始错误信息
            elif result == '处理被用户终止':
                self.status_label.setText(self.tr('status_label_stopped'))
            else:
                self.status_label.setText(self.tr('status_label_finished', result=result))
        except Exception as e:
            self.handle_error(f"处理完成回调时发生错误: {str(e)}")

    def handle_error(self, error_message):
        """全局错误处理函数"""
        print(f"[ERROR] {error_message}")
        self.status_label.setText(self.tr('status_label_error_generic', error_message=error_message))
        
        if hasattr(self, 'video_processor') and self.video_processor and self.video_processor.debug_output:
            self.video_processor._log_error(error_message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 启动时显示语言选择对话框
    selected_lang = LanguageSelectionDialog.get_language()
    
    if selected_lang:
        window = MainWindow(language=selected_lang)
        window.show()
        sys.exit(app.exec_())
    else:
        # 如果用户关闭了语言选择对话框，则退出程序
        sys.exit(0)