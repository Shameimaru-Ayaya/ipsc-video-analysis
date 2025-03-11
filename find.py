#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 14:47:34 2025

@author: GitHub@Shameimaru-Ayaya (shameimaru.ayaaya@gmail.com). All rights reserved. © 2021~2025
"""

import cv2
import numpy as np
import os
from datetime import datetime
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class MotionDetector:
    def __init__(self, output_dir=None):
        """
        初始化参数及输出目录
        """
        if output_dir is None:
            self.output_dir = os.path.expanduser("~/Documents/iPSC_Analysis_Results")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    @staticmethod
    def merge_close_rois(rois, min_distance, frame_width, frame_height):
        """合并距离较近的ROI"""
        if not rois:
            return []
        
        merged_rois = []
        used = set()
        
        for i in range(len(rois)):
            if i in used:
                continue
                
            current_roi = list(rois[i])  # 转换为list以便修改
            used.add(i)
            
            # 检查其他ROI是否需要合并
            for j in range(i + 1, len(rois)):
                if j in used:
                    continue
                    
                x1, y1, w1, h1 = current_roi
                x2, y2, w2, h2 = rois[j]
                
                center1 = (x1 + w1 / 2, y1 + h1 / 2)
                center2 = (x2 + w2 / 2, y2 + h2 / 2)
                
                distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
                
                if distance < min_distance:
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    
                    current_roi = [x, y, w, h]
                    used.add(j)
            
            # 确保ROI不超出边界
            current_roi[0] = max(0, current_roi[0])
            current_roi[1] = max(0, current_roi[1])
            current_roi[2] = min(frame_width - current_roi[0], current_roi[2])
            current_roi[3] = min(frame_height - current_roi[1], current_roi[3])
            
            merged_rois.append(tuple(current_roi))
        
        return merged_rois

    @staticmethod
    def optimize_roi_shape(binary_heatmap, x, y, w, h, padding=20):
        """优化ROI形状，使用最小外接椭圆"""
        roi_mask = np.zeros_like(binary_heatmap)
        roi_mask[y:y + h, x:x + w] = binary_heatmap[y:y + h, x:x + w]
        points = np.column_stack(np.where(roi_mask > 0))
        
        if len(points) < 5:
            return (x, y, w, h)
        
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
        half_w = width / 2 + padding
        half_h = height / 2 + padding
        
        new_x = max(0, int(center_x - half_w))
        new_y = max(0, int(center_y - half_h))
        new_w = min(binary_heatmap.shape[1] - new_x, int(width + 2 * padding))
        new_h = min(binary_heatmap.shape[0] - new_y, int(height + 2 * padding))
        
        return (new_x, new_y, new_w, new_h)

    @staticmethod
    def find_intensity_boundary(heatmap, center_y, center_x, max_radius=100):
        """根据热力梯度寻找边界"""
        center_intensity = heatmap[center_y, center_x]
        height, width = heatmap.shape
        angles = np.linspace(0, 2 * np.pi, 16)  # 16个方向
        boundary_points = []
        
        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)
            r = 0
            last_intensity = center_intensity
            
            while r < max_radius:
                r += 1
                x = int(center_x + r * dx)
                y = int(center_y + r * dy)
                
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                    
                current_intensity = heatmap[y, x]
                intensity_drop = (last_intensity - current_intensity) / last_intensity
                
                if intensity_drop > 0.3 or current_intensity < center_intensity * 0.2:
                    boundary_points.append((x, y))
                    break
                    
                last_intensity = current_intensity
        
        return np.array(boundary_points) if boundary_points else None

    @staticmethod
    def generate_ellipse_roi(points, padding=20):
        """生成椭圆形ROI，返回 (center_x, center_y, axes_width, axes_height, angle)"""
        if len(points) < 5:
            return None
        
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
        width += 2 * padding
        height += 2 * padding
        
        return (int(center_x), int(center_y), int(width / 2), int(height / 2), angle)

    @staticmethod
    def has_cuda():
        """检查是否支持CUDA"""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            return count > 0
        except:
            return False

    @classmethod
    def process_frame_gpu(cls, frame, backSub, kernel_noise):
        """使用GPU处理单帧"""
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            gray = gpu_blurred.download()
            fgMask = backSub.apply(gray)
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(fgMask)
            gpu_mask = cv2.cuda.morphologyEx(gpu_mask, cv2.MORPH_OPEN, kernel_noise)
            return gpu_mask.download()
        except Exception as e:
            print(f"GPU处理出错，切换到CPU: {str(e)}")
            return cls.process_frame_cpu(frame, backSub, kernel_noise)

    @staticmethod
    def process_frame_cpu(frame, backSub, kernel_noise):
        """使用CPU处理单帧"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fgMask = backSub.apply(gray)
        return cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_noise)

    @classmethod
    def process_frame_batch(cls, args):
        """处理一批帧"""
        frames, backSub, kernel_noise, use_gpu = args
        results = np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.float32)
        
        for frame in frames:
            if use_gpu:
                mask = cls.process_frame_gpu(frame, backSub, kernel_noise)
            else:
                mask = cls.process_frame_cpu(frame, backSub, kernel_noise)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            results += mask.astype(np.float32)
        
        return results

    class Logger:
        """日志记录类，将输出同时写入终端和日志文件"""
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    def analyze(self, video_path):
        """
        主函数：输入视频路径，处理视频，保存输出图像、视频和日志，返回ROI的各项数值
        """
        try:
            # 创建输出结果目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(self.output_dir, f"analysis_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            log_file = os.path.join(result_dir, "analysis_log.txt")
            sys.stdout = MotionDetector.Logger(log_file)
            
            use_gpu = MotionDetector.has_cuda()
            if use_gpu:
                print("检测到CUDA支持，将使用GPU加速")
            else:
                print("未检测到CUDA支持，将使用CPU处理")
            
            cpu_count = mp.cpu_count()
            print(f"检测到{cpu_count}个CPU核心")
            
            print(f"尝试打开视频：{video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误：无法打开视频文件：{video_path}")
                return None
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print("视频信息：")
            print(f"- 分辨率：{frame_width}x{frame_height}")
            print(f"- 总帧数：{total_frames}")
            print(f"- 帧率：{fps}")
            
            print("初始化热力图...")
            motion_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
            
            print("配置背景分割器...")
            backSub = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=False
            )
            
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            
            batch_size = 32
            frame_step = 3
            frames_batch = []
            frame_count = 0
            
            print(f"开始处理视频帧（批大小：{batch_size}，步长：{frame_step}）...")
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_step == 0:
                        frames_batch.append(frame.copy())
                        if len(frames_batch) >= batch_size:
                            future = executor.submit(
                                MotionDetector.process_frame_batch,
                                (frames_batch.copy(), backSub, kernel_noise, use_gpu)
                            )
                            futures.append(future)
                            frames_batch = []
                    frame_count += 1
                    if frame_count % 1000 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧")
                
                if frames_batch:
                    future = executor.submit(
                        MotionDetector.process_frame_batch,
                        (frames_batch.copy(), backSub, kernel_noise, use_gpu)
                    )
                    futures.append(future)
                
                print("等待所有任务完成...")
                for i, future in enumerate(futures):
                    try:
                        motion_heatmap += future.result()
                        if (i + 1) % 10 == 0:
                            print(f"已完成 {i + 1}/{len(futures)} 批处理")
                    except Exception as e:
                        print(f"处理批次 {i} 时出错: {str(e)}")
            
            print("热力图生成完成，开始分析ROI...")
            normalized_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            smoothed_heatmap = cv2.GaussianBlur(normalized_heatmap, (15, 15), 0)
            
            min_area = frame_width * frame_height * 0.001
            max_area = frame_width * frame_height * 0.05
            
            kernel_size = 50
            max_filtered = cv2.dilate(smoothed_heatmap, np.ones((kernel_size, kernel_size), np.uint8))
            maxima = (smoothed_heatmap == max_filtered) & (smoothed_heatmap > np.mean(smoothed_heatmap) + np.std(smoothed_heatmap))
            
            coordinates = np.column_stack(np.where(maxima))
            intensities = [smoothed_heatmap[y, x] for y, x in coordinates]
            sorted_indices = np.argsort(intensities)[::-1]
            
            print("\n开始基于面积的ROI检测...")
            print(f"图像尺寸: {frame_width}x{frame_height}")
            print(f"面积阈值范围: {min_area:.1f} - {max_area:.1f} 像素")
            
            _, binary_heatmap = cv2.threshold(
                smoothed_heatmap,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            print(f"Otsu阈值处理后的非零像素数: {np.count_nonzero(binary_heatmap)}")
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_heatmap = cv2.morphologyEx(binary_heatmap, cv2.MORPH_OPEN, kernel)
            binary_heatmap = cv2.morphologyEx(binary_heatmap, cv2.MORPH_CLOSE, kernel)
            binary_heatmap = cv2.dilate(binary_heatmap, kernel, iterations=2)
            print(f"形态学处理后的非零像素数: {np.count_nonzero(binary_heatmap)}")
            
            contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            print(f"检测到的轮廓数量: {len(contours)}")
            
            min_distance = 100
            stable_rois = []
            valid_contours = []
            padding = 20
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                print(f"\n轮廓 {i+1}:")
                print(f"  面积: {area:.2f} (阈值范围: {min_area:.2f} - {max_area:.2f})")
                if min_area <= area <= max_area:
                    if len(contour) >= 5:
                        (cx, cy), (width, height), angle = cv2.fitEllipse(contour)
                        aspect_ratio = min(width, height) / max(width, height)
                        print(f"  椭圆拟合: 中心=({cx:.1f}, {cy:.1f}), 宽度={width:.2f}, 高度={height:.2f}, 纵横比={aspect_ratio:.2f}")
                        if aspect_ratio > 0.3:
                            valid_contours.append(contour)
                            width += 2 * padding
                            height += 2 * padding
                            stable_rois.append((int(cx), int(cy), int(width/2), int(height/2), angle))
                            print(f"  ✓ 通过所有检查，添加面积法椭圆ROI (添加padding={padding})")
                        else:
                            print("  ✗ 纵横比不符合要求")
                    else:
                        print("  ✗ 点数不足，无法拟合椭圆")
                else:
                    print("  ✗ 面积不在有效范围内")
            
            print(f"\n面积法检测到的ROI数量: {len(stable_rois)}")
            
            if len(stable_rois) < 2:
                print("\n面积法检测ROI不足，使用热力图方法补充...")
                for idx in sorted_indices:
                    y, x = coordinates[idx]
                    if any(np.linalg.norm(np.array([y, x]) - np.array([cy, cx])) < min_distance 
                           for cx, cy, _, _, _ in stable_rois):
                        continue
                    boundary_points = MotionDetector.find_intensity_boundary(smoothed_heatmap, y, x)
                    if boundary_points is not None:
                        ellipse_roi = MotionDetector.generate_ellipse_roi(boundary_points)
                        if ellipse_roi:
                            stable_rois.append(ellipse_roi)
                            cx, cy, ax, ay, angle = ellipse_roi
                            print(f'热力椭圆ROI: 中心=({cx}, {cy}), 轴长=({ax*2}, {ay*2}), 角度={angle:.1f}°')
                            if len(stable_rois) >= 3:
                                break
            
            print("开始显示结果...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取视频第一帧")
                return stable_rois
            
            # 生成状态图（左：原始帧，中间：热力图，右：结果帧）
            status_img = np.zeros((frame_height, frame_width * 3, 3), dtype=np.uint8)
            status_img[:, :frame_width] = frame
            status_img[:, frame_width:frame_width*2] = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
            
            result_frame = frame.copy()
            legend_height = 30 * len(stable_rois)
            cv2.rectangle(result_frame, (10, 10), (200, 20 + legend_height), (0, 0, 0), -1)
            for i, (cx, cy, ax, ay, angle) in enumerate(stable_rois):
                # 绘制椭圆ROI
                cv2.ellipse(result_frame, (cx, cy), (ax, ay), angle, 0, 360, (0, 255, 0), 2)
                
                # 优化ROI编号显示
                number = str(i+1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                # 获取文本大小
                (text_width, text_height), baseline = cv2.getTextSize(number, font, font_scale, thickness)
                
                # 计算文本背景矩形的位置（位于椭圆上方）
                text_x = cx - text_width // 2
                text_y = cy - ay - 5  # 将编号放在椭圆上方
                
                # 绘制文本背景（半透明黑色矩形）
                padding = 4
                bg_rect = ((text_x - padding, text_y - text_height - padding),
                          (text_x + text_width + padding, text_y + padding))
                overlay = result_frame.copy()
                cv2.rectangle(overlay, bg_rect[0], bg_rect[1], (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)
                
                # 绘制文本（白色）
                cv2.putText(result_frame, number, (text_x, text_y),
                           font, font_scale, (255, 255, 255), thickness)
                
                # 绘制图例
                cv2.putText(result_frame, f'ROI {i+1}: ({cx},{cy}) {ax*2}x{ay*2}', 
                           (15, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            status_img[:, frame_width*2:] = result_frame
            cv2.imwrite(os.path.join(result_dir, "processing_status.png"), status_img)
            
            output_video = os.path.join(result_dir, "analysis_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.rectangle(frame, (10, 10), (200, 20 + legend_height), (0, 0, 0), -1)
                for i, (cx, cy, ax, ay, angle) in enumerate(stable_rois):
                    cv2.ellipse(frame, (cx, cy), (ax, ay), angle, 0, 360, (0, 255, 0), 2)
                    
                    # 在视频中使用相同的优化编号显示方式
                    number = str(i+1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(number, font, font_scale, thickness)
                    
                    text_x = cx - text_width // 2
                    text_y = cy - ay - 5
                    
                    padding = 4
                    bg_rect = ((text_x - padding, text_y - text_height - padding),
                              (text_x + text_width + padding, text_y + padding))
                    overlay = frame.copy()
                    cv2.rectangle(overlay, bg_rect[0], bg_rect[1], (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    cv2.putText(frame, number, (text_x, text_y),
                               font, font_scale, (255, 255, 255), thickness)
                    
                    cv2.putText(frame, f'ROI {i+1}: ({cx},{cy}) {ax*2}x{ay*2}', 
                               (15, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            print(f"\n分析结果已保存至: {result_dir}")
            print(f"- 处理状态图: processing_status.png")
            print(f"- 分析日志: analysis_log.txt")
            print(f"- 结果视频: analysis_result.mp4")
            
            return stable_rois
        except Exception as e:
            print(f"发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        if MotionDetector.has_cuda():
            cv2.setUseOptimized(True)
            cv2.cuda.setDevice(0)
        # 示例：传入视频路径，返回ROI数值
        video_path = '/Users/page/Documents/-文稿/lpz/4-OldCM_Old/20241120/20241120_photo=1.25x_cell(1,1,M)=Old-CM_Day6_2826.avi'
        detector = MotionDetector()
        rois = detector.analyze(video_path)
        if rois is None:
            print("处理失败")
            sys.exit(1)
        else:
            print("检测到的ROI：")
            for roi in rois:
                print(roi)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
