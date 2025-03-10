#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 14:47:34 2025

@author: GitHub@Shameimaru-Ayaya (shameimaru.ayaaya@gmail.com). All rights reserved. © 2021~2025
"""

import cv2
import numpy as np

def merge_close_rois(rois, min_distance, frame_width, frame_height):
    """合并距离较近的ROIs"""
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
                
            # 计算ROI中心点距离
            x1, y1, w1, h1 = current_roi
            x2, y2, w2, h2 = rois[j]
            
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance < min_distance:
                # 合并ROI
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

def optimize_roi_shape(binary_heatmap, x, y, w, h, padding=20):
    """优化ROI形状，使用最小外接椭圆"""
    # 提取ROI区域
    roi_mask = np.zeros_like(binary_heatmap)
    roi_mask[y:y+h, x:x+w] = binary_heatmap[y:y+h, x:x+w]
    
    # 找到ROI区域内的非零点
    points = np.column_stack(np.where(roi_mask > 0))
    
    if len(points) < 5:  # 椭圆拟合需要至少5个点
        return (x, y, w, h)
    
    # 计算最小外接椭圆
    (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
    
    # 计算椭圆边界（添加padding）
    half_w = width/2 + padding
    half_h = height/2 + padding
    
    # 确保边界在图像内
    new_x = max(0, int(center_x - half_w))
    new_y = max(0, int(center_y - half_h))
    new_w = min(binary_heatmap.shape[1] - new_x, int(width + 2*padding))
    new_h = min(binary_heatmap.shape[0] - new_y, int(height + 2*padding))
    
    return (new_x, new_y, new_w, new_h)

def find_intensity_boundary(heatmap, center_y, center_x, max_radius=100):
    """根据热力梯度寻找边界"""
    center_intensity = heatmap[center_y, center_x]
    height, width = heatmap.shape
    angles = np.linspace(0, 2*np.pi, 16)  # 16个方向
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
            
            # 当热力值突降超过阈值时认为找到边界
            if intensity_drop > 0.3 or current_intensity < center_intensity * 0.2:
                boundary_points.append((x, y))
                break
                
            last_intensity = current_intensity
    
    return np.array(boundary_points) if boundary_points else None

def generate_ellipse_roi(points, padding=20):
    """生成椭圆形ROI，返回(center_x, center_y, axes_width, axes_height, angle)"""
    if len(points) < 5:  # 椭圆拟合需要至少5个点
        return None
        
    # 计算最小外接椭圆
    (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
    
    # 添加padding
    width += 2 * padding
    height += 2 * padding
    
    return (int(center_x), int(center_y), int(width/2), int(height/2), angle)

def motion_detection():
    video_path = '/Users/page/Documents/-文稿/day8 stimulation/X1.25_#14E_5s_background+stimulation_2574.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件：{video_path}")
        return
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算面积阈值
    min_area = frame_width * frame_height * 0.001  # 从0.5%降低到0.1%
    max_area = frame_width * frame_height * 0.05   # 从15%降低到5%
    
    print(f"开始处理视频，总帧数：{total_frames}")
    
    # 创建累积热力图
    motion_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # 优化的背景分割器
    backSub = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=True
    )
    
    frame_count = 0
    frame_step = 2
    
    # 创建椭圆形结构元素
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 创建用于显示的窗口
    cv2.namedWindow('分析结果', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_step != 0:
            frame_count += 1
            continue
            
        # 高斯模糊预处理
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # 使用KNN算法生成前景蒙版
        fgMask = backSub.apply(gray)
        
        # 形态学操作优化
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_noise)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_noise)
        
        # 累积运动区域
        motion_heatmap += fgMask.astype(np.float32)
        frame_count += 1
    
    print("热力图生成完成，开始分析ROI...")
    
    # 归一化热力图到0-255
    normalized_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 使用高斯模糊平滑热力图
    smoothed_heatmap = cv2.GaussianBlur(normalized_heatmap, (15, 15), 0)
    
    # 计算面积阈值
    min_area = frame_width * frame_height * 0.001  # 最小面积为0.1%
    max_area = frame_width * frame_height * 0.05   # 最大面积为5%
    
    # 1. 基于热力强度的ROI候选区域检测
    kernel_size = 50  # 局部区域大小
    max_filtered = cv2.dilate(smoothed_heatmap, np.ones((kernel_size, kernel_size), np.uint8))
    maxima = (smoothed_heatmap == max_filtered) & (smoothed_heatmap > np.mean(smoothed_heatmap) + np.std(smoothed_heatmap))
    
    # 获取局部最大值的坐标
    coordinates = np.column_stack(np.where(maxima))
    intensities = [smoothed_heatmap[y, x] for y, x in coordinates]
    sorted_indices = np.argsort(intensities)[::-1]
    
    # 2. 基于面积的ROI候选区域检测
    print("\n开始基于面积的ROI检测...")
    print(f"图像尺寸: {frame_width}x{frame_height}")
    print(f"面积阈值范围: {min_area:.1f} - {max_area:.1f} 像素")
    
    # 使用Otsu's二值化方法替代自适应阈值
    _, binary_heatmap = cv2.threshold(
        smoothed_heatmap,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu阈值处理后的非零像素数: {np.count_nonzero(binary_heatmap)}")
    
    # 增强形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_heatmap = cv2.morphologyEx(binary_heatmap, cv2.MORPH_OPEN, kernel)
    binary_heatmap = cv2.morphologyEx(binary_heatmap, cv2.MORPH_CLOSE, kernel)
    binary_heatmap = cv2.dilate(binary_heatmap, kernel, iterations=2)
    print(f"形态学处理后的非零像素数: {np.count_nonzero(binary_heatmap)}")

    # 查找轮廓
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    print(f"检测到的轮廓数量: {len(contours)}")

    # 3. 结合两种方法选择最终的ROI
    min_distance = 100
    # 存储椭圆形ROI参数：(center_x, center_y, axes_width, axes_height, angle)
    stable_rois = []
    
    # 首先基于面积筛选合适的轮廓
    valid_contours = []
    padding = 20  # 添加padding参数
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
                    # 添加padding并直接使用椭圆拟合结果
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
    
    # 只有在面积法检测不足时才使用热力图方法补充
    if len(stable_rois) < 2:
        print("\n面积法检测ROI不足，使用热力图方法补充...")
        for idx in sorted_indices:
            y, x = coordinates[idx]
            
            # 检查是否与现有ROI太近
            if any(np.linalg.norm(np.array([y, x]) - np.array([cy, cx])) < min_distance 
                   for cx, cy, _, _, _ in stable_rois):
                continue
            
            # 使用热力梯度寻找边界点
            boundary_points = find_intensity_boundary(smoothed_heatmap, y, x)
            if boundary_points is not None:
                ellipse_roi = generate_ellipse_roi(boundary_points)
                if ellipse_roi:
                    stable_rois.append(ellipse_roi)
                    cx, cy, ax, ay, angle = ellipse_roi
                    print(f'热力椭圆ROI: 中心=({cx}, {cy}), 轴长=({ax*2}, {ay*2}), 角度={angle:.1f}°')
                    
                    if len(stable_rois) >= 3:
                        break

    print("开始显示结果...")
    
    # 第二次遍历：显示结果
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 创建显示画布
        display_height = frame_height
        display_width = frame_width * 2
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # 在原始帧上绘制ROI
        frame_with_roi = frame.copy()
        for i, (cx, cy, ax, ay, angle) in enumerate(stable_rois):
            # 绘制椭圆
            cv2.ellipse(frame_with_roi, (cx, cy), (ax, ay), angle, 0, 360, (0, 255, 0), 2)
            
            # 添加ROI标签
            label_x = cx - 20
            label_y = cy - ay - 10
            cv2.putText(frame_with_roi, f'ROI {i+1}', (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        display[:, :frame_width] = frame_with_roi
        heatmap_colored = cv2.applyColorMap(
            cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        display[:, frame_width:] = heatmap_colored
        
        cv2.imshow('分析结果', display)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return stable_rois

if __name__ == "__main__":
    stable_rois = motion_detection()