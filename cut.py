#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:16:50 2025

@author: GitHub@KirisameMarisa-DAZE (master.spark.kirisame.marisa.daze@gmail.com). All rights reserved. © 2021~2025
"""

import os
import cv2

def cut_video_to_roi(input_video_path, output_video_path, roi_x, roi_y, roi_width, roi_height):
    """
    Cut the video into regions of interest and save.
    """
    # 打开输入视频并校验
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open input video: {input_video_path}")

    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 校验ROI范围
    if (roi_x + roi_width > width) or (roi_y + roi_height > height):
        cap.release()
        raise ValueError(f"The ROI area exceeds the video size (original size: {width}x{height})")

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_width, roi_height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Failed to create output video: {output_video_path}")

    # 处理每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        out.write(roi_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"Output video saved to: {output_video_path}")

# 用户交互部分
if __name__ == "__main__":
    # 输入路径处理
    input_path = os.path.expanduser(
        input('Input video path >>>').strip().strip('"\''))
    input_path = os.path.normpath(input_path)

    # 自动生成输出文件名（原名称_output.扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # 获取无扩展名的文件名
    # extension = os.path.splitext(input_path)[1]  # 获取原扩展名（如 .avi）
    output_name = f"{base_name}_output.mp4"

    # 获取输出目录
    output_dir = os.path.expanduser(
        input('Output directory (leave it blank to save to the original directory) >>>').strip().strip('"\''))
    if not output_dir:  # 如果用户直接回车，使用输入文件所在目录
        output_dir = os.path.dirname(input_path)
    output_dir = os.path.normpath(output_dir)

    # 拼接完整输出路径
    output_path = os.path.join(output_dir, output_name)


    # ROI参数处理
    while True:
        try:
            x, y, w, h = map(int, input('ROI parameters: x y w h (separated by spaces) >>> ').split())
            break
        except ValueError:
            print("The input format is incorrect, please enter 4 integers (separated by spaces)")

    # 执行处理
    print('>>>Start processing<<<')
    cut_video_to_roi(input_path, output_path, x, y, w, h)