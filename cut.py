#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:16:50 2025

@author: page (shiki.no.omusubi@gmail.com). All rights reserved. © 2021~2025
"""

import cv2

def cut_video_to_roi(input_video_path, output_video_path, roi_x, roi_y, roi_width, roi_height):
    """
    将视频剪切成感兴趣区域并保存

    :param input_video_path: 输入视频的路径
    :param output_video_path: 输出视频的路径
    :param roi_x: 感兴趣区域的左上角x坐标
    :param roi_y: 感兴趣区域的左上角y坐标
    :param roi_width: 感兴趣区域的宽度
    :param roi_height: 感兴趣区域的高度
    """
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要修改编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_width, roi_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 剪切感兴趣区域
        roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        # 写入输出视频
        out.write(roi_frame)

    # 释放资源
    cap.release()
    out.release()

# 示例用法
input_video_path = '2848.avi'
output_video_path = '2848.mp4'
roi_x, roi_y, roi_width, roi_height = 530, 237, 116, 91

cut_video_to_roi(input_video_path, output_video_path, roi_x, roi_y, roi_width, roi_height)