# hooks/hook-cv2.py
from PyInstaller.utils.hooks import collect_data_files

# 确保包含所有 OpenCV 数据文件
datas = collect_data_files('cv2')