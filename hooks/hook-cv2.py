# hooks/hook-cv2.py
from PyInstaller.utils.hooks import copy_metadata

# 告诉 PyInstaller 复制 opencv-python 包的所有元数据（metadata）文件。
# 这其中就包括了它运行时需要的 config.py 和其他重要数据。
datas = copy_metadata('opencv-python')