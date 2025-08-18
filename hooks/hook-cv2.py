# hooks/hook-cv2.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集所有 cv2 的子模块
hiddenimports = collect_submodules('cv2')

# 收集 cv2 的数据文件（包含 config.py 等）
datas = collect_data_files('cv2')