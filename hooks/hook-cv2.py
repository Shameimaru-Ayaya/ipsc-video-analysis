# hooks/hook-cv2.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 只收集子模块和数据文件
hiddenimports = collect_submodules('cv2')
datas = collect_data_files('cv2')