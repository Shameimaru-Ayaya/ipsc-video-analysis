# hooks/hook-cv2.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, get_module_file_attribute
import os

# 收集所有 cv2 的子模块
hiddenimports = collect_submodules('cv2')

# 收集 cv2 的数据文件（包含 config.py 等）
datas = collect_data_files('cv2')

# 显式包含 config.py
cv2_dir = os.path.dirname(get_module_file_attribute('cv2'))
config_path = os.path.join(cv2_dir, 'config.py')
if os.path.exists(config_path):
    datas.append((config_path, 'cv2'))