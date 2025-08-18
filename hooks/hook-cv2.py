# hooks/hook-cv2.py
from PyInstaller.utils.hooks import collect_all, collect_data_files, get_module_file_attribute
import os
import glob

# 收集所有 cv2 模块及其依赖
datas, binaries, hiddenimports = collect_all('cv2')

# 显式添加 config.py 和 config-*.py
cv2_dir = os.path.dirname(get_module_file_attribute('cv2'))
config_files = glob.glob(os.path.join(cv2_dir, "config*.py"))
for f in config_files:
    datas.append((f, 'cv2'))

# 确保 cv2 的 .so 或 .pyd 文件也被打包
for ext in ('*.pyd', '*.so'):
    for f in glob.glob(os.path.join(cv2_dir, ext)):
        binaries.append((f, 'cv2'))