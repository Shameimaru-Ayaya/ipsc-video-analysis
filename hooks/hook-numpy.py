# hooks/hook-numpy.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('numpy')
hiddenimports += collect_submodules('numpy._core')
datas = collect_data_files('numpy')