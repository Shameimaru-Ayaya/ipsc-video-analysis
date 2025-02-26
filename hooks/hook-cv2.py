from PyInstaller.utils.hooks import collect_dynamic_libs

hiddenimports = ['cv2']
binaries = collect_dynamic_libs('cv2')