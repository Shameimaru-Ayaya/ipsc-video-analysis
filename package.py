import PyInstaller.__main__
import platform

system = platform.system()

params = [
    'cut.py',             # 你的主程序文件
    '--name=VideoCropper', # 程序名称
    '--onefile',           # 打包为单个文件
    '--windowed',          # 不显示控制台窗口
    '--add-data=ffmpeg;.', # 包含ffmpeg二进制文件（根据系统调整路径）
    '--icon=cut.ico'       # 程序图标
]

if system == 'Darwin':
    params.append('--osx-bundle-identifier=com.yourcompany.VideoCropper')

PyInstaller.__main__.run(params)