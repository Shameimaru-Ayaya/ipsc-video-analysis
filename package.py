import PyInstaller.__main__
import platform
import os

system = platform.system()

# 动态获取 FFmpeg 路径（适配不同环境）
ffmpeg_src = os.path.expanduser("/usr/local/bin/ffmpeg")  # 假设 Homebrew 安装路径

params = [
    'cut.py',
    '--name=VideoCropper',
    '--onefile',
    '--windowed',
    f'--add-data={ffmpeg_src}:bin/ffmpeg',  # 源路径绝对，目标路径相对
    '--icon=cut.ico'
]

if system == 'Darwin':
    params.append('--osx-bundle-identifier=io.github.KirisameMarisa-DAZE.VideoCropper')

PyInstaller.__main__.run(params)