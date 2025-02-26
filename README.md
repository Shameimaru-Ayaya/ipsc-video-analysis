# iPSC-Video-Analysis

[![Version](https://img.shields.io/badge/Version-2.2.1-blue?style=flat-square)](https://github.com/KirisameMarisa-DAZE/ipsc-video-analysis/releases)
[![GitHub Activity](https://img.shields.io/badge/GitHub-Active-brightgreen)](https://github.com/KirisameMarisa-DAZE/ipsc-video-analysis)
[![Build Status](https://img.shields.io/badge/Progress-In%20Progress-yellow)](https://github.com/KirisameMarisa-DAZE/ipsc-video-analysis/)
[![GitHub License](https://img.shields.io/github/license/KirisameMarisa-DAZE/ipsc-video-analysis)](LICENSE)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/KirisameMarisa-DAZE/ipsc-video-analysis/build.yml?branch=main)](https://github.com/KirisameMarisa-DAZE/ipsc-video-analysis/branches)

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

A method for analyzing mechanical signals from iPSC experimental videos.

# VideoCropper

# 中文版

## Abstract ✨
你说的对，但是 **VideoCropper** 是由 [GitHub@霧雨魔理沙](https://github.com/KirisameMarisa-DAZE/) 自主研发的一款全新视频裁剪应用程序。程序发生在一个被称作「Python」的系统环境，在这里，被系统选中的视频将被授予「ROI」，导引裁剪之力。你将扮演一位名为「用户」的神秘角色，在自由的开发中邂逅状态各异、色彩独特的视频们，和他们一起击败干扰源，找回失散的关键信息——同时，逐步发掘「实验数据」的真相。

## Introduction 🎯
VideoCropper 是一款基于 Python 的用于裁剪视频 ROI 区域并输出裁剪后视频的小软件，其源代码是 `./cut.py`。~~真没啥技术含量~~

因此最简便的使用方法：
1. 安装并配置 Python 环境
2. 使用 `pip install` 安装相关包
3. 直接运行 `cut.py`（不推荐从 Releases 下载程序包）

> **环境配置指南**  
> - 未安装 Python? [立即下载](https://www.python.org/downloads/)  
> - 懒人福音 [Anaconda](https://www.anaconda.com/download)（自动配置系统路径）  
> - 环境配置教程 [菜鸟教程](https://www.runoob.com/python3/)

## Methods ⚙️
### For Rebellious Users
开发者被迫为坚持不安装 Python 的用户提供应用程序包（支持 Windows/Linux/macOS），但：
- 🐛 可能存在亿点点 bug（笑）
- 🔄 更新支持力度 <<< `cut.py`

> **安全警告** ⚠️  
> - 所有安装包均无签名验证，必被系统拦截  
> - Windows 用户：长按 `无视风险继续运行`  
> - Mac 用户：思考の时间到（乐）  
> - Linux 用户：开发者已跑路，请自求多福（逃）

~~系统是这样的，Windows用户只要无视风险继续运行就可以，但是Unix用户要考虑的事情就很多了~~

## MacOS Special Support 🍎
### 解压后文件说明
```
VideoCropper_macOS_*.app      # 唯一有用の存在
VideoCropper_macOS_*/         # 建议直接删除！
├── VideoCropper_macOS_*      # 伪装成Unix可执行文件の谜の物体
└── _internal/                # 程序の心脏（严禁触碰！）
```

### 开光仪式 🔮
```bash
# 清除隔离属性
sudo xattr -cr <拖入目标文件到此处自动生成路径>
# 修复权限
sudo chmod -R 755 <同上操作>
```

> 连终端都不会用的大哥哥真是❤️雑魚❤️呢～

## Results 🎉
执行完咒语后，即可享受魔法般的裁剪体验~

## Future Work 🌌
> 还早嘞！这才哪到哪（悲）  
> まだまだ

## Conclusion 🤔
> 我打代码？真的假的  
> 会赢吗


# English Version

## Abstract ✨
You're absolutely right, but **VideoCropper** is a brand-new video cropping application independently developed by [GitHub@KirisameMarisa-DAZE](https://github.com/KirisameMarisa-DAZE). The program takes place in a system environment called "Python", where videos chosen by the system will be granted "ROI" to guide the power of cropping. You'll play as a mysterious character named "User", encountering uniquely colorful videos in free development, fighting against noise sources together to recover lost critical information — while gradually uncovering the truth of "experimental data".

## Introduction 🎯
VideoCropper is a small Python-based software for cropping video ROI regions and exporting processed videos. Source code: `./cut.py`. ~~Totally no technical content~~

Recommended workflow:
1. Install and configure Python environment
2. Install dependencies via `pip install`
3. Directly run `cut.py` (Not recommended to download packages from Releases)

> **Environment Setup Guide**  
> - No Python? [Download Now](https://www.python.org/downloads/)  
> - For lazy souls: [Anaconda](https://www.anaconda.com/download) (auto PATH configuration)  
> - Tutorial: [Rookie Tutorial](https://www.runoob.com/python3/)

## Methods ⚙️
### For Rebellious Users
We reluctantly provide executable packages (Windows/Linux/macOS) with:
- 🐛 Potential "minor" bugs (lol)
- 🔄 Update support <<< `cut.py`

> **Security Warning** ⚠️  
> - All packages are unsigned and WILL be blocked  
> - Windows users: Long-press `Run anyway`  
> - Mac users: Time for contemplation (heh)  
> - Linux users: Developer has fled, good luck (escaped)

~~ The system is such that Windows users just have to ignore the risks and keep running, but Unix users have a lot to think about ~~

## macOS Special Support 🍎
### Post-extraction Guide
```
VideoCropper_macOS_*.app      # The Chosen One
VideoCropper_macOS_*/         # Delete immediately!
├── VideoCropper_macOS_*      # Suspicious "Unix executable"
└── _internal/                # Program's heart (DO NOT TOUCH!)
```

### Enchantment Ritual 🔮
```bash
# Clean quarantine attributes
sudo xattr -cr <Drag file here for auto-path>
# Fix permissions
sudo chmod -R 755 <Same as above>
```

> Even a grown-up who can't even handle basic terminal commands is seriously ❤️small fry❤️, huh~

## Results 🎉
After chanting the spells, enjoy the magic cropping experience~

## Future Work 🌌
> Way too early! This is just the beginning (sigh)

## Conclusion 🤔
> Me coding? Seriously  
> Will we win?