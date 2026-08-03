[app]

# 应用名称
title = 诺脊康

# 包名（全小写，无特殊字符）
package.name = nuojikang

# 包域名（反向域名，用于生成完整包名）
package.domain = org.example

# 源码目录（当前目录）
source.dir = .

# 包含的源码扩展名
source.include_exts = py,png,jpg,kv,atlas,ttf,json

# 排除的文件扩展名
source.exclude_exts = spec,pyc,pyo

# 排除的目录
source.exclude_dirs = tests, bin, .git, buildozer, __pycache__, build, dist

# 应用版本号
version = 0.1

# 要求（已锁定 numpy 版本为 1.26.5，确保与 opencv-python 兼容）
requirements = python3==3.11.9, kivy==2.3.0, plyer, opencv-python, numpy==1.26.5, onnxruntime, pyjnius==1.6.0

# 指定 Python 版本（必须和 requirements 一致）
android.python_version = 3.11.9

# Gradle 依赖（留空即可）
android.gradle_dependencies =

# Android SDK 和 NDK（使用稳定版本）
android.ndk = 23b
android.sdk = 33

# API 级别
android.api = 33
android.minapi = 24

# 权限（摄像头、网络、存储）
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE

# 允许接受 SDK 许可（Colab 中必须）
android.accept_sdk_license = True

# 其他可选配置
android.allow_backup = True
android.manifest_application = <application android:label="诺脊康" android:icon="@drawable/icon" />

# 日志级别
android.logcat_filters = *:S python:D

# 忽略 setup.py 问题（Colab 常见）
android.ignore_setup_py = True

# 复制库文件
android.copy_libs = True

# 支持多架构（保留 arm64-v8a 和 armeabi-v7a）
android.archs = arm64-v8a, armeabi-v7a

# 启用调试
android.debug = True

# 输出颜色
color = always

# 启动方式（默认 sdl2）
p4a.bootstrap = sdl2