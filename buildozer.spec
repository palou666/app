[app]
title = 诺脊康
package.name = nuojikang
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx
source.exclude_exts = spec,pyc,pyo
source.exclude_dirs = tests, bin, .git, .buildozer, __pycache__, build, dist
version = 0.1
requirements = python3,kivy==2.2.0,plyer,opencv-python==4.5.5.64,numpy,onnxruntime==1.15.0
android.api = 33
android.minapi = 24
android.ndk = 28c
android.permissions = CAMERA,INTERNET,WRITE_EXTERNAL_STORAGE
android.accept_sdk_license = True
android.allow_backup = True
android.archs = arm64-v8a, armeabi-v7a
android.copy_libs = True

[buildozer]
log_level = 2
warn_on_root = 1
