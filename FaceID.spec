# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# Paths
project_root = os.path.abspath("Authentication")
app_script = os.path.join(project_root, "app.py")
icon_path = os.path.join(project_root, "icon.ico")
model_path = os.path.join(project_root, "siamesemodelv2.h5")
model_py = os.path.join(project_root, "model.py")
data_prep_py = os.path.join(project_root, "data_preprocessing.py")

# Collect TensorFlow dependencies
datas, binaries, hiddenimports = collect_all('tensorflow')

# Include model and helper files
datas += [
    (model_path, '.'),            # siamesemodelv2.h5
    (model_py, '.'),              # model.py
    (data_prep_py, '.'),          # data_preprocessing.py
]

# ---------------- ANALYSIS ---------------- #
a = Analysis(
    [app_script],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ---------------- PYZ ---------------- #
pyz = PYZ(a.pure)

# ---------------- EXECUTABLE ---------------- #
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceID',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_path,
)

# ---------------- COLLECT ---------------- #
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceID'
)
