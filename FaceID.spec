# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
import tensorflow as tf

# Collect all TensorFlow files
datas_tf, binaries_tf, hiddenimports_tf = collect_all('tensorflow')

a = Analysis(
    ['Authentication/app.py'],
    pathex=[],
    binaries=binaries_tf,  # Include TensorFlow DLLs
    datas=[
        ('Authentication/siamesemodelv2_keras', 'siamesemodelv2_keras'),
        ('Authentication/model.py', '.'),
        ('Authentication/util.py', '.'),
    ] + datas_tf,  # Include TensorFlow data files
    hiddenimports=hiddenimports_tf,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

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
    console=True, #console included for process monitoring  
    icon='Authentication/icon.ico',  
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceID',
)
