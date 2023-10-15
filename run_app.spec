# -*- mode: python ; coding: utf-8 -*-


datas = [('./test_env', './env')]
datas += [('./.streamlit', './.streamlit')]
datas += [('./model.onnx', './model.onnx')]
datas += [('./coco_test.jpg', '././coco_test.jpg')]

datas += [('./test_env/lib/python3.10/site-packages/typing_extensions.py', './typing_extensions.py')]
datas += [('./test_env/lib/python3.10/site-packages/coloredlogs', './coloredlogs')]
datas += [('./test_env/lib/python3.10/site-packages/flatbuffers', './flatbuffers')]
datas += [('./test_env/lib/python3.10/site-packages/humanfriendly', './humanfriendly')]
datas += [('./test_env/lib/python3.10/site-packages/mpmath', './mpmath')]
datas += [('./test_env/lib/python3.10/site-packages/onnxruntime', './onnxruntime')]
datas += [('./test_env/lib/python3.10/site-packages/onnx', './onnx')]
datas += [('./test_env/lib/python3.10/site-packages/sympy', './sympy')]

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('main.py', '.')] + datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='run_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
