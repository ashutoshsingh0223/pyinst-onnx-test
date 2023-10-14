# -*- mode: python ; coding: utf-8 -*-

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata


block_cipher = None


datas = [("/home/AD.IGD.FRAUNHOFER.DE/sashutosh/app/env/lib/python3.10/site-packages/streamlit/runtime", "./streamlit/runtime")]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
datas += [('./.streamlit', './.streamlit')]
datas += [('./env/lib/python3.10/site-packages', '.')]


a = Analysis(['run_app.py'],
             pathex=[],
             binaries=[],
             datas=[('main.py', '.')] + datas,
             hiddenimports=[],
             hookspath=['./env/lib/python3.10/site-packages', './hooks'],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['_bootlocale'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
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
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
