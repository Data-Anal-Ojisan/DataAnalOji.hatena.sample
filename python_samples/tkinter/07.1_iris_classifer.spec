# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['07.1_iris_classifer.py'],
             pathex=['C:\\Users\\rhira\\Documents\\GitHub\\DataAnalOji.hatena.sample\\python_samples\\tkinter'],
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn','six', 'sklearn','scipy.special.cython_special', 'sklearn.ensemble','sklearn.neighbors._typedefs','sklearn.utils._cython_blas', 'sklearn.neighbors._quad_tree'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          name='07.1_iris_classifer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
