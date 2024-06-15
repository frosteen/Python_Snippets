import os, sys

# When converting with PyInstaller, add argument --add-data of relative_path
# to compress .exe file to smaller size, add argument --upx-dir of upx directory (upx-3.96-win32/upx-3.96-win64)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
