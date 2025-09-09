"""概要
    jupyter notebookを使いやすくするためのmethod集
"""

import sys
sys.dont_write_bytecode = True # __pycache__を生まない

# 基本モジュール
import os
from datetime import datetime
import subprocess

# ファイル操作等
import pickle

# tkinter
from tkinter import filedialog, messagebox, Tk
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1) # tkinterの解像度を上げる

"""変更履歴
* 2025/09/09: 作成
"""

def save_pickle(data: dict, initialdir = None):

    """変更履歴
    * 2025/09/09: notebookから移植
    * 2025/09/08: initialdirの追加
    * 2025/08/05: defaultextensitionを追加、initialdirの削除
    * 2025/08/01: 作成ののち変更
    """

    # 現在時刻の取得
    dt = datetime.now()
    initfilename = dt.strftime('%Y%m%d%H%M%S%f') + ".pkl"

    # tkinterでファイルダイアログを開く
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    filename = filedialog.asksaveasfilename(
        parent = window,
        filetypes = [
            ("pkl", "*.pkl"),
        ],
        initialfile=initfilename,
        initialdir=initialdir,
        defaultextension=".pkl",
    )

    # filenameがセットされたら保存する
    if not (filename == ""):
        with open(filename, mode = "wb") as f:
            pickle.dump(
                obj = data,
                file = f
            )
        return filename
    else:
        return None
    
def load_pickle(initialdir = None):

    """変更履歴
    * 2025/09/09: notebookから移植
    * 2025/09/08: initialdirの追加
    * 2025/08/04: dataの上書きができていませんでした
    * 2025/07/25: 作成
    """

    # tkinterでファイルダイアログを開く
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    filename = filedialog.askopenfilename(
        parent = window,
        filetypes = [
            ("pkl", "*.pkl"),
        ],
        initialdir=initialdir,
    )
    if not (filename == ""):
        with open(filename, mode = "rb") as f:
            data = pickle.load(
                file = f
            )

        return data, filename
    else:
        return None
    
def export_html(ipynbfile, initialdir = None):

    """変更履歴
    * 2025/09/09: save_pickleを参考に書き換え
    * 2025/09/09: notebookから移植
    * 2025/08/01: 変更履歴作成
    """

    # 現在時刻の取得
    dt = datetime.now()
    initfilename = dt.strftime('%Y%m%d%H%M%S%f') + ".pkl"

    # tkinterでファイルダイアログを開く
    window = Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    filename = filedialog.asksaveasfilename(
        parent = window,
        filetypes = [
            ("html", "*.html"),
            ("md", "*.md")
        ],
        initialfile=initfilename,
        initialdir=initialdir,
        defaultextension=".html",
    )

    # filenameがセットされたら保存する
    if not (filename == ""):
        extension = os.path.splitext(filename)[1]
        if extension == ".html":
            subprocess.run(["jupyter", "nbconvert", "--to", "html", ipynbfile, "--output", filename])
        elif extension == ".md":
            subprocess.run(["jupyter", "nbconvert", "--to", "markdown", ipynbfile, "--output", filename])
        return filename
    else:
        return None

