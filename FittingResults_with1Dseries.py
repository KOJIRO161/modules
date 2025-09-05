import sys
sys.dont_write_bytecode = True # cacheを出さない

import os
# from datetime import datetime
from pprint import pprint
import logging
# import pickle
# import struct
from tqdm import tqdm
import h5py
# import threading
import json

# tkinter
from tkinter import filedialog, messagebox, Tk

# データ分析ツール
import pandas as pd
import numpy as np
import scipy as sp
# import math
# from sklearn.linear_model import LinearRegression

# グラフ等作成用
import matplotlib
import matplotlib.pyplot as plt         # 図の作成用
from PIL import Image as im
# import cv2
from IPython.display import display, HTML, clear_output, update_display

# 自作モジュール
sys.path.append(r"C:\Users\okaza\pythonenvs")
from modules.Mytools.Tools import print_fileinfo, h5_tree
import modules.Mytools.Settings
import modules.fitXRD as fx
from modules.peakfit import peakfit, pseudoVoigt

# ログ管理ツール作成

# chche directoryの設定
cachedir = os.path.abspath(os.path.dirname(__file__) + "/.cache")
os.makedirs(cachedir, exist_ok=True)

# loggerの作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = "%(levelname)-9s  %(asctime)s [%(filename)s:%(lineno)d] %(message)s"

# Streamハンドラクラスを作成
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter(format))
logger.addHandler(sh)

args = sys.argv
if args[1].strip() == "-SetConfig":
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    configfile = filedialog.asksaveasfilename(
        title = "configファイルの保存",
        filetypes=[("JSON", ".json")],
        defaultextension=".json"
    )

def get_filelist():
    
    # 読込ディレクトリを設定
    _filedir = r"E:\okaza\YNFeH04\1st\1D"

    # データ名を設定(sortするときに必要になります。)
    _dataname = "YNFeH04_1"
    header = "YNFeH04_1"
    footer = ""

    if True: # [ Main ]
        logger.debug("[Session processing]: get_filelist")

        # global filedir
        filedir = _filedir
        logger.info("[Add variables]: filedir")

        # # ファイルリストを初期化
        # global filelist
        filelist = []
        logger.info("[Add variables]: filelist")

        # global dataname
        dataname = _dataname
        logger.info("[Add variables]: dataname")

        # ディレクトリ内の全ファイル名を検索
        for f in tqdm(os.listdir(filedir)):
            # データ名が含まれるかつcsvであるファイルを取得
            if not dataname in f:
                continue
            if not "csv" in f:
                continue
            filelist.append(f)
            continue
        # frame numberでソート
        filelist.sort(
            key=lambda x: int(
                os.path.splitext(x)[0].replace(header+"_", "").replace("_"+footer, "")
            )
        )

        pass
    return filedir, dataname, filelist

def read_allcsv(filedir, filelist):
    logger.debug("[Session processing]: read_allcsv")

    tht_list = [] # theta
    intst_list = [] # intensity

    # ファイルリスト内の全ファイルを読み込む
    for f in tqdm(filelist):
        df = pd.read_csv(
            filedir + "/" + f,
            header = None,
        )
        tht_list.append(df.values[:,0])
        intst_list.append(df.values[:,1])
        continue

    # 各データを保存
    # global theta # [degree]
    theta = np.array(tht_list)
    logger.info("[Add variables]: theta")

    # global intensity # Intensity
    intensity = np.array(intst_list)
    logger.info("[Add variables]: intensty")

    display(HTML("<font size = '3'>Theta [degree]</font>"))
    display(pd.DataFrame(theta[:5,:5]))
    pd.DataFrame(theta).info()
    print("-"*30)

    display(HTML("<font size = '3'>Intensity</font>"))
    display(pd.DataFrame(intensity[:5,:5]))
    pd.DataFrame(intensity).info()
    print("-"*30)
    
    return theta, intensity

class peakfit_withImage(peakfit):
    
    def __init__(self,
                 theta: np.ndarray,
                 intensity: np.ndarray,
                 logger = None,
                 ) -> None:
        
        super().__init__(theta = theta, intensity = intensity)

        return
    
    def imshow_data(self, data: dict) -> im.Image:

        (fig, ax) = plt.subplots()
        ax.set_xlabel("Theta [degree]",fontsize = 10)
        ax.set_ylabel("Frame number",fontsize = 10)
        
        fig.set_size_inches((5,5))
        fig.set_dpi(300)

        ax.imshow(
            self.intensity, #引数は2次元配列か3次元配列。3次元配列の場合はRGB/RGBA方式の配列を渡す。
            vmax = self.intensity.max(), #プロットの最大値 #不具合が起こる可能性があるので、書いておくのが望ましい。
            vmin = self.intensity.min(), #プロットの最小値
            cmap = 'gray', #色の種類 #"jet","gray"など様々ある。 #"jet_r"のように色の名前の後に"_r"をつけると、反転した色が表示される。
            origin = "upper", #左上を原点とするか、左下を原点とするか #デフォルトは左上("upper") #右下にしたい場合は"lower"
            extent = (self.theta[0],self.theta[-1],self.intensity.shape[0],0), #軸の設定 #(x軸の最小値、x軸の最大値、y軸の最小値、y軸の最大値)の順番で設定する。
            aspect = "auto", #プロットのアスペクト比 #数字の他、"equal"や"auto"を選べる。
            alpha = 1, #不透明度
        )

        for i, key in enumerate(data):
            x = data[key]["data"]
            color = data[key].get("color", list(matplotlib.colors.TABLEAU_COLORS.keys())[i])
            ax.errorbar(
                x, #横軸成分
                np.arange(0,self.intensity.shape[0]), #縦軸成分
                # xerr = peak1[1], #横軸誤差 #2行配列可能(左右で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
                # yerr = [data[3],data[4]], #縦軸誤差 #2行配列可能(上下で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
                alpha = 1, #不透明度 #デフォルトは1
                # color = "0", #線の色 #`c`に省略可 #指定方法は様々あるので、各自ネットで調べてください。
                # linestyle = "solid", #線のスタイル #`ls`に省略可 #デフォルトは"solid" #" ","dashed","dotted","dashdot"等がある。
                # linewidth = 1, #線幅 #`lw`に省略可 #デフォルトは1ピクセル
                label = key, #ラベル #`ax.legend()`と併用する必要がある（後述）。
                fmt = "o", #マーカーの種類 #"s","o","D","d","h","H","^","v",">","<"等がある。 #指定しなければ描かれない。
                markersize = 1, #マーカーのサイズ #デフォルトは6?
                markeredgecolor = color, #マーカーの枠色 #`mec`に省略可
                markeredgewidth = 1, #マーカーの線幅 #`mew`に省略可
                markerfacecolor = color, #マーカーの塗色 #`mfc`に省略可
                # capsize = 0, #エラーバーのキャップのサイズ デフォルトは0?
                # capthick = 1, #エラーバーのキャップの太さ
                # elinewidth = 1, #エラーバーの太さ
                # ecolor = color, #エラーバーの色
                clip_on = True, #枠からはみ出した部分を切り落とすかどうか。 #デフォルトはTrue
            )
        
        if data:
            ax.legend(fontsize = 10)
        fig.subplots_adjust(
            left = 0.125, # default = 0.125
            bottom = 0.1, # default = 0.1
            right = 0.9, # default = 0.9
            top = 0.9, # default = 0.9
            hspace = 0.2, # default = 0.2
            wspace = 0.2, # default = 0.2
        )
        fig.canvas.draw()
        img = im.frombytes(
                "RGBA",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
                "raw",
            )
        plt.close()
        return img
    
    def plot_data(self, data: dict):

        frame = data["frame"]

        (fig, ax) = plt.subplots()
        # figの設定
        fig.set_size_inches((5,4))
        fig.set_dpi(120)
        fig.subplots_adjust(
            left = 0.125, # default = 0.125
            bottom = 0.1, # default = 0.1
            right = 0.875, # default = 0.9
            top = 0.9, # default = 0.9
            hspace = 0.2, # default = 0.2
            wspace = 0.2, # default = 0.2
        )

        # axの設定
        ax.set_xlabel("Theta [degree]", fontsize = 10)
        ax.set_ylabel("Intensity", fontsize = 10)
        ax.set_title("frame number = {:>4}".format(frame), fontsize = 12)

        ax.scatter(self.theta.copy(),
                   self.intensity[frame].copy(),
                   c="1",
                   edgecolors="0")
        
        x = np.linspace(self.theta[0],self.theta[-1],100)
        for label in data["peaks"].keys():
            y = pseudoVoigt(x,*data["peaks"][label]["params"])
            ax.plot(
                x,
                y,
                lw = 1,
                c = data["peaks"][label]["color"],
                label = label
            )
        if data["peaks"]:
            ax.legend(fontsize = 10)

        fig.canvas.draw()
        img = im.frombytes(
                "RGBA",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
                "raw",
            )
        plt.close()
        return img

if __name__ == "__main__":
    filedir, dataname, filelist = get_filelist()

    # 確認用出力
    print(np.array(filelist))

    if not input("Continue...? "):
        sys.exit()

    theta, intensity = read_allcsv(filedir, filelist)

    if not input("Continue...? "):
        sys.exit()

    json_files = [
        os.path.join(os.path.dirname(__file__), f) for f in os.listdir(os.path.dirname(__file__)) if (f.endswith(".json"))
    ]
    print(np.array(json_files))

    if not input("Continue...? "):
        sys.exit()
    
    mask = (theta[0] > 9) * (theta[0] < 15)
    tth = theta[0][mask]
    ii = intensity.T[mask].T

    peak = peakfit_withImage(theta = tth, intensity = ii)
    data = dict()
    colors = matplotlib.colors.TABLEAU_COLORS
    for i, filename in enumerate(json_files):
        with open(filename, mode = "r") as f:
            results = json.load(f)
        fitresult = peak.pick_max()
        for key in results["results"].keys():
            fitresult[int(key[5:])] = results["results"][key]["params"]["Peak1"]["params"]["mu"]
        data[os.path.splitext(os.path.basename(filename))[0]] = {
            "data": fitresult,
            "color": colors[list(colors.keys())[i]]
        }
        
    img = peak.imshow_data(data = data)
    imgfilename = cachedir + "/"+ os.path.basename(os.path.splitext(__file__)[0]) +"_tmp.png"
    img.save(imgfilename)
    logger.info("[Save img]: " + os.path.abspath(imgfilename))
