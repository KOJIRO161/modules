import sys
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import scipy as sp

import tkinter as tk
from tkinter import ttk
import json
import os

class peakfit:
    # 最終更新日:2024/11/06
    # 2024/11/06
    # フィッティング画像の出力方法を変えました。

    # 2024/10/31
    # 初期値の推定方法を変えました。
    
    def __init__(self,
                 theta: np.ndarray,
                 intensity: np.ndarray,
                 err: np.ndarray = None,
                 **kwargs) -> None:
        
        self.logger = kwargs.get("logger", None)
        
        if self.logger:
            self.logger.debug("[Start processing]")
        
        self._theta = theta
        self._err = err
        self._intensity = intensity

        self._peakrange = (self._theta[0,0], self._theta[-1,-1])
        self._viewrange = self._peakrange

        self.dataname = None

        self.set_data()

        return
    
    def set_dataname(self,dataname):
        self.dataname = dataname
        return
    
    @property
    def peakrange(self) -> tuple:
        return self._peakrange
    
    @peakrange.setter
    def peakrange(self, value: tuple):
        if self.logger:
            self.logger.info("Set peak range ({}, {})".format(value[0], value[1]))
        self._peakrange = value
        self.set_data()
        return
    
    @property
    def viewrange(self) -> tuple:
        return self._viewrange
    
    @viewrange.setter
    def viewrange(self, value: tuple):
        if self.logger:
            self.logger.info("Set view range ({}, {})".format(value[0], value[1]))
        self._viewrange = value
        self.set_data()
        return
    
    def set_data(self):
        self._theta_array = self._theta[(self._theta > self._peakrange[0]) * (self._theta < self._peakrange[1])].reshape((self._intensity.shape[0],-1))[0]
        self._plotdata = self._intensity[(self._theta > self._peakrange[0]) * (self._theta < self._peakrange[1])].reshape((self._intensity.shape[0],-1))
        self._errdata = self._err[(self._theta > self._peakrange[0]) * (self._theta < self._peakrange[1])].reshape((self._intensity.shape[0],-1))

        
        self._theta_array_view = self._theta[(self._theta > self._viewrange[0]) * (self._theta < self._viewrange[1])].reshape((self._intensity.shape[0],-1))[0]
        self._plotdata_view = self._intensity[(self._theta > self._viewrange[0]) * (self._theta < self._viewrange[1])].reshape((self._intensity.shape[0],-1))
        self._errdata_view = self._err[(self._theta > self._viewrange[0]) * (self._theta < self._viewrange[1])].reshape((self._intensity.shape[0],-1))
        return

    @property
    def plotdata(self) -> np.ndarray:
        return self._plotdata

    def pick_max(self) -> np.ndarray:
        if self.logger:
            self.logger.debug("peakfit.pick_max()")
        return self._theta_array[np.argmax(self._plotdata, axis=1)]
    
    def plot_data(self, *data) -> im.Image:
        if self.logger:
            self.logger.debug("peakfit.plot_data()")

        (fig, ax) = plt.subplots()
        fig.suptitle(self.dataname + ": peak range = {}".format(self._viewrange), fontsize = 12)
        ax.set_xlabel("Theta [degree]",fontsize = 10)
        ax.set_ylabel("Frame number",fontsize = 10)

        ax.imshow(
            self._plotdata_view, #引数は2次元配列か3次元配列。3次元配列の場合はRGB/RGBA方式の配列を渡す。
            vmax = self._plotdata_view.max(), #プロットの最大値 #不具合が起こる可能性があるので、書いておくのが望ましい。
            vmin = self._plotdata_view.min(), #プロットの最小値
            cmap = 'gray', #色の種類 #"jet","gray"など様々ある。 #"jet_r"のように色の名前の後に"_r"をつけると、反転した色が表示される。
            origin = "upper", #左上を原点とするか、左下を原点とするか #デフォルトは左上("upper") #右下にしたい場合は"lower"
            extent = (self._viewrange[0],self._viewrange[1],self._intensity.shape[0],0), #軸の設定 #(x軸の最小値、x軸の最大値、y軸の最小値、y軸の最大値)の順番で設定する。
            aspect = "auto", #プロットのアスペクト比 #数字の他、"equal"や"auto"を選べる。
            alpha = 1, #不透明度
        )

        for i in range(len(data)):
            ax.errorbar(
                data[i], #横軸成分
                np.arange(0,self._intensity.shape[0]), #縦軸成分
                # xerr = peak1[1], #横軸誤差 #2行配列可能(左右で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
                # yerr = [data[3],data[4]], #縦軸誤差 #2行配列可能(上下で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
                alpha = 1, #不透明度 #デフォルトは1
                # color = "0", #線の色 #`c`に省略可 #指定方法は様々あるので、各自ネットで調べてください。
                # linestyle = "solid", #線のスタイル #`ls`に省略可 #デフォルトは"solid" #" ","dashed","dotted","dashdot"等がある。
                # linewidth = 1, #線幅 #`lw`に省略可 #デフォルトは1ピクセル
                label = "Errorbar1", #ラベル #`ax.legend()`と併用する必要がある（後述）。
                fmt = "o", #マーカーの種類 #"s","o","D","d","h","H","^","v",">","<"等がある。 #指定しなければ描かれない。
                markersize = 1, #マーカーのサイズ #デフォルトは6?
                markeredgecolor = list(matplotlib.colors.TABLEAU_COLORS.keys())[i], #マーカーの枠色 #`mec`に省略可
                markeredgewidth = 1, #マーカーの線幅 #`mew`に省略可
                markerfacecolor = list(matplotlib.colors.TABLEAU_COLORS.keys())[i], #マーカーの塗色 #`mfc`に省略可
                capsize = 0, #エラーバーのキャップのサイズ デフォルトは0?
                capthick = 1, #エラーバーのキャップの太さ
                elinewidth = 1, #エラーバーの太さ
                ecolor = list(matplotlib.colors.TABLEAU_COLORS.keys())[i], #エラーバーの色
                clip_on = True, #枠からはみ出した部分を切り落とすかどうか。 #デフォルトはTrue
            )
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

    def fit_Vigot_func_solo(self,
                            frame: int,
                            nop: int = 1,
                            image: bool = False,
                            log: bool = False,
                            fitting: bool = True,
                            ) -> tuple:
        
        if bool(self.logger) * log:
            self.logger.info("peakfit.fit_Vigot_func_solo()")

        d = self._plotdata[frame].copy()
        e = self._errdata[frame].copy()
        
        if image:
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

            ax.scatter(self._theta_array_view,
                       self._plotdata_view[frame].copy(),
                       c="1", edgecolors="0")
            ax.scatter(self._theta_array,d,c="0")

            if not fitting:
                fig.canvas.draw()
                img = im.frombytes(
                        "RGBA",
                        fig.canvas.get_width_height(),
                        fig.canvas.buffer_rgba(),
                        "raw",
                    )
                plt.close()
                return (None, img)
            
        if not fitting:
            return None
        
        methods = ["trf", "dogbox"]

        bounds_up = [np.inf, self._theta_array[-1], np.inf, 1] * nop
        bounds_down = [0, self._theta_array[0], 0, 0] * nop
        bounds = (
            tuple([-np.inf, -np.inf] + bounds_down),
            tuple([np.inf, np.inf] + bounds_up)
        )
        
        _a0 = (d[-1]-d[0])/(self._theta_array[-1]-self._theta_array[0])
        _b0 = d[0]-_a0*self._theta_array[0]
        initparams = [
            _a0,
            _b0
        ]
        for j in range(nop):
            initparams += [
                (d.max()-d[0])/nop,
                self._theta_array[(d-_a0*self._theta_array - _b0).argmax()],
                (self._theta_array[-1]-self._theta_array[0])/5,
                0.5
            ]

        for method in methods:
            try:
                func = Vigot_func
                (popt, pcov) = sp.optimize.curve_fit(func,
                                                    self._theta_array,
                                                    d,
                                                    p0 = initparams,
                                                    maxfev = 4000,
                                                    sigma = e,
                                                    bounds=bounds,
                                                    method = method,
                                                    )
            except RuntimeError as errorcontent:
                if method == methods[-1]:
                    return errorcontent
                pass
            else:
                res = d - func(self._theta_array,*popt)
                rss = np.sum(np.square(res)) # residual sum of squares
                tss = np.sum(np.square(d-np.mean(d))) # total sum of squares = tss
                r_squared = 1 - (rss / tss)
                
                break

        if image:
            x = np.linspace(self._theta_array[0],self._theta_array[-1],200)
            y_init = Vigot_func(x,*initparams
                                )
            ax.plot(x,y_init,
                    c = "tab:blue")

            y_fit = Vigot_func(x,*popt
                                )
            ax.plot(x,y_fit,
                    c = "tab:orange")

            if (nop-1):
                for i in range(nop):
                    y_fit = Vigot_func(x,
                                    *(list(popt[:2]) + list(popt[4*i+2:4*i+6]))
                                        )
                    ax.plot(x,y_fit,
                            c = "tab:red")
            ax.text(
                x = 0.99,
                y = 0.99,
                s = "r2 = {:.3f}%".format(r_squared*100),
                ha = "right",
                va = "top",
                transform = ax.transAxes,
            )
                    
            fig.canvas.draw()
            img = im.frombytes(
                    "RGBA",
                    fig.canvas.get_width_height(),
                    fig.canvas.buffer_rgba(),
                    "raw",
                )
            plt.close()
            return ([popt, np.diag(pcov), r_squared], img)
        else:
            return ([popt, np.diag(pcov), r_squared], None)

    def fit_Vigot_func(self,
                       nop: int = 1,
                       video: bool = False,
                       video_filename: str = os.path.dirname(__file__) + "output.mp4",
                       ) -> tuple:

        if self.logger:
            self.logger.debug("peakfit.fit_Vigot_func()")

        l_popt = []
        l_pcov = []
        l_r2 = []

        if video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            tape = cv2.VideoWriter(filename=video_filename,
                                   fourcc=fourcc,
                                   fps=4,
                                   frameSize=[600,480],
                                   )

        loopcycle = tqdm(range(self._plotdata.shape[0])) if self.logger else range(self._plotdata.shape[0])
        for i in loopcycle:
            output = self.fit_Vigot_func_solo(frame = i,
                                              nop = nop,
                                              image = video,
                                              fitting = True)
            if type(output) == RuntimeError:
                try:
                    tape.release()
                    if self.logger:
                        self.logger.info("Save video: "+os.path.abspath(video_filename))
                except:
                    pass
                raise RuntimeError
            elif type(output) == tuple:
                (_, img) = output
                popt = _[0]
                pcov = _[1]
                r_squared = _[2]

                l_popt.append([popt])
                l_pcov.append([pcov])
                l_r2.append(r_squared)
            else:
                raise NameError()

            if video:
                tape.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR))
            continue

        try:
            tape.release()
            if self.logger:
                self.logger.info("Save video: "+os.path.abspath(video_filename))
        except:
            pass

        popts = np.block(l_popt)
        pcovs = np.block(l_pcov)
        r2 = np.array(l_r2)

        return (popts, pcovs, r2)

def Vigot_func(x, ba, bb, *ps):

    value = ba*x + bb

    for i in range(len(ps)//4):
        amp = ps[4*i]
        mu = ps[4*i + 1]
        sigma = ps[4*i + 2]
        p = ps[4*i + 3]

        g = np.exp((-(x-mu)**2)/(2*sigma**2))
        l = sigma**2 / (((x-mu)**2) + sigma**2)
        value += amp*(g*p + l*(1-p))

    return value

class ConfigApp(tk.Tk):
    def __init__(self, json_file="config.json"):
        super().__init__()
        self.title("JSON同期設定 GUI")
        self.json_file = json_file
        
        # 初期設定のJSONデータ
        self.default_config = {
            "operation": "add",
            "value1": 10,
            "value2": 5
        }

        # JSONファイルが存在しなければデフォルト設定で作成
        if not os.path.exists(self.json_file):
            self.save_config(self.default_config)

        # 現在の設定を読み込む
        self.config = self.load_config()

        # GUIの構築
        self.create_widgets()
        self.update_result()  # 最初の結果を計算

    def load_config(self):
        """JSONファイルから設定を読み込む関数"""
        with open(self.json_file, "r") as f:
            return json.load(f)

    def save_config(self, config):
        """設定をJSONファイルに保存する関数"""
        with open(self.json_file, "w") as f:
            json.dump(config, f, indent=4)

    def update_config(self, key, value):
        """設定を更新してJSONファイルに保存し、結果をリアルタイムで更新する関数"""
        self.config[key] = value
        self.save_config(self.config)
        self.update_result()

    def create_widgets(self):
        """GUIのウィジェットを作成する関数"""

        # Operation選択ドロップダウン
        ttk.Label(self, text="Operation:").grid(row=0, column=0, padx=10, pady=5)
        self.operation_var = tk.StringVar(value=self.config["operation"])
        operation_dropdown = ttk.Combobox(
            self,
            textvariable=self.operation_var,
            values=["add", "subtract", "multiply", "divide"]
        )
        operation_dropdown.grid(row=0,
                                column=1,
                                padx=10,
                                pady=5)
        operation_dropdown.bind("<<ComboboxSelected>>",
                                lambda e: self.update_config(
                                    "operation", self.operation_var.get()
                                ))

        # Value1入力
        ttk.Label(self, text="Value 1:").grid(row=1,
                                                   column=0,
                                                   padx=10,
                                                   pady=5)
        self.value1_var = tk.IntVar(value=self.config["value1"])
        value1_entry = ttk.Entry(self, textvariable=self.value1_var)
        value1_entry.grid(row=1, column=1, padx=10, pady=5)
        value1_entry.bind("<FocusOut>",
                          lambda e: self.update_config(
                              "value1",
                              self.value1_var.get()
                          ))

        # Value2入力
        ttk.Label(self, text="Value 2:").grid(row=2, column=0, padx=10, pady=5)
        self.value2_var = tk.IntVar(value=self.config["value2"])
        value2_entry = ttk.Entry(self, textvariable=self.value2_var)
        value2_entry.grid(row=2, column=1, padx=10, pady=5)
        value2_entry.bind("<FocusOut>", lambda e: self.update_config("value2", self.value2_var.get()))

        # 結果表示ラベル
        self.result_label = ttk.Label(self, text="Result: ")
        self.result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def update_result(self):
        """設定に基づき演算結果を更新して表示する関数"""
        value1 = self.config["value1"]
        value2 = self.config["value2"]
        operation = self.config["operation"]

        try:
            if operation == "add":
                result = value1 + value2
            elif operation == "subtract":
                result = value1 - value2
            elif operation == "multiply":
                result = value1 * value2
            elif operation == "divide":
                result = value1 / value2 if value2 != 0 else "Error (Div by 0)"
            else:
                result = "Unknown operation"
        except Exception as e:
            result = f"Error: {e}"

        self.result_label.config(text=f"Result: {result}")

def run_App():
    app = ConfigApp()
    app.mainloop()
    return

class ButtonCreator(tk.Tk):
    def __init__(self):
        # 初期化
        super().__init__()
        self.title("Set configuration")

        # widgetを作成する
        self.create_widget()

        # 初期のボタン数を保持する変数
        self.button_count = 1

    def create_widget(self):

        # auto/step
        self.combo, self.cval = self.create_combobox()
        self.combo.pack()

        # ranges
        self.frame_range, self.dict_ranges = self.create_rangeEntries()
        self.frame_range.pack()
        
        self.frame_peaknumber, self.peaknumber = self.create_peaknumber()
        self.frame_peaknumber.pack()

        # ボタンを作成するボタン
        self.submit_button = tk.Button(self, text="ボタンを作成", command=self.create_buttons)
        self.submit_button.pack()

        # 1つボタンを増やすボタン
        self.add_button_button = tk.Button(self, text="ボタンを1つ増やす", command=self.add_button)
        self.add_button_button.pack()

        # ボタンを表示するフレーム
        self.frame = tk.Frame(self)
        self.frame.pack()
    
    def create_combobox(self):
        comboval = ["auto", "step"]
        cval = tk.StringVar()
        combo = ttk.Combobox(
            self,
            values = comboval,
            state = "readonly",
            textvariable=cval,
        )
        combo.set(comboval[0])
        return combo, cval

    def create_rangeEntries(self):

        frame_range = tk.Frame(self,
                            #    bd = 1,
                            #    relief = "solid",
                               )
        dict_ranges = dict()

        if True: # plotrange
            frame_plotrange = tk.Frame(frame_range)
            frame_plotrange.grid(column = 0, row = 0)

            title_plotrange = tk.Label(frame_plotrange,
                                    text = "plot range")
            title_plotrange.pack(
                side = "top",
                anchor = "w"
            )
            
            if True: # min
                plotmin = tk.Frame(frame_plotrange)
                plotmin.pack()

                label_plotmin = ttk.Label(plotmin,
                                        text = "min")
                label_plotmin.pack(side = "left")

                entry_plotmin = ttk.Entry(plotmin)
                entry_plotmin.pack(side = "left",
                                expand = True,
                                fill = "y")
                entry_plotmin.bind("<KeyRelease>", self.change_ranges)
                dict_ranges["plot min"] = entry_plotmin
                
            if True: # max
                plotmax = tk.Frame(frame_plotrange)
                plotmax.pack()

                label_plotmax = ttk.Label(plotmax,
                                        text = "max")
                label_plotmax.pack(side = "left")

                entry_plotmax = ttk.Entry(plotmax)
                entry_plotmax.pack(side = "left",
                                expand = True,
                                fill = "y")
                entry_plotmax.bind("<KeyRelease>", self.change_ranges)
                dict_ranges["plot max"] = entry_plotmax

        if True: # plotrange
            frame_fitrange = tk.Frame(frame_range)
            frame_fitrange.grid(column = 1, row = 0)

            title_fitrange = tk.Label(frame_fitrange,
                                    text = "fit range")
            title_fitrange.pack(
                side = "top",
                anchor = "w"
            )
            
            if True: # min
                fitmin = tk.Frame(frame_fitrange)
                fitmin.pack()

                label_fitmin = ttk.Label(fitmin,
                                        text = "min")
                label_fitmin.pack(side = "left")

                entry_fitmin = ttk.Entry(fitmin)
                entry_fitmin.pack(side = "left",
                                expand = True,
                                fill = "y")
                entry_fitmin.bind("<KeyRelease>", self.change_ranges)
                dict_ranges["fit min"] = entry_fitmin
                
            if True: # max
                fitmax = tk.Frame(frame_fitrange)
                fitmax.pack()

                label_fitmax = ttk.Label(fitmax,
                                        text = "max")
                label_fitmax.pack(side = "left")

                entry_fitmax = ttk.Entry(fitmax)
                entry_fitmax.pack(side = "left",
                                expand = True,
                                fill = "y")
                entry_fitmax.bind("<KeyRelease>", self.change_ranges)
                dict_ranges["fit max"] = entry_fitmax

        return frame_range, dict_ranges

    def create_peaknumber(self):
        # エントリーフィールドとラベルを作成
        frame_peaknumber = tk.Frame(self)

        entry_label = tk.Label(frame_peaknumber,
                                    text="number of peaks:")
        entry_label.pack(
            side = "left"
        )

        entry = tk.Entry(frame_peaknumber)
        entry.pack(side = "left",
                   fill = "y",
                   expand = True)
        entry.bind("<KeyRelease>", self.change_peaknumber)

        return frame_peaknumber, entry

    def create_buttons(self):
        # 入力された数を取得
        try:
            num_buttons = int(self.entry.get())
        except ValueError:
            return
        
        # 現在のボタンをクリア
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 新しいボタンを追加
        self.button_count = num_buttons
        for i in range(self.button_count):
            button = tk.Button(self.frame, text=f"ボタン {i+1}")
            button.pack()

    def add_button(self):
        # 現在のボタン数を1つ増やす
        self.button_count += 1
        
        # エントリーの中の数を更新
        self.entry.delete(0, tk.END)  # エントリーをクリア
        self.entry.insert(0, str(self.button_count))  # 新しい数を挿入

        button = tk.Button(self.frame, text=f"ボタン {self.button_count}")
        button.pack()

    def change_ranges(self, e = None):
        for key in self.dict_ranges:
            print(self.dict_ranges[key].get())

    def change_peaknumber(self, e = None):
        print(self.peaknumber.get())

class RealTimeGraphApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("リアルタイムグラフ")

        # エントリーフィールドとラベルを作成
        self.entry_label = tk.Label(self, text="グラフに表示するx軸の最大値:")
        self.entry_label.pack(pady=5)

        self.entry = tk.Entry(self)
        self.entry.pack(pady=5)
        self.entry.insert(0, "10")  # 初期値

        # entryの内容が変わるたびにグラフを更新
        self.entry.bind("<KeyRelease>", self.update_graph)

        # グラフを表示するウィンドウを作成
        self.graph_window = tk.Toplevel(self)
        self.graph_window.title("グラフ")

        # グラフの描画
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("y = x^2")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.canvas = FigureCanvasTkAgg(self.figure, self.graph_window)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.graph_window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_graph(self, event=None):
        # 入力されたx軸の最大値を取得
        try:
            max_x = int(self.entry.get())
        except ValueError:
            return  # 数値でない場合は何もしない

        # xの値を生成
        x = np.linspace(0, max_x, 100)
        y = x ** 2  # y = x^2の関数

        # 既存のグラフをクリアして新しいデータで更新
        self.ax.clear()
        self.ax.plot(x, y, label="y = x^2")
        self.ax.set_title("y = x^2")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()

        # グラフを再描画
        self.canvas.draw()

    def on_closing(self):
        self.quit()

if __name__ == "__main__":

    app = ButtonCreator()
    app.mainloop()