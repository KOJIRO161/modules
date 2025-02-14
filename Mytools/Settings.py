import matplotlib.pyplot as plt
import pandas as pd

if True: # グラフの出力設定
    plt.rcParams['font.family'] ='Arial' # 使用するフォント Arial or DejaVu Serif
    plt.rcParams['mathtext.fontset'] = 'cm'     # 数式用のフォント
    plt.rcParams['xtick.direction'] = 'in'      # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'      # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0     # x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0     # y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 8               # フォントの大きさ
    plt.rcParams['axes.linewidth'] = 0.8        # 軸の線幅edge linewidth。囲みの太さ
    plt.rcParams['pdf.fonttype'] = 42           # フォントを埋め込むための設定(よくわからん)
    plt.rcParams['ps.fonttype'] = 42            # これも上と同じなのかな(よくわからん)

# if True: # pandasの出力設定
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


### cheatsheet
# save a binary file
"""
* numpy
    np.save(savedir + filename, np_a)
    np_a = np.load(savedir + filename + npy)

* pandas
    df.to_pickle(savedir + filename + pkl)
    df = pd.read_pickle(savedir + filename + pkl)

* others
    with open(savedir + filename + txt, 'wb') as f:
        pickle.dump(l_a, f)
    with open(savedir + filename + txt, 'rb') as f:
        l_a = pickle.load(f)
"""

# make directory:
"""
if True:
    os.makedirs(_directoryname, exist_ok = True)
"""

# show file information
"""
if True:
    file_stat = os.stat(filename)
    print("")
    print('File name: {}'.format(filename))
    print('File size: {:,} bites'.format(file_stat.st_size))
    print('Last update time: {}'.format(datetime.fromtimestamp(file_stat.st_mtime)))
    print("")
    return file_stat
"""

# output dataimage

# make matplotlib imshow
"""
Z = data
colormap1 = ax.imshow(
    Z, #引数は2次元配列か3次元配列。3次元配列の場合はRGB/RGBA方式の配列を渡す。
    vmax = Z.max(), #プロットの最大値 #不具合が起こる可能性があるので、書いておくのが望ましい。
    vmin = Z.min(), #プロットの最小値
    cmap = 'jet', #色の種類 #"jet","gray"など様々ある。 #"jet_r"のように色の名前の後に"_r"をつけると、反転した色が表示される。
    origin = "upper", #左上を原点とするか、左下を原点とするか #デフォルトは左上("upper") #右下にしたい場合は"lower"
    extent = (x.min(),x.max(),y.min(),y.max()), #軸の設定 #(x軸の最小値、x軸の最大値、y軸の最小値、y軸の最大値)の順番で設定する。
    aspect = "equal", #プロットのアスペクト比 #数字の他、"equal"や"auto"を選べる。
    alpha = 1, #不透明度
)
colorbar = fig.colorbar( #colorbarの表示 #難しい
    mappable = colormap1, #カラーバーをつけるカラーマップ
    ax = ax, #カラーバーを属させる軸
)
"""

# make matplotlib errorbar
"""
ax.errorbar(
    data[0], #横軸成分
    data[1], #縦軸成分
    xerr = data[2], #横軸誤差 #2行配列可能(左右で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
    yerr = [data[3],data[4]], #縦軸誤差 #2行配列可能(上下で異なるエラーをつけられる。) #書かなければエラーバーが表示されない。
    alpha = 1, #不透明度 #デフォルトは1
    color = "0", #線の色 #`c`に省略可 #指定方法は様々あるので、各自ネットで調べてください。
    linestyle = "solid", #線のスタイル #`ls`に省略可 #デフォルトは"solid" #" ","dashed","dotted","dashdot"等がある。
    linewidth = 1, #線幅 #`lw`に省略可 #デフォルトは1ピクセル
    label = "Errorbar1", #ラベル #`ax.legend()`と併用する必要がある（後述）。
    fmt = "o", #マーカーの種類 #"s","o","D","d","h","H","^","v",">","<"等がある。 #指定しなければ描かれない。
    markersize = 6, #マーカーのサイズ #デフォルトは6?
    markeredgecolor = '0', #マーカーの枠色 #`mec`に省略可
    markeredgewidth = 1, #マーカーの線幅 #`mew`に省略可
    markerfacecolor = "0", #マーカーの塗色 #`mfc`に省略可
    capsize = 0, #エラーバーのキャップのサイズ デフォルトは0?
    capthick = 1, #エラーバーのキャップの太さ
    elinewidth = 1, #エラーバーの太さ
    ecolor = "0", #エラーバーの色
    clip_on = True, #枠からはみ出した部分を切り落とすかどうか。 #デフォルトはTrue
)
"""

# save matplotlib figure
"""
if True: #余白の設定
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
graphname = ""
# fig.savefig(savedir + graphname + ".pdf")
plt.close()
display(img)
"""