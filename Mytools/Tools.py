import sys
sys.dont_write_bytecode = True

import os
# import matplotlib.pyplot as plt
# import matplotlib.figure as figure
# import matplotlib
# import pickle
import pandas as pd
import numpy as np
# import scipy as sp
from datetime import datetime
# from PIL import Image as im
# from IPython.display import display
import struct
import h5py

# ファイル情報を出力する関数
def print_fileinfo(filename):
    file_stat = os.stat(filename)
    print("")
    print('File name: {}'.format(os.path.abspath(filename)))
    print('File size: {:,} bites'.format(file_stat.st_size))
    print('Last update time: {}'.format(datetime.fromtimestamp(file_stat.st_mtime)))
    print("")
    return file_stat

# hdfファイルを階層表示する関数
def h5_tree(val, layer=0, pre='', lim=5):
    if not layer:
        print(val)
    items = len(val)
    for key, val in val.items():
        items -= 1
        is_last = (items == 0)
        branch = '└── ' if is_last else '├── '

        if isinstance(val, h5py.Group):
            print(pre + branch + key)
            if not (layer+1 == lim):
                h5_tree(val, layer+1, pre + ('    ' if is_last else '│   '), lim)
        elif isinstance(val, h5py.Dataset):
            try:
                print(pre + branch + key + ' ({}, {})'.format(val.shape, val.dtype))
            except TypeError:
                print(pre + branch + key + ' (scalar)')
        else:
            print(pre + branch + key + ' (unknown)')
    return

# dictを階層表示する関数
def dict_tree(d, indent="", last='updown'):
    """辞書（dict）をツリー状にビジュアルで表示する関数"""
    if isinstance(d, dict):
        for i, (key, value) in enumerate(d.items()):
            is_last = (i == len(d) - 1)  # 最後の要素かどうか
            connector = '└── ' if is_last else '├── '
            print("\n" + indent + connector + str(key), end = "")  # キーを表示
            new_indent = indent + ('    ' if is_last else '│   ')  # 次のレベルのインデント
            dict_tree(value, new_indent, 'up' if is_last else 'down')  # 再帰呼び出し
    else:
        if isinstance(d, np.ndarray):
            print(" " +  str(type(d)) + " [shape = {}]".format(d.shape), end = "")  # 辞書でない場合、値を表示
        else:
            print(" " +  str(type(d)), end = "")  # 辞書でない場合、値を表示

# hisファイルを読み込んで、np.ndarrayにする関数
def his2array(filename):
    with open(filename, 'rb') as file:
        data = file.read()
    
    # Define the keys for the header information
    keylist = [
        "ID", "headerSize", "headerVersion", "fileSize", "imageHeaderSize",
        "ULX", "ULY", "BRX", "BRY", "numberOfFrame", "correction",
        "frameTimeInMicroseconds", "frameTimeInMilliseconds"
    ]

    # Unpack the header data based on the expected structure
    info = dict(
        zip(
            keylist,
            struct.unpack('<HHHLHHHHHHHdH', data[:34])
        )
    )

    # Calculate the image dimensions
    imageWidth = info["BRX"] - info["ULX"] + 1
    imageHeight = info["BRY"] - info["ULY"] + 1

    # Calculate the starting position of the image data
    start_position = info["headerSize"] + info["imageHeaderSize"]

    # Extract all image data as a numpy array
    image_data = np.frombuffer(data[start_position:], dtype=np.uint16)

    # Reshape the image data to separate frames
    sequential_image_intensities = image_data.reshape((info["numberOfFrame"], imageHeight, imageWidth))

    # Select the first frame's intensities for further processing
    img = sequential_image_intensities[0]
    return img

if __name__ == "__main__":

    print("OK")