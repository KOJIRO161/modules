import sys
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp

import tkinter as tk
from tkinter import ttk
import json
import os
from pprint import pprint

class peakfit:    
    def __init__(self,
                 theta: np.ndarray,
                 intensity: np.ndarray,
                 ) -> None:
        """peakをフィットする関数です"""
        self.theta = theta # 1次元
        self.intensity = intensity # shape = (frame, theta)
        return

    def pick_max(self):
        value = self.theta[np.argmax(self.intensity, axis=1)]
        return value
    
    def fit_Vigot_func(self,
                       frame: int,
                       nop: int = 1,
                       ) -> tuple:

        d = self.intensity[frame].copy()

        if not initparams:
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

        methods = ["trf", "dogbox"]

        bounds_up = [np.inf, self.theta[-1], np.inf, 1] * nop
        bounds_down = [0, self.theta[0], 0, 0] * nop
        bounds = (
            tuple([-np.inf, -np.inf] + bounds_down),
            tuple([np.inf, np.inf] + bounds_up)
        )
        
        for method in methods:
            try:
                func = Vigot_func
                (popt, pcov) = sp.optimize.curve_fit(func,
                                                    self.theta,
                                                    d,
                                                    p0 = initparams,
                                                    maxfev = 4000,
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
        return [popt, np.diag(pcov), r_squared]

    def fit(self,
            nop: int = 1
            ) -> tuple:

        l_popt = []
        l_pcov = []
        l_r2 = []

        for i in range(self.intensity.shape[0]):
            _ = self.fit_Vigot_func_solo(frame = i,
                                              nop = nop,
                                              )
            popt = _[0]
            pcov = _[1]
            r_squared = _[2]

            l_popt.append([popt])
            l_pcov.append([pcov])
            l_r2.append(r_squared)

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