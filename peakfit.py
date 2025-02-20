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

from tqdm import tqdm

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
    
    def calc_init(self,
                  frame,
                  nop):
        d_copy = self.intensity[frame].copy()
        _a0 = (d_copy[-1]-d_copy[0])/(self.theta[-1]-self.theta[0])
        _b0 = d_copy[0]-_a0*self.theta[0]
        initparams = [
            _b0,
            _a0
        ]
        d_copy -= _a0*self.theta[0] + _b0
        for _ in range(nop):
            initparams += [
                d_copy.max()-d_copy.min(), # amp
                self.theta[d_copy.argmax()], # mu
                (self.theta[-1]-self.theta[0])/4, # fwhm_g
                (self.theta[-1]-self.theta[0])/4, # fwhm_l
                0.5
            ]
            d_copy -= pseudoVoigt(self.theta, *initparams)
        return initparams

    def fit_Vigot_func(self,
                       frame: int,
                       nop: int = 1,
                       initparams: list = None,
                       ) -> tuple:

        d = self.intensity[frame].copy()

        if not initparams:
            initparams = self.calc_init(frame = frame, nop = nop)

        methods = ["trf", "dogbox"]

        bounds_up = [np.inf, self.theta[-1], np.inf, np.inf, 1] * nop
        bounds_down = [0, self.theta[0], 0, 0, 0] * nop
        bounds = (
            tuple([-np.inf, -np.inf] + bounds_down),
            tuple([np.inf, np.inf] + bounds_up)
        )
        
        for method in methods:
            try:
                func = pseudoVoigt
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
                res = d - func(self.theta,*popt)
                rss = np.sum(np.square(res)) # residual sum of squares
                tss = np.sum(np.square(d-np.mean(d))) # total sum of squares = tss
                r_squared = 1 - (rss / tss)
                
                break
        return [popt, np.diag(pcov), r_squared]

    def fit(self,
            nop: int = 1,
            log = False
            ) -> tuple:

        l_popt = []
        l_pcov = []
        l_r2 = []

        loop = tqdm(list(range(self.intensity.shape[0]))) if log else range(self.intensity.shape[0])
        for i in loop:
            _ = self.fit_Vigot_func(frame = i,
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
    
def pseudoVoigt(x, b0, b1, *ps):
    value = b0 + b1*x
    for i in range(len(ps)//5):
        amp = ps[5*i]
        mu = ps[5*i+1]
        fwhm_g = ps[5*i+2]
        fwhm_l = ps[5*i+3]
        eta = ps[5*i+4]

        sigma = fwhm_g/2/np.sqrt(2*np.log(2))
        gamma = fwhm_l/2
        
        g = np.exp(-np.power(x-mu, 2)/2/np.power(sigma,2))
        l = 1/(1+np.power((x-mu)/gamma, 2))

        value += amp*(g*eta + l*(1-eta))
        return value

if __name__ == "__main__":

    fig, ax = plt.subplots()

    x = np.linspace(-1,1, 100)
    for eta in np.linspace(0,1,10):
        popt = [
            0,
            0,
            1, # amp
            0, # mu
            0.5, # fwhm_g
            0.5, # fwhm_l
            eta, # eta
        ]
        y = pseudoVoigt(x, *popt)
        ax.plot(x,y)
    plt.show()
    plt.close()