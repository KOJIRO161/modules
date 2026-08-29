# 変更履歴
## 2026-01-23 fitの出力を変更しました
## 2026-01-23 pseudoVoigt関数を書き換えました

import sys
sys.dont_write_bytecode = True # cacheを出さない

import numpy as np # type: ignore
import scipy as sp # type: ignore

class peakfit:
    def __init__(self):
        """peakをフィットするクラス"""
        self._set_initparams()

    def _set_initparams(self):
        self.x = np.array([])
        self.y = np.array([])
    
    def calc_init(self,
                  x: np.ndarray = np.array([]),
                  y:np.ndarray = np.array([]),
                  nop: int = 1,
                  ) -> list:
        """
        To calculate initial parameters for fitting
        
        :param x: Value of x
        :type x: np.ndarray
        :param y: Value of y
        :type y: np.ndarray
        :param nop: Number of peaks
        :type nop: int
        :return: Intial parameters for fitting
        :rtype: list
        """
        
        if len(x) == 0:
            _x = self.x.copy()
        else:
            _x = x.copy()

        if len(y) == 0:
            _y = self.y.copy()
        else:
            _y = y.copy()

        _a0 = (_y[-1]-_y[0])/(_x[-1]-_x[0])
        _b0 = _y[0]-_a0*_x[0]
        initparams = [
            _b0,
            _a0
        ]
        _y -= _a0*_x[0] + _b0
        for _ in range(nop):
            initparams += [
                _y.max()-_y.min(), # amp
                _x[_y.argmax()], # mu
                (_x[-1]-_x[0])/4, # fwhm_g
                # (_x[-1]-_x[0])/4, # fwhm_l
                0.5
            ]
            _y -= pseudoVoigt(_x, *initparams)
        return initparams

    def fit(self,
            x: np.ndarray = np.array([]),
            y: np.ndarray = np.array([]), 
            nop: int = 1,
            initparams: list = [],
            ) -> tuple:
        """
        To fit data with pseudo-Voigt function
        
        :param x: Value of x
        :type x: np.ndarray
        :param y: Value of y
        :type y: np.ndarray
        :param nop: Number of peaks
        :type nop: int
        :param initparams: Initial parameters for fitting
        :type initparams: list
        :return: Fitting result
        :rtype: tuple[np.ndarray, np.ndarray, dict]
        """

        if len(x) == 0:
            _x = self.x.copy()
        else:
            _x = x.copy()
        if len(y) == 0:
            _y = self.y.copy()
        else:
            _y = y.copy()

        if len(initparams) == 0:
            _initparams = self.calc_init(_x,_y,nop)
        else:
            _initparams = initparams
        
        methods = ["trf", "dogbox"]

        # bounds_up = [np.inf, _x[-1], np.inf, np.inf, 1] * nop
        # bounds_down = [0, _x[0], 0, 0, 0] * nop
        bounds_up = [np.inf, _x[-1], np.inf, 1] * nop
        bounds_down = [0, _x[0], 0, 0] * nop
        bounds = (
            tuple([-np.inf, -np.inf] + bounds_down),
            tuple([np.inf, np.inf] + bounds_up)
        )
        
        for method in methods:
            try:
                func = pseudoVoigt
                (popt, pcov) = sp.optimize.curve_fit(func,
                                                     _x,
                                                     _y,
                                                     p0 = _initparams,
                                                     maxfev = 4000,
                                                     bounds=bounds,
                                                     method = method,
                                                     )
            except RuntimeError as errorcontent:
                if method == methods[-1]:
                    return errorcontent # type: ignore
                pass
            else:
                # 格納用の辞書
                others = dict()

                # r2
                res = _y - func(_x,*popt)
                rss = np.sum(np.square(res)) # residual sum of squares
                tss = np.sum(np.square(_y-np.mean(_y))) # total sum of squares = tss
                r_squared = float(1 - (rss / tss))
                others["r_squared"] = r_squared

                # 誤差の算出
                for i in range(nop):
                    # 格納用
                    others[i] = dict()

                    # 値の抽出
                    amp = popt[2+4*i]
                    fwhm = popt[4+4*i]
                    eta = popt[5+4*i]

                    # 誤差の計算
                    bg = np.std(res)
                    height = 2*amp/np.pi/fwhm * (1+(np.sqrt(np.pi*np.log(2)) - 1)*eta)
                    sn = height / np.sqrt(height + bg)
                    err = fwhm/sn

                    # 格納
                    others[i]["Peak height"] = height
                    others[i]["Back ground"] = bg
                    others[i]["S/N ratio"] = sn
                    others[i]["Error"] = err

                break
        
        return (popt, np.diag(pcov), others) # type: ignore
    
    def res2dict(self,
                 res: tuple):
        """
        To make dict from fitting result
        
        :param res: Fitting result
        :type res: tuple[np.ndarray, np.ndarray, float]
        """

        # データ読み込み
        popt, pcov, r_squared = res
        
        # ピーク数の取得
        nop = (len(popt)-2)//5

        # 辞書の作成
        d_popt = dict(zip(self.variables(nop), list(popt)))
        d_pcov = dict(zip(self.variables(nop), list(pcov)))
        d = {
            'popt': d_popt,
            'pcov': d_pcov,
            'r2': r_squared,
        }
        return d

    def variables_regacy(self, nop = 1):
        value = ["b0", "b1"]
        value += ["amp", "mu", "fwhm_g", "fwhm_l", "eta"] * nop
        return value

    def variables(self, nop = 1):
        value = ["b0", "b1"]
        value += ["amp", "mu", "fwhm", "eta"] * nop
        return value

def pseudoVoigt_regacy(x, b0, b1, *ps):
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


def spseudoVoigt(x, b0, b1, *ps):
    value = b0 + b1*x
    for i in range(len(ps)//4):
        amp = ps[4*i]
        mu = ps[4*i+1]
        fwhm = ps[4*i+2]
        eta = ps[4*i+3]

        # gaussian
        sigma = fwhm/2/np.sqrt(2*np.log(2))
        ag = 1/np.sqrt(2*np.pi) / sigma
        bg = 4*np.log(2) / np.square(fwhm)
        g = ag * np.exp(-bg*np.square(x-mu))

        # lorenzian
        l = 1/np.pi * fwhm/2/(np.square(x-mu) + np.square(fwhm/2))

        value += amp*(g*eta + l*(1-eta))
    return value