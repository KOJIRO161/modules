import sys
sys.dont_write_bytecode = True

# データ分析ツール
import numpy as np
import scipy as sp

class Vinet:
    def __init__(self):
        self.v0 = 50
        self.k0 = 15
        self.k0prime = 6
        
        self.err_v0 = 0
        self.err_k0 = 0
        self.err_k0prime = 0
    
    def get_pressure(self, v: np.ndarray):
        x = (self.v0 / v)**(1/3)
        p = 3*self.k0*(x**2-x)*np.exp(3/2 * (self.k0prime-1) * (1-1/x))
        return p
    
    def get_pressure_err(self, v: np.ndarray, errv: np.ndarray):
        x = (self.v0 / v)**(1/3)
        dxdv = -1/3 * x**4 / self.v0
        dxdv0 = 1/3 * x**(-2) / v
        dpdx = 3 * self.k0 * ((2*x - 1)
                              + 3/2 * (self.k0prime-1) * (1-x**(-1))) * np.exp(3/2 * (self.k0prime-1) * (1-1/x))
        dpdk0 = 3*(x**2-x)*np.exp(3/2 * (self.k0prime-1) * (1-1/x))
        dpdk0prime = 3*self.k0*(x**2-x)*np.exp(3/2 * (self.k0prime-1) * (1-1/x)) * 3/2*(1-x**(-1))
        err = np.sqrt(
            np.square(dpdx*dxdv*errv)
            + np.square(dpdx*dxdv0*self.err_v0)
            + np.square(dpdk0*self.err_k0)
            + np.square(dpdk0prime*self.err_k0prime)
        )
        return err
    
    def get_dpdv(self, v):
        x = (self.v0 / v)**(1/3)
        dxdv = -1/3 * x**4 / self.v0
        dpdx = 3 * self.k0 * ((2*x - 1)
                              + 3/2 * (self.k0prime-1) * (1-x**(-1))) * np.exp(3/2 * (self.k0prime-1) * (1-1/x))
        return dpdx*dxdv

class KCl_Dewele(Vinet):
    def __init__(self):
        super().__init__()
        self.v0 = 54.5
        self.k0 = 17.2
        self.k0prime = 5.89
        self.akt = 2.24e-3

        self.err_v0 = 0.05
        self.err_k0 = 0.05
        self.err_k0prime = 0.005
        self.err_akt = 1e-5

    def get_pressure_with_thermal(self, v: np.ndarray, t: np.ndarray):
        p = self.get_pressure(v)
        p += self.akt *(t-300)
        return p
    
    def get_pressure_err_with_thermal(self, v: np.ndarray, t: np.ndarray,
                                      errv: np.ndarray, errt: np.ndarray):
        perr_300 = self.get_pressure_err(v = v, errv=errv)
        perr_themal = np.sqrt(np.square((t-300) * self.err_akt) + np.square(self.akt*errt))
        return np.sqrt(np.square(perr_300) + np.square(perr_themal))

class Fe_Dorogokupets(Vinet):
    def __init__(self):
        super().__init__()
        NA = 6.02214076e23 # /mol
        v0 =  6.9285e24 # A^3/mol
        
        self.v0 = v0/NA # A^3
        self.k0 = 146.2 # GPa
        self.k0prime = 4.67

        self.theta0 = 222.5 # K
        self.gamma_inf = 0
        self.gamma_0 = 2.203
        self.beta = 0.01
        self.n = 1
        
        self.e0 = 198e-6
        self.g = 0.5

    def get_pressure_with_thermal(self, v, t):
        p0 = self.get_pressure(v)

        gruneisen = thermalEOS_gruneisen()
        gruneisen.v0 = self.v0
        gruneisen.theta0 = self.theta0
        gruneisen.gamma_inf = self.gamma_inf
        gruneisen.gamma_0 = self.gamma_0
        gruneisen.beta = self.beta
        gruneisen.n = self.n
        pth =  gruneisen.get_thermal_pressure_Einstein(v,t)
        pth_300 = gruneisen.get_thermal_pressure_Einstein(v,300)

        electron = thermalEOS_electron()
        electron.v0 = self.v0
        electron.n = self.n
        electron.e0 = self.e0
        electron.g = self.g
        pe = electron.get_thermal_pressure(v,t)
        pe_300 = electron.get_thermal_pressure(v,300)

        return p0 + pth-pth_300 + pe-pe_300

    def get_dpdv(self,v,t):
        dp0dv = super().get_dpdv(v)

        gruneisen = thermalEOS_gruneisen()
        gruneisen.v0 = self.v0
        gruneisen.theta0 = self.theta0
        gruneisen.gamma_inf = self.gamma_inf
        gruneisen.gamma_0 = self.gamma_0
        gruneisen.beta = self.beta
        gruneisen.n = self.n
        dpthdv = gruneisen.get_dpdv_Einstein(v,t)
        dpthdv_300 = gruneisen.get_dpdv_Einstein(v,300)

        electron = thermalEOS_electron()
        electron.v0 = self.v0
        electron.n = self.n
        electron.e0 = self.e0
        electron.g = self.g
        dpedv = electron.get_dpdv(v,t)
        dpedv_300 = electron.get_dpdv(v,300)

        return dp0dv + dpthdv-dpthdv_300 + dpedv-dpedv_300
    
    def get_dpdt(self,v,t):

        gruneisen = thermalEOS_gruneisen()
        gruneisen.v0 = self.v0
        gruneisen.theta0 = self.theta0
        gruneisen.gamma_inf = self.gamma_inf
        gruneisen.gamma_0 = self.gamma_0
        gruneisen.beta = self.beta
        gruneisen.n = self.n
        dpthdt = gruneisen.get_dpdt_Einstein(v,t)

        electron = thermalEOS_electron()
        electron.v0 = self.v0
        electron.n = self.n
        electron.e0 = self.e0
        electron.g = self.g
        dpedt = electron.get_dpdt(v,t)

        return dpthdt+dpedt        

class Birch_Murnaghan_3rd:
    def __init__(self):
        self.v0 = 50
        self.k0 = 15
        self.k0prime = 6
        
        self.err_v0 = 0
        self.err_k0 = 0
        self.err_k0prime = 0
        return
    
    def get_pressure(self, v: np.ndarray):
        x = (self.v0 / v)**(1/3)
        p = 3/2 * self.k0 *(x**7 - x**5) * (1 + 3/4*(self.k0prime-4)*(x**2 - 1))
        return p
    
    def get_pressure_err(self, v: np.ndarray, errv: np.ndarray):
        x = (self.v0 / v)**(1/3)
        dxdv = -1/3 * x**4 / self.v0
        dxdv0 = 1/3 * x**(-2) / v
        dpdx = 3/2 * self.k0 + x**4 * ((7*x*2 - 5)
                                       + 3/4 * (self.k0prime-4) * (x**2-1) * (9*x**2-5))
        dpdk0 = 3/2 + (x**7 - x**5) * (1 + 3/4*(self.k0prime-4)*(x**2 - 1))
        dpdk0prime = 3/2 * self.k0 * (x**7 - x**5) * 3/4 * (x**2 - 1)
        err = np.sqrt(
            np.square(dpdx*dxdv*errv)
            + np.square(dpdx*dxdv0*self.err_v0)
            + np.square(dpdk0*self.err_k0)
            + np.square(dpdk0prime*self.err_k0prime)
        )
        return err

class thermalEOS_gruneisen:
    def __init__(self):
        self.v0 = None
        self.theta0 = None
        self.gamma_inf = None
        self.gamma_0 = None
        self.beta = None
        self.n = None

    def get_thermal_pressure(self, v, t):

        flag = False
        if type(v) == np.ndarray:
            flag = True
        if type(t) == np.ndarray:
            flag = True
        if flag:
            if not type(v) == np.ndarray:
                v = np.ones(t.shape)*v
            if not type(t) ==  np.ndarray:
                t = np.ones(v.shape)*t

        x = (self.v0 / v)**(1/3)
        thetaD = self.theta0 * x**(-self.gamma_inf)*np.exp((self.gamma_0-self.gamma_inf)/self.beta * (1-x**self.beta))

        gamma = self.gamma_inf + (self.gamma_0 - self.gamma_inf)*x**self.beta
        
        if flag:
            deth_list = []
            for i in range(len(v)):
                deth_list.append(
                    9*self.n*sp.constants.k * (
                        t[i]*(t[i]/thetaD[i])**3
                        * sp.integrate.quad(self.integral_func, 0, (thetaD[i]/t[i]))[0]
                        -
                        300*(300/thetaD[i])**3
                        * sp.integrate.quad(self.integral_func, 0, (thetaD[i]/300))[0]
                    )
                )
            deth = np.array(deth_list)
        else:
            deth = 9*self.n*sp.constants.k * (
                        t*(t/thetaD)**3
                        * sp.integrate.quad(self.integral_func, 0, (thetaD/t))[0]
                        -
                        300*(300/thetaD)**3
                        * sp.integrate.quad(self.integral_func, 0, (thetaD/300))[0]
                    )
            
        return gamma/v*deth * 1e21 # Pa * A^3 / m^3 --> GPa

    # グリュナイゼンエネルギーの計算
    def integral_func(self, x):
        return x**3 / (np.exp(x) - 1)
    
    def get_thermal_pressure_Einstein(self, v, t):
        x = v/self.v0
        gamma = self.gamma_inf + (self.gamma_0 - self.gamma_inf)*np.power(x, self.beta)
        theta = self.theta0 * np.power(x,-self.gamma_inf) * np.exp((self.gamma_0 - self.gamma_inf)/self.beta * (1-np.power(x,self.beta)))

        R = 1.380649e-2 # A^3 GPa K^-1

        value = 3*self.n*R*gamma/v*theta/(np.exp(theta/t)-1)
        return value
    
    def get_dpdv_Einstein(self,v,t):
        x = v/self.v0
        gamma = self.gamma_inf + (self.gamma_0 - self.gamma_inf)*np.power(x, self.beta)
        theta = self.theta0 * np.power(x,-self.gamma_inf) * np.exp((self.gamma_0 - self.gamma_inf)/self.beta * (1-np.power(x,self.beta)))

        pth = self.get_thermal_pressure_Einstein(v,t)

        dpthdv = -pth/v
        
        dpthdgamma = pth/gamma
        dgammadx = self.beta*(self.gamma_0-self.gamma_inf)*np.power(x,self.beta-1)

        dpthdtheta = pth/theta * (1 - theta/t*(np.exp(theta/t) / (np.exp(theta/t) - 1)))
        dthetadx = -gamma*theta/x

        dxdv = 1/self.v0

        value = dpthdv + (dpthdgamma*dgammadx + dpthdtheta*dthetadx)*dxdv
        return value
    
    def get_dpdt_Einstein(self,v,t):
        x = v/self.v0
        # gamma = self.gamma_inf + (self.gamma_0 - self.gamma_inf)*np.power(x, self.beta)
        theta = self.theta0 * np.power(x,-self.gamma_inf) * np.exp((self.gamma_0 - self.gamma_inf)/self.beta * (1-np.power(x,self.beta)))

        pth = self.get_thermal_pressure_Einstein(v,t)
        dpthdt = pth/t * theta/t*(np.exp(theta/t) / (np.exp(theta/t) - 1))

        return dpthdt

class thermalEOS_electron:
    def __init__(self):
        self.n = None
        self.e0 = None
        self.g = None
        self.v0 = None

    def get_thermal_pressure(self, v, t):
        NA = 6.02214076e23 # /mol
        R = 8.31447e21 / NA # GPa A^3 / K
        p = 3/2 * self.n * R * self.e0 * (v/self.v0)**self.g / v * t**2
        return p

    def get_dpdv(self,v,t):
        x = v/self.v0
        pe = self.get_thermal_pressure(v,t)
        
        dpedv = -pe/v
        dpedx = self.g*pe/x

        dxdv = 1/self.v0

        value = dpedv + dpedx*dxdv
        return value
    
    def get_dpdt(self, v, t):
        pe = self.get_thermal_pressure(v,t)
        dpedt = 2*pe/t
        return dpedt

class FeH_Tagawa(Birch_Murnaghan_3rd):
    def __init__(self):
        super().__init__()
        self.v0 = 13.45 # [A^3 / f.u.]
        self.k0 = 183 # [GPa]
        self.k0prime = 3.84

        self.theta0 = 758 # [K]
        self.gamma_inf = 0.547
        self.gamma_0 = 0.738
        self.beta = self.gamma_0 / (self.gamma_0 - self.gamma_inf)
        self.n = 2
        
        return
    
    def get_pressure_with_thermal(self, v, t):

        thermalEOS = thermalEOS_gruneisen()
        thermalEOS.v0 = self.v0
        thermalEOS.theta0 = self.theta0
        thermalEOS.gamma_inf = self.gamma_inf
        thermalEOS.gamma_0 = self.gamma_0
        thermalEOS.beta = self.beta
        thermalEOS.n = self.n

        p = self.get_pressure(v) + thermalEOS.get_thermal_pressure(v, t)
        return p
