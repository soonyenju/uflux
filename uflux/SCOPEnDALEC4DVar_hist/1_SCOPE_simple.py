import numpy as np
from scipy.optimize import fsolve

R = 8.314
KELVIN = 273.15

def par_from_sw(sw_down):
    return sw_down * 4.6

def farquhar_a(ci, tleaf, v_cmax25, j_max25, par_leaf, rd25=1.0):
    tk = tleaf + KELVIN
    Ea_v, Ea_j = 65300.0, 43540.0
    Vcmax = v_cmax25 * np.exp((Ea_v/R)*(1/298.15 - 1/tk))
    Jmax = j_max25 * np.exp((Ea_j/R)*(1/298.15 - 1/tk))
    alpha = 0.3
    J = (alpha * par_leaf) / (1 + (alpha*par_leaf)/Jmax)
    Kc25, Ko25, O2 = 404.9, 278400.0, 210000.0
    Km = Kc25*(1 + O2/Ko25)
    Ac = Vcmax*(ci)/(ci + Km)
    Aj = (J/4.0)*(ci)/(ci + 2.0)
    Rd = rd25*2**((tleaf-25)/10)
    Agross = min(Ac, Aj)
    A_net = Agross - Rd
    return A_net

def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
    A_mol = A_net*1e-6
    gs = gs0 + m*(A_mol*rh)/(cs*1e-6)
    return gs

def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
    rho_air = 1.2
    cp = 1010
    lambda_v = 2.45e6
    g_v = 1.6*gs
    gb = gb0 + 0.01*wind
    Rn = par_abs_leaf*(1-0.15)
    def residual(tleaf):
        H = rho_air*cp*(tleaf - tair)*gb
        LE = lambda_v*g_v*1.0*1e-3
        return Rn - (H + LE)
    tleaf_solution = fsolve(residual, tleaf_guess)
    tleaf_final = float(tleaf_solution)
    H = rho_air*cp*(tleaf_final - tair)*gb
    LE = lambda_v*g_v*1.0*1e-3
    return tleaf_final, H, LE

def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
    phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
    npq = 0.3
    phi_F = phi_fmax * (1 - phi_P - npq)
    phi_F = np.clip(phi_F, 0, phi_fmax)
    F_leaf = phi_F * par_abs_leaf
    return F_leaf

class LayeredSCOPE:
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10):
        self.lai = lai
        self.v_cmax25 = v_cmax25
        self.j_max25 = j_max25
        self.n_layers = n_layers

    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        n = len(tair)
        H_series = np.zeros(n)
        LE_series = np.zeros(n)
        GPP_series = np.zeros(n)
        SIF_series = np.zeros(n)
        tleaf_prev = tair[0]

        lai_layer = self.lai / self.n_layers
        k = 0.5

        for i in range(n):
            par_layer = par_from_sw(sw_down[i])
            H_canopy = 0.0
            LE_canopy = 0.0
            GPP_canopy = 0.0
            SIF_canopy = 0.0

            for l in range(self.n_layers):
                par_abs_leaf = par_layer * (1 - np.exp(-k * lai_layer))
                tleaf = tleaf_prev
                for _ in range(3):
                    A_net = farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = ball_berry(A_net, rh[i], cs=co2)
                    tleaf, H_layer, LE_layer = energy_balance_leaf(tleaf, tair[i], par_abs_leaf, gs, wind[i])
                F_leaf = leaf_sif(par_abs_leaf, A_net)

                # 累加冠层
                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k * lai_layer * (self.n_layers-l-1))
                H_canopy += H_layer * lai_layer / self.lai  # 按叶面积加权
                LE_canopy += LE_layer * lai_layer / self.lai

                par_layer = par_layer - par_abs_leaf
                tleaf_prev = tleaf

            H_series[i] = H_canopy
            LE_series[i] = LE_canopy
            GPP_series[i] = GPP_canopy
            SIF_series[i] = SIF_canopy

        return {
            "H": H_series,
            "LE": LE_series,
            "GPP": GPP_series,
            "SIF": SIF_series
        }

# ------------------- 示例 -------------------
if __name__ == "__main__":
    hours = 24
    tair = np.linspace(15, 30, hours)
    sw_down = np.linspace(0, 800, hours)
    rh = np.linspace(0.5, 0.8, hours)
    vpd = np.linspace(0.5, 2.0, hours)
    wind = np.linspace(0.5, 3.0, hours)

    model = LayeredSCOPE(lai=4.0, v_cmax25=80.0, j_max25=160.0, n_layers=10)
    out = model.run_time_series(tair, sw_down, rh, vpd, wind)
    for k,v in out.items():
        print(f"{k}: {v}")
