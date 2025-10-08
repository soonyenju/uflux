"""
scope_simplified.py
简化版 SCOPE 风格模型原型（Python）
依赖: numpy, scipy

包含模块：
- PAR / SW 吸收（Beer-Lambert）
- Farquhar 光合作用 (Vcmax, Jmax)
- Ball-Berry 型气孔传导
- 叶片能量平衡（求叶温）
- 驱动示例
"""

import numpy as np
from scipy.optimize import fsolve

# 常数
R = 8.314  # J mol-1 K-1
KELVIN = 273.15

def par_from_sw(sw_down):  # 粗略：将总短波转为PAR (µmol m-2 s-1)
    # 1 W m-2 ≈ 4.6 µmol m-2 s-1 (近似)
    return sw_down * 4.6

def attenuate_beer(par_down, lai, k=0.5):
    """
    Beer-Lambert: I(z) = I0 * exp(-k * LAI)
    返回吸收光合有效辐射 PAR_abs （顶端—未穿透）
    """
    trans = np.exp(-k * lai)
    par_absorbed = par_down * (1 - trans)
    return par_absorbed, trans

def farquhar_a_and_rates(ci, tleaf, v_cmax25, j_max25, i_par, rd25=1.0):
    """
    简化Farquhar模型，温度用Q10或Arrhenius校正。
    输入:
      ci: 叶内CO2 (µmol mol-1)
      tleaf: 叶温 (°C)
      v_cmax25, j_max25: 在25°C的参数
      i_par: 光合有效辐射 absorbed by leaf (µmol m-2 s-1)
    输出:
      A_net (µmol CO2 m-2 s-1)
    """
    # 温度校正 (Arrhenius 简化)
    tk = tleaf + KELVIN
    # activation energies (approx)
    Ea_v = 65300.0
    Ea_j = 43540.0
    def arrh(V25, Ea):
        return V25 * np.exp((Ea/R) * (1/298.15 - 1.0/tk))
    Vcmax = arrh(v_cmax25, Ea_v)
    Jmax = arrh(j_max25, Ea_j)

    # 简化电子传递速率 J (非光-饱和函数)
    alpha = 0.3  # quantum efficiency
    J = (alpha * i_par) / (1 + (alpha * i_par) / Jmax)

    # Michaelis-Menten constants (approx at 25C)
    Kc25 = 404.9
    Ko25 = 278400.0
    O2 = 210000.0
    # temp-dependence omitted for brevity; use constants
    Km = Kc25 * (1 + O2/Ko25)

    # Rubisco-limited
    Ac = Vcmax * (ci - 0.0) / (ci + Km)  # gammastar omitted
    # RuBP-regeneration-limited
    Aj = (J / 4.0) * (ci - 0.0) / (ci + 2.0)  # very simplified
    # respiration
    Rd = rd25 * 2.0 ** ((tleaf - 25.0)/10.0)  # Q10 = 2
    Agross = np.minimum(Ac, Aj)
    A_net = Agross - Rd
    return A_net, Vcmax, J

def ball_berry(gs0=0.01, m=9.0, rh=0.7, A_net=10.0, cs=400.0):
    """
    Ball-Berry stomatal model:
      gs = gs0 + m * (A * RH) / cs
    gs units: mol m-2 s-1
    A_net in µmol CO2 m-2 s-1 -> convert to µmol -> mol
    cs: CO2 at leaf surface (µmol mol-1)
    """
    A_mol = A_net * 1e-6  # µmol m-2 s-1 -> mol m-2 s-1
    gs = gs0 + m * (A_mol * rh) / (cs * 1e-6)  # cs convert to mol mol-1
    return gs

def energy_balance_leaf(tleaf_guess, tair, sw_down, lai, gs, gb=0.2, emissivity=0.98):
    """
    叶片能量平衡: Rn = H + LE
    简化：Net radiation ~ absorbed SW (忽略LW差别)
    H = rho_air * cp * (tleaf - tair) * conductance_H
    LE = lambda_v * evap = lambda * g_v * VPD_term (simplified)
    用 fsolve 解 tleaf
    """
    rho_air = 1.2  # kg/m3
    cp = 1010.0  # J kg-1 K-1
    lambda_v = 2.45e6  # J kg-1
    # convert gs (mol m-2 s-1) to water vapor conductance g_v (mol H2O m-2 s-1)
    # approximate: g_v = 1.6 * gs
    g_v = 1.6 * gs

    # VPD (kPa) assume from tair and rh ~ 0.7 (not full psychrometric calc)
    # We'll assume fixed VPD for simplification:
    vpd = 1.0  # kPa

    Rn = sw_down * (1 - 0.15)  # net absorbed (fractional albedo ~0.15)
    def residual(tleaf):
        # sensible
        H = rho_air * cp * (tleaf - tair) * (gb)
        # latent (approx in energy flux) — proportional to g_v and VPD
        LE = lambda_v * g_v * vpd * 1e-3  # scale to J m-2 s-1 (very rough)
        return Rn - (H + LE)
    tleaf_solution = fsolve(residual, tleaf_guess)
    return float(tleaf_solution)

class SimpleSCOPE:
    def __init__(self, params=None):
        # 默认参数
        p = {
            "lai": 3.0,
            "v_cmax25": 60.0,
            "j_max25": 120.0,
            "sw_down": 800.0,  # W m-2
            "tair": 25.0,  # °C
            "rh": 0.7,
            "co2": 410.0,  # ppm
            "k": 0.5,  # light extinction
        }
        if params: p.update(params)
        self.p = p

    def run_one_step(self):
        p = self.p
        par = par_from_sw(p["sw_down"])
        par_abs, trans = attenuate_beer(par, p["lai"], k=p["k"])
        # assume uniform per-leaf absorbed PAR (divide by LAI)
        i_par_leaf = par_abs / (p["lai"] + 1e-9)

        # initial leaf temp guess
        tleaf = p["tair"]
        # iterate between photosynthesis/stomata and energy balance
        for _ in range(8):
            A_net, Vcmax, J = farquhar_a_and_rates(p["co2"]*0.7, tleaf, p["v_cmax25"], p["j_max25"], i_par_leaf)
            gs = ball_berry(gs0=0.01, m=9.0, rh=p["rh"], A_net=A_net, cs=p["co2"])
            tleaf_new = energy_balance_leaf(tleaf, p["tair"], p["sw_down"], p["lai"], gs)
            # relax
            tleaf = 0.5 * tleaf + 0.5 * tleaf_new

        # canopy assimilation (convert per-leaf to canopy scale)
        A_canopy = A_net * p["lai"]  # µmol m-2 s-1 canopy
        # return summary
        return {
            "par_top": par,
            "par_absorbed": par_abs,
            "par_leaf": i_par_leaf,
            "A_net_leaf": A_net,
            "A_canopy": A_canopy,
            "gs": gs,
            "tleaf": tleaf,
            "Vcmax": Vcmax,
            "J": J,
            "transmittance": trans
        }

if __name__ == "__main__":
    model = SimpleSCOPE(params={
        "lai": 4.0,
        "v_cmax25": 80.0,
        "j_max25": 160.0,
        "sw_down": 1000.0,
        "tair": 30.0,
        "rh": 0.6,
        "co2": 420.0
    })
    out = model.run_one_step()
    for k,v in out.items():
        print(f"{k}: {v}")
