import numpy as np
from scipy.optimize import fsolve

# -------------------------------
# 常数定义
# -------------------------------
R = 8.314  # 通用气体常数 [J mol⁻¹ K⁻¹]
KELVIN = 273.15  # 摄氏转开尔文

# ============================================================
# 🌞 辅助函数
# ============================================================

def par_from_sw(sw_down):
    """
    将短波辐射（SWdown, W m⁻²）转换为光合有效辐射（PAR, μmol photon m⁻² s⁻¹）
    经验公式：1 W m⁻² ≈ 4.6 μmol photon m⁻² s⁻¹
    """
    return sw_down * 4.6


# ============================================================
# 🌿 Farquhar光合模型 (单叶)
# ============================================================
def farquhar_a(ci, tleaf, v_cmax25, j_max25, par_leaf, rd25=1.0):
    """
    计算叶片净光合速率 (μmol CO2 m⁻² s⁻¹)

    参数:
    - ci: 细胞间CO₂浓度 [μmol mol⁻¹]
    - tleaf: 叶温 [°C]
    - v_cmax25: 25°C下最大羧化速率 [μmol m⁻² s⁻¹]
    - j_max25: 25°C下最大电子传递速率 [μmol m⁻² s⁻¹]
    - par_leaf: 叶片吸收的光合有效辐射 [μmol photon m⁻² s⁻¹]
    - rd25: 25°C下呼吸速率 [μmol m⁻² s⁻¹]

    返回:
    - A_net: 净光合速率 [μmol CO₂ m⁻² s⁻¹]
    """
    tk = tleaf + KELVIN  # 转为K
    Ea_v, Ea_j = 65300.0, 43540.0  # 温度响应活化能 [J mol⁻¹]
    Vcmax = v_cmax25 * np.exp((Ea_v/R)*(1/298.15 - 1/tk))
    Jmax = j_max25 * np.exp((Ea_j/R)*(1/298.15 - 1/tk))
    alpha = 0.3  # 光量子利用率 (经验值)
    J = (alpha * par_leaf) / (1 + (alpha*par_leaf)/Jmax)  # 电子传递速率

    # Michaelis-Menten常数
    Kc25, Ko25, O2 = 404.9, 278400.0, 210000.0
    Km = Kc25 * (1 + O2/Ko25)

    Ac = Vcmax * (ci) / (ci + Km)        # Rubisco限制
    Aj = (J/4.0) * (ci) / (ci + 2.0)     # 电子传递限制
    Rd = rd25 * 2 ** ((tleaf - 25)/10)   # 呼吸温度响应
    Agross = min(Ac, Aj)
    A_net = Agross - Rd
    return A_net


# ============================================================
# 🍃 Ball–Berry 气孔导度模型
# ============================================================
def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
    """
    Ball-Berry模型计算气孔导度 [mol m⁻² s⁻¹]

    参数:
    - A_net: 净光合速率 [μmol CO₂ m⁻² s⁻¹]
    - rh: 相对湿度 [0–1]
    - cs: 叶片表面CO₂浓度 [μmol mol⁻¹]
    - gs0: 最小导度 [mol m⁻² s⁻¹]
    - m: 经验斜率参数

    返回:
    - gs: 气孔导度 [mol m⁻² s⁻¹]
    """
    A_mol = A_net * 1e-6  # μmol→mol
    gs = gs0 + m * (A_mol * rh) / (cs * 1e-6)
    return gs


# ============================================================
# 🌡️ 叶片能量平衡方程
# ============================================================
def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
    """
    叶片能量平衡方程：求解叶温，使得净辐射 ≈ 感热 + 潜热

    参数:
    - tleaf_guess: 初始叶温猜测 [°C]
    - tair: 空气温度 [°C]
    - par_abs_leaf: 叶片吸收光通量 [μmol photon m⁻² s⁻¹]
    - gs: 气孔导度 [mol m⁻² s⁻¹]
    - wind: 风速 [m s⁻¹]
    - gb0: 静风边界层导度 [mol m⁻² s⁻¹]

    返回:
    - tleaf_final: 平衡叶温 [°C]
    - H: 感热通量 [W m⁻²]
    - LE: 潜热通量 [W m⁻²]
    """
    rho_air = 1.2        # 空气密度 [kg m⁻³]
    cp = 1010            # 比热容 [J kg⁻¹ K⁻¹]
    lambda_v = 2.45e6    # 水汽潜热 [J kg⁻¹]
    g_v = 1.6 * gs       # 水汽导度 [mol m⁻² s⁻¹]
    gb = gb0 + 0.01 * wind  # 边界层导度随风速增加
    Rn = par_abs_leaf * (1 - 0.15)  # 净辐射 (简化)

    def residual(tleaf):
        H = rho_air * cp * (tleaf - tair) * gb
        LE = lambda_v * g_v * 1e-3  # 假设饱和
        return Rn - (H + LE)

    tleaf_solution = fsolve(residual, tleaf_guess)
    tleaf_final = float(tleaf_solution)
    H = rho_air * cp * (tleaf_final - tair) * gb
    LE = lambda_v * g_v * 1e-3
    return tleaf_final, H, LE


# ============================================================
# 🌈 叶绿素荧光 (SIF)
# ============================================================
def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
    """
    模拟叶片层的荧光辐射强度

    参数:
    - par_abs_leaf: 吸收光合光量子通量 [μmol photon m⁻² s⁻¹]
    - A_leaf: 光合速率 [μmol CO₂ m⁻² s⁻¹]
    - phi_fmax: 最大荧光量子产率 (典型值0.02–0.05)

    返回:
    - F_leaf: 荧光辐射 [μmol photon m⁻² s⁻¹]
    """
    phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
    npq = 0.3  # 非光化学淬灭系数
    phi_F = phi_fmax * (1 - phi_P - npq)
    phi_F = np.clip(phi_F, 0, phi_fmax)
    F_leaf = phi_F * par_abs_leaf
    return F_leaf


# ============================================================
# 🌿 LayeredSCOPE 冠层模型
# ============================================================
class LayeredSCOPE:
    """
    简化SCOPE结构的逐层冠层模型
    计算逐时 H / LE / GPP / SIF
    """
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10):
        self.lai = lai
        self.v_cmax25 = v_cmax25
        self.j_max25 = j_max25
        self.n_layers = n_layers

    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        """
        逐小时运行冠层能量-光合-荧光模块

        输入：
        - tair: 气温 [°C]
        - sw_down: 短波辐射 [W m⁻²]
        - rh: 相对湿度 [0–1]
        - vpd: 饱和水汽压差 [kPa]
        - wind: 风速 [m s⁻¹]
        - co2: 大气CO₂浓度 [μmol mol⁻¹]
        """
        n = len(tair)
        H_series = np.zeros(n)
        LE_series = np.zeros(n)
        GPP_series = np.zeros(n)
        SIF_series = np.zeros(n)
        tleaf_prev = tair[0]

        lai_layer = self.lai / self.n_layers
        k = 0.5  # 光衰减系数

        for i in range(n):
            par_layer = par_from_sw(sw_down[i])
            H_canopy = LE_canopy = GPP_canopy = SIF_canopy = 0.0

            for l in range(self.n_layers):
                par_abs_leaf = par_layer * (1 - np.exp(-k * lai_layer))
                tleaf = tleaf_prev

                # 迭代求解能量平衡
                for _ in range(3):
                    A_net = farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = ball_berry(A_net, rh[i], cs=co2)
                    tleaf, H_layer, LE_layer = energy_balance_leaf(tleaf, tair[i], par_abs_leaf, gs, wind[i])

                F_leaf = leaf_sif(par_abs_leaf, A_net)

                # 层积分
                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k * lai_layer * (self.n_layers-l-1))
                H_canopy += H_layer * lai_layer / self.lai
                LE_canopy += LE_layer * lai_layer / self.lai

                par_layer -= par_abs_leaf
                tleaf_prev = tleaf

            H_series[i], LE_series[i], GPP_series[i], SIF_series[i] = H_canopy, LE_canopy, GPP_canopy, SIF_canopy

        return {"H": H_series, "LE": LE_series, "GPP": GPP_series, "SIF": SIF_series}


# ============================================================
# 🌲 DALEC 模块
# ============================================================
class DALEC:
    """
    简化版DALEC碳循环模型
    包括叶、木、根、凋落物、土壤碳库，按日更新
    """
    def __init__(self,
                 C_leaf=100, C_wood=500, C_root=200, C_litter=100, C_soil=1000,
                 tau_leaf=180, tau_wood=3650, tau_root=1095, tau_litter=365, tau_soil=10950):
        """
        初始化碳库和周转时间
        - C_* : 碳库量 [g C m⁻²]
        - tau_* : 周转时间 [天]
        """
        self.C = {'leaf': C_leaf, 'wood': C_wood, 'root': C_root, 'litter': C_litter, 'soil': C_soil}
        self.tau = {'leaf': tau_leaf, 'wood': tau_wood, 'root': tau_root, 'litter': tau_litter, 'soil': tau_soil}
        self.f_leaf = 0.3
        self.f_wood = 0.3
        self.f_root = 0.4

    def update(self, GPP, Tair):
        """
        更新碳库

        输入:
        - GPP: 冠层光合碳固定 [μmol CO₂ m⁻² s⁻¹]
        - Tair: 气温 [°C]

        返回:
        - Reco: 总呼吸 [μmol CO₂ m⁻² s⁻¹]
        - LAI: 更新后的叶面积指数 [m² m⁻²]
        """
        # 光合碳转为gC单位: 1 μmol CO₂ = 12e-6 gC
        GPP_gC = GPP * 12e-6 * 3600 * 24  # [gC m⁻² d⁻¹]

        # 自养呼吸
        Ra = 0.5 * GPP_gC
        NPP = GPP_gC - Ra

        # 碳分配
        dC_leaf = NPP * self.f_leaf - self.C['leaf'] / self.tau['leaf']
        dC_wood = NPP * self.f_wood - self.C['wood'] / self.tau['wood']
        dC_root = NPP * self.f_root - self.C['root'] / self.tau['root']
        dC_litter = (self.C['leaf'] / self.tau['leaf']) - (self.C['litter'] / self.tau['litter'])
        dC_soil = (self.C['litter'] / self.tau['litter']) - (self.C['soil'] / self.tau['soil'])

        for k in self.C:
            if f'dC_{k}' in locals():
                self.C[k] += locals()[f'dC_{k}']

        # 生态系统呼吸
        Reco = Ra + (self.C['soil'] / self.tau['soil'])
        Reco_umol = Reco / (12e-6 * 3600 * 24)
        LAI = max(0.1, 0.01 * self.C['leaf'])  # 经验关系

        return Reco_umol, LAI


# ============================================================
# 🌍 耦合模型
# ============================================================
class SCOPE_DALEC:
    def __init__(self, scope_params, dalec_params):
        self.scope = LayeredSCOPE(**scope_params)
        self.dalec = DALEC(**dalec_params)
        self.results = {'GPP': [], 'Reco': [], 'NEE': [], 'H': [], 'LE': [], 'SIF': [], 'LAI': []}

    def run(self, met):
        """
        输入：
        met = {
          'Tair': [°C],
          'SWdown': [W m⁻²],
          'RH': [0–1],
          'VPD': [kPa],
          'Wind': [m s⁻¹]
        }
        """
        tair, sw_down, rh, vpd, wind = met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind']
        LAI = self.dalec.C['leaf'] * 0.01

        for i in range(len(tair)):
            scope_out = self.scope.run_time_series(
                tair[i:i+1], sw_down[i:i+1], rh[i:i+1], vpd[i:i+1], wind[i:i+1]
            )
            GPP = scope_out['GPP'][0]
            H = scope_out['H'][0]
            LE = scope_out['LE'][0]
            SIF = scope_out['SIF'][0]

            Reco, LAI = self.dalec.update(GPP, tair[i])
            NEE = Reco - GPP

            for k, v in zip(['GPP','Reco','NEE','H','LE','SIF','LAI'], [GPP,Reco,NEE,H,LE,SIF,LAI]):
                self.results[k].append(v)

        return self.results


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    hours = 24
    met = {
        "Tair": np.linspace(15, 30, hours),
        "SWdown": np.linspace(0, 800, hours),
        "RH": np.linspace(0.5, 0.8, hours),
        "VPD": np.linspace(0.5, 2.0, hours),
        "Wind": np.linspace(0.5, 3.0, hours),
    }

    scope_params = {"lai": 3.0, "v_cmax25": 70, "j_max25": 140, "n_layers": 10}
    dalec_params = {}

    model = SCOPE_DALEC(scope_params, dalec_params)
    out = model.run(met)

    for k,v in out.items():
        print(f"{k}: {np.round(v[:5],2)}")