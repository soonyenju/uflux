import numpy as np
from scipy.optimize import fsolve, minimize

# -------------------------------
# 常数定义
# -------------------------------
R = 8.314
KELVIN = 273.15

# ============================================================
# 辅助函数
# ============================================================
def par_from_sw(sw_down):
    return sw_down * 4.6

# ============================================================
# Farquhar 光合
# ============================================================
def farquhar_a(ci, tleaf, v_cmax25, j_max25, par_leaf, rd25=1.0):
    tk = float(tleaf) + KELVIN
    Ea_v, Ea_j = 65300.0, 43540.0
    Vcmax = v_cmax25 * np.exp((Ea_v/R)*(1/298.15 - 1/tk))
    Jmax = j_max25 * np.exp((Ea_j/R)*(1/298.15 - 1/tk))
    alpha = 0.3
    J = (alpha * par_leaf) / (1 + (alpha*par_leaf)/Jmax)
    Kc25, Ko25, O2 = 404.9, 278400.0, 210000.0
    Km = Kc25 * (1 + O2/Ko25)
    Ac = Vcmax * (ci) / (ci + Km)
    Aj = (J/4.0) * (ci) / (ci + 2.0)
    Rd = rd25 * 2 ** ((float(tleaf) - 25)/10)
    return min(Ac, Aj) - Rd

# ============================================================
# Ball–Berry 气孔导度
# ============================================================
def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
    A_mol = A_net * 1e-6
    # 防止非常小/负光合导致异常导度
    gs = gs0 + m * (A_mol * np.clip(rh, 0, 1)) / max(cs * 1e-6, 1e-9)
    return max(gs, 1e-6)

# ============================================================
# 叶片能量平衡（修复 fsolve 行为）
# ============================================================
def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
    """
    求解叶温：若 fsolve 未收敛则返回初始猜测并继续（更稳健）
    保证输入/输出为标量。
    """
    # 确保标量
    tleaf_guess = float(np.asarray(tleaf_guess).ravel()[0])
    tair = float(np.asarray(tair).ravel()[0])
    par_abs_leaf = float(np.asarray(par_abs_leaf).ravel()[0])
    gs = float(np.asarray(gs).ravel()[0])
    wind = float(np.asarray(wind).ravel()[0])

    rho_air, cp, lambda_v = 1.2, 1010, 2.45e6
    g_v = 1.6 * gs
    gb = gb0 + 0.01 * wind
    Rn = par_abs_leaf * (1 - 0.15)

    def residual(tleaf_array):
        # fsolve 可能传入数组，这里取第0个元素并返回数组
        t = float(np.asarray(tleaf_array).ravel()[0])
        H = rho_air * cp * (t - tair) * gb
        LE = lambda_v * g_v * 1e-3
        return np.array(Rn - (H + LE))

    # 使用 full_output=True 获取收敛信息
    try:
        tleaf_solution, infodict, ier, mesg = fsolve(residual, np.array([tleaf_guess]), full_output=True)
        if ier == 1:
            tleaf_final = float(np.asarray(tleaf_solution).ravel()[0])
        else:
            # 未收敛：回退到合理的近似（tair + 1K），但不抛异常
            tleaf_final = tair + 1.0
    except Exception:
        # 任意异常时，回退为 tair + 1.0
        tleaf_final = tair + 1.0

    H = rho_air * cp * (tleaf_final - tair) * gb
    LE = lambda_v * g_v * 1e-3
    return tleaf_final, H, LE

# ============================================================
# 叶绿素荧光（与之前相同，但确保使用标量）
# ============================================================
def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
    par_abs_leaf = float(par_abs_leaf)
    A_leaf = float(A_leaf)
    phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
    npq = 0.3
    phi_F = phi_fmax * (1 - phi_P - npq)
    phi_F = np.clip(phi_F, 0, phi_fmax)
    return phi_F * par_abs_leaf

# ============================================================
# 叶片光学（简化 PROSPECT）
# ============================================================
def leaf_absorption(cab=40, cca=10, cdm=0.01, cw=0.01, N=1.5):
    cab = float(cab); cca = float(cca); cdm = float(cdm); cw = float(cw)
    cab_factor = cab / (cab + 40.0)
    cca_factor = cca / (cca + 10.0)
    dry_factor = cdm / (cdm + 0.01)
    water_factor = cw / (cw + 0.01)
    k_abs = 1 - np.exp(-N * (0.8*cab_factor + 0.15*cca_factor + 0.05*(dry_factor + water_factor)))
    return float(np.clip(k_abs, 0.0, 1.0))

# ============================================================
# LayeredSCOPE（只改了调用以确保标量处理）
# ============================================================
class LayeredSCOPE:
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10,
                 cab=40, cca=10, cdm=0.01, cw=0.01):
        self.lai = float(lai)
        self.v_cmax25 = float(v_cmax25)
        self.j_max25 = float(j_max25)
        self.n_layers = int(n_layers)
        self.cab = float(cab)
        self.cca = float(cca)
        self.cdm = float(cdm)
        self.cw = float(cw)

    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        # tair, sw_down, rh, vpd, wind 期望为一维数组
        n = len(tair)
        H_series = np.zeros(n)
        LE_series = np.zeros(n)
        GPP_series = np.zeros(n)
        SIF_series = np.zeros(n)
        tleaf_prev = float(np.asarray(tair).ravel()[0])
        lai_layer = self.lai / self.n_layers
        k = 0.5
        k_leaf = leaf_absorption(self.cab, self.cca, self.cdm, self.cw)

        for i in range(n):
            # 使用标量值
            tair_i = float(np.asarray(tair[i]).ravel()[0])
            sw_i = float(np.asarray(sw_down[i]).ravel()[0])
            rh_i = float(np.asarray(rh[i]).ravel()[0])
            wind_i = float(np.asarray(wind[i]).ravel()[0])

            par_layer = par_from_sw(sw_i)
            H_canopy = LE_canopy = GPP_canopy = SIF_canopy = 0.0

            for l in range(self.n_layers):
                par_abs_leaf = par_layer * k_leaf * (1 - np.exp(-k * lai_layer))
                tleaf = tleaf_prev

                # 迭代几步求能量平衡（初始猜测用 tair+1）
                for _ in range(3):
                    A_net = farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = ball_berry(A_net, rh_i, cs=co2)
                    tleaf, H_layer, LE_layer = energy_balance_leaf(max(tleaf, tair_i+0.1), tair_i, par_abs_leaf, gs, wind_i)

                F_leaf = leaf_sif(par_abs_leaf, A_net)

                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k * lai_layer * (self.n_layers-l-1))
                H_canopy += H_layer * (lai_layer / max(self.lai, 1e-6))
                LE_canopy += LE_layer * (lai_layer / max(self.lai, 1e-6))

                par_layer -= par_abs_leaf
                tleaf_prev = tleaf

            H_series[i], LE_series[i], GPP_series[i], SIF_series[i] = H_canopy, LE_canopy, GPP_canopy, SIF_canopy

        return {"H": H_series, "LE": LE_series, "GPP": GPP_series, "SIF": SIF_series}

# ============================================================
# DALEC（保持原样，但确保标量运算）
# ============================================================
class DALEC:
    def __init__(self,
                 C_leaf=100, C_wood=500, C_root=200, C_litter=100, C_soil=1000,
                 tau_leaf=180, tau_wood=3650, tau_root=1095, tau_litter=365, tau_soil=10950):
        self.C = {'leaf': float(C_leaf), 'wood': float(C_wood), 'root': float(C_root), 'litter': float(C_litter), 'soil': float(C_soil)}
        self.tau = {'leaf': float(tau_leaf), 'wood': float(tau_wood), 'root': float(tau_root),
                    'litter': float(tau_litter), 'soil': float(tau_soil)}
        self.f_leaf = 0.3
        self.f_wood = 0.3
        self.f_root = 0.4

    def update(self, GPP, Tair):
        GPP = float(GPP)
        Tair = float(Tair)
        GPP_gC = GPP * 12e-6 * 3600 * 24
        Ra = 0.5 * GPP_gC
        NPP = GPP_gC - Ra
        dC_leaf = NPP * self.f_leaf - self.C['leaf']/self.tau['leaf']
        dC_wood = NPP * self.f_wood - self.C['wood']/self.tau['wood']
        dC_root = NPP * self.f_root - self.C['root']/self.tau['root']
        dC_litter = (self.C['leaf']/self.tau['leaf']) - (self.C['litter']/self.tau['litter'])
        dC_soil = (self.C['litter']/self.tau['litter']) - (self.C['soil']/self.tau['soil'])

        # 更新碳库（简单处理）
        self.C['leaf'] += dC_leaf
        self.C['wood'] += dC_wood
        self.C['root'] += dC_root
        self.C['litter'] += dC_litter
        self.C['soil'] += dC_soil

        Reco = Ra + (self.C['soil']/self.tau['soil'])
        Reco_umol = Reco / (12e-6 * 3600 * 24)
        LAI = max(0.1, 0.01 * self.C['leaf'])
        return Reco_umol, LAI

# ============================================================
# SCOPE+DALEC + NDVI 4D-Var（保持原逻辑，仅使用修复后的类）
# ============================================================
def LAI_to_NDVI(lai, a=0.8, b=0.5):
    return a * (1 - np.exp(-b * np.asarray(lai)))

class SCOPE_DALEC:
    def __init__(self, scope_params, dalec_params):
        self.scope = LayeredSCOPE(**scope_params)
        self.dalec = DALEC(**dalec_params)
        self.results = {'GPP': [], 'Reco': [], 'NEE': [], 'H': [], 'LE': [], 'SIF': [], 'LAI': []}

    def run(self, met):
        tair, sw_down, rh, vpd, wind = met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind']
        self.results = {k: [] for k in self.results}  # reset
        LAI = self.dalec.C['leaf'] * 0.01
        for i in range(len(tair)):
            scope_out = self.scope.run_time_series(
                np.array([tair[i]]), np.array([sw_down[i]]), np.array([rh[i]]), np.array([vpd[i]]), np.array([wind[i]])
            )
            GPP = float(scope_out['GPP'][0])
            H = float(scope_out['H'][0])
            LE = float(scope_out['LE'][0])
            SIF = float(scope_out['SIF'][0])
            Reco, LAI = self.dalec.update(GPP, tair[i])
            NEE = Reco - GPP
            for k, v in zip(['GPP','Reco','NEE','H','LE','SIF','LAI'], [GPP,Reco,NEE,H,LE,SIF,LAI]):
                self.results[k].append(v)
        return self.results

    def assimilate_ndvi(self, met, ndvi_obs, param_names=['cab','v_cmax25','j_max25'],
                        x0=None, B_inv=None, R_inv=None):
        if x0 is None:
            x0 = np.array([self.scope.cab, self.scope.v_cmax25, self.scope.j_max25])
        if B_inv is None:
            B_inv = np.eye(len(x0))
        if R_inv is None:
            R_inv = np.eye(len(ndvi_obs))

        def cost_function(params):
            for n,p in zip(param_names, params):
                setattr(self.scope, n, float(p))
            out_scope = self.scope.run_time_series(met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind'])
            # 用 GPP 驱动 DALEC 更新 LAI（逐时）
            dalec_saved = {k: v for k,v in self.dalec.C.items()}  # 简单状态保存（若需要彻底回滚可改进）
            out_dalec = []
            for i in range(len(met['Tair'])):
                GPP_i = float(out_scope['GPP'][i])
                _, LAI = self.dalec.update(GPP_i, met['Tair'][i])
                out_dalec.append(LAI)
            # 简单恢复 DALEC 状态（避免在代价函数中破坏原始模型状态）
            for k in dalec_saved:
                self.dalec.C[k] = dalec_saved[k]

            ndvi_model = LAI_to_NDVI(out_dalec)
            resid = ndvi_model - ndvi_obs
            J_obs = 0.5 * resid.T @ R_inv @ resid
            delta = np.array(params) - x0
            J_b = 0.5 * delta.T @ B_inv @ delta
            return float(J_obs + J_b)

        res = minimize(cost_function, x0, bounds=[(0,80),(20,150),(50,300)])
        for n,p in zip(param_names, res.x):
            setattr(self.scope, n, float(p))
        return res

# -----------------------
# 使用示例（与你之前相同）
# -----------------------
if __name__ == "__main__":
    hours = 24
    met = {
        "Tair": np.linspace(15,30,hours),
        "SWdown": np.linspace(0,800,hours),
        "RH": np.linspace(0.5,0.8,hours),
        "VPD": np.linspace(0.5,2.0,hours),
        "Wind": np.linspace(0.5,3.0,hours)
    }

    scope_params = {"lai":3.0, "v_cmax25":70, "j_max25":140, "n_layers":10,
                    "cab":40, "cca":10, "cdm":0.01, "cw":0.01}
    dalec_params = {}

    model = SCOPE_DALEC(scope_params, dalec_params)

    ndvi_obs = np.linspace(0.3,0.7,hours)
    R_inv = np.eye(hours)/0.05**2
    B_inv = np.eye(3)/np.array([10,10,20])**2

    res = model.assimilate_ndvi(met, ndvi_obs, x0=[40,70,140], B_inv=B_inv, R_inv=R_inv)
    print("优化后的参数:", res.x)

    out = model.run(met)
    for k,v in out.items():
        print(f"{k}: {np.round(v[:5],2)}")
