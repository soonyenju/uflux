import numpy as np
from scipy.optimize import fsolve, minimize

# -------------------------------
# 常数定义
# -------------------------------
R = 8.314
KELVIN = 273.15

# ============================================================
# 🌞 辅助函数
# ============================================================
def par_from_sw(sw_down):
    return sw_down * 4.6

# ============================================================
# 🌿 Farquhar光合模型
# ============================================================
def farquhar_a(ci, tleaf, v_cmax25, j_max25, par_leaf, rd25=1.0):
    tk = tleaf + KELVIN
    Ea_v, Ea_j = 65300.0, 43540.0
    Vcmax = v_cmax25 * np.exp((Ea_v/R)*(1/298.15 - 1/tk))
    Jmax = j_max25 * np.exp((Ea_j/R)*(1/298.15 - 1/tk))
    alpha = 0.3
    J = (alpha * par_leaf) / (1 + (alpha*par_leaf)/Jmax)
    Kc25, Ko25, O2 = 404.9, 278400.0, 210000.0
    Km = Kc25 * (1 + O2/Ko25)
    Ac = Vcmax * (ci) / (ci + Km)
    Aj = (J/4.0) * (ci) / (ci + 2.0)
    Rd = rd25 * 2 ** ((tleaf - 25)/10)
    return min(Ac, Aj) - Rd

# ============================================================
# 🍃 Ball–Berry 气孔导度
# ============================================================
def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
    A_mol = A_net * 1e-6
    gs = gs0 + m * (A_mol * rh) / (cs * 1e-6)
    return gs

# ============================================================
# 🌡️ 叶片能量平衡
# ============================================================
def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
    rho_air, cp, lambda_v = 1.2, 1010, 2.45e6
    g_v = 1.6 * gs
    gb = gb0 + 0.01 * wind
    Rn = par_abs_leaf * (1 - 0.15)
    def residual(tleaf):
        H = rho_air * cp * (tleaf - tair) * gb
        LE = lambda_v * g_v * 1e-3
        return Rn - (H + LE)
    tleaf_solution = fsolve(residual, tleaf_guess)
    tleaf_final = float(tleaf_solution)
    H = rho_air * cp * (tleaf_final - tair) * gb
    LE = lambda_v * g_v * 1e-3
    return tleaf_final, H, LE

# ============================================================
# 🌈 叶绿素荧光
# ============================================================
def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
    phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
    npq = 0.3
    phi_F = phi_fmax * (1 - phi_P - npq)
    phi_F = np.clip(phi_F, 0, phi_fmax)
    return phi_F * par_abs_leaf

# ============================================================
# 🌿 叶片光学（简化PROSPECT）
# ============================================================
def leaf_absorption(cab=40, cca=10, cdm=0.01, cw=0.01, N=1.5):
    cab_factor = cab / (cab + 40)
    cca_factor = cca / (cca + 10)
    dry_factor = cdm / (cdm + 0.01)
    water_factor = cw / (cw + 0.01)
    k_abs = 1 - np.exp(-N * (0.8*cab_factor + 0.15*cca_factor + 0.05*(dry_factor + water_factor)))
    return np.clip(k_abs, 0, 1)

# ============================================================
# 🌿 LayeredSCOPE 冠层模型
# ============================================================
class LayeredSCOPE:
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10,
                 cab=40, cca=10, cdm=0.01, cw=0.01):
        self.lai = lai
        self.v_cmax25 = v_cmax25
        self.j_max25 = j_max25
        self.n_layers = n_layers
        self.cab = cab
        self.cca = cca
        self.cdm = cdm
        self.cw = cw

    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        n = len(tair)
        H_series = np.zeros(n)
        LE_series = np.zeros(n)
        GPP_series = np.zeros(n)
        SIF_series = np.zeros(n)
        tleaf_prev = tair[0]
        lai_layer = self.lai / self.n_layers
        k = 0.5
        k_leaf = leaf_absorption(self.cab, self.cca, self.cdm, self.cw)
        for i in range(n):
            par_layer = par_from_sw(sw_down[i])
            H_canopy = LE_canopy = GPP_canopy = SIF_canopy = 0.0
            for l in range(self.n_layers):
                par_abs_leaf = par_layer * k_leaf * (1 - np.exp(-k * lai_layer))
                tleaf = tleaf_prev
                for _ in range(3):
                    A_net = farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = ball_berry(A_net, rh[i], cs=co2)
                    tleaf, H_layer, LE_layer = energy_balance_leaf(tleaf, tair[i], par_abs_leaf, gs, wind[i])
                F_leaf = leaf_sif(par_abs_leaf, A_net)
                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k * lai_layer * (self.n_layers-l-1))
                H_canopy += H_layer * lai_layer / self.lai
                LE_canopy += LE_layer * lai_layer / self.lai
                par_layer -= par_abs_leaf
                tleaf_prev = tleaf
            H_series[i], LE_series[i], GPP_series[i], SIF_series[i] = H_canopy, LE_canopy, GPP_canopy, SIF_canopy
        return {"H": H_series, "LE": LE_series, "GPP": GPP_series, "SIF": SIF_series}

# ============================================================
# 🌿 DALEC 模块
# ============================================================
class DALEC:
    def __init__(self,
                 C_leaf=100, C_wood=500, C_root=200, C_litter=100, C_soil=1000,
                 tau_leaf=180, tau_wood=3650, tau_root=1095, tau_litter=365, tau_soil=10950):
        self.C = {'leaf': C_leaf, 'wood': C_wood, 'root': C_root, 'litter': C_litter, 'soil': C_soil}
        self.tau = {'leaf': tau_leaf, 'wood': tau_wood, 'root': tau_root,
                    'litter': tau_litter, 'soil': tau_soil}
        self.f_leaf = 0.3
        self.f_wood = 0.3
        self.f_root = 0.4

    def update(self, GPP, Tair):
        GPP_gC = GPP * 12e-6 * 3600 * 24
        Ra = 0.5 * GPP_gC
        NPP = GPP_gC - Ra
        dC_leaf = NPP * self.f_leaf - self.C['leaf']/self.tau['leaf']
        dC_wood = NPP * self.f_wood - self.C['wood']/self.tau['wood']
        dC_root = NPP * self.f_root - self.C['root']/self.tau['root']
        dC_litter = (self.C['leaf']/self.tau['leaf']) - (self.C['litter']/self.tau['litter'])
        dC_soil = (self.C['litter']/self.tau['litter']) - (self.C['soil']/self.tau['soil'])
        for k in self.C:
            if f'dC_{k}' in locals():
                self.C[k] += locals()[f'dC_{k}']
        Reco = Ra + (self.C['soil']/self.tau['soil'])
        Reco_umol = Reco / (12e-6 * 3600 * 24)
        LAI = max(0.1, 0.01 * self.C['leaf'])
        return Reco_umol, LAI

# ============================================================
# 🌍 SCOPE+DALEC 耦合 + NDVI 4D-Var
# ============================================================
def LAI_to_NDVI(lai, a=0.8, b=0.5):
    return a * (1 - np.exp(-b * lai))

class SCOPE_DALEC:
    def __init__(self, scope_params, dalec_params):
        self.scope = LayeredSCOPE(**scope_params)
        self.dalec = DALEC(**dalec_params)
        self.results = {'GPP': [], 'Reco': [], 'NEE': [], 'H': [], 'LE': [], 'SIF': [], 'LAI': []}

    def run(self, met):
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

    # NDVI 4D-Var 同化
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
                setattr(self.scope, n, p)
            # 模型前向
            out_scope = self.scope.run_time_series(met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind'])
            out_dalec = []
            for i in range(len(met['Tair'])):
                _, LAI = self.dalec.update(out_scope['GPP'][i], met['Tair'][i])
                out_dalec.append(LAI)
            out_dalec = np.array(out_dalec)
            ndvi_model = LAI_to_NDVI(out_dalec)
            resid = ndvi_model - ndvi_obs
            J_obs = 0.5 * resid.T @ R_inv @ resid
            delta = np.array(params) - x0
            J_b = 0.5 * delta.T @ B_inv @ delta
            return J_obs + J_b

        res = minimize(cost_function, x0, bounds=[(0,80),(20,150),(50,300)])
        for n,p in zip(param_names, res.x):
            setattr(self.scope, n, p)
        return res

# -----------------------
# 气象驱动
# -----------------------
hours = 24
met = {
    "Tair": np.linspace(15,30,hours),
    "SWdown": np.linspace(0,800,hours),
    "RH": np.linspace(0.5,0.8,hours),
    "VPD": np.linspace(0.5,2.0,hours),
    "Wind": np.linspace(0.5,3.0,hours)
}

# -----------------------
# 模型参数初始化
# -----------------------
scope_params = {"lai":3.0, "v_cmax25":70, "j_max25":140, "n_layers":10,
                "cab":40, "cca":10, "cdm":0.01, "cw":0.01}
dalec_params = {}

# -----------------------
# 构建模型
# -----------------------
model = SCOPE_DALEC(scope_params, dalec_params)

# -----------------------
# 模拟观测 NDVI
# -----------------------
ndvi_obs = np.linspace(0.3,0.7,hours)
R_inv = np.eye(hours)/0.05**2
B_inv = np.eye(3)/np.array([10,10,20])**2

# -----------------------
# 进行NDVI 4D-Var同化
# -----------------------
res = model.assimilate_ndvi(met, ndvi_obs, x0=[40,70,140], B_inv=B_inv, R_inv=R_inv)
print("优化后的参数:", res.x)

# -----------------------
# 运行耦合模型
# -----------------------
out = model.run(met)
for k,v in out.items():
    print(f"{k}: {np.round(v[:5],2)}")
