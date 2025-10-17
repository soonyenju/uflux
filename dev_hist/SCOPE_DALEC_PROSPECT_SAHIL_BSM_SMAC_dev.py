import numpy as np
from scipy.optimize import fsolve, minimize
import scipy.integrate as integrate
from dataclasses import dataclass
from typing import Dict, List, Any
from scipy.stats import poisson

# -------------------------------
# 核心常数定义
# -------------------------------
R = 8.314 # 理想气体常数 (J / (mol * K))
KELVIN = 273.15 # 绝对零度与摄氏度的差值 (K)
H_PLANCK = 6.626e-34 # 普朗克常数 (J * s)
C_LIGHT = 2.998e8 # 光速 (m / s)


# ============================================================
# 模块 A: PROSPECT/叶片光学模型 (Leaf Optics Module)
# ============================================================

@dataclass
class LeafBiology:
    """叶片生物参数 (用于 PROSPECT 输入)"""
    Cab: float # 叶绿素 a+b 含量 (ug / cm^2)
    Cdm: float # 干物质含量 (g / cm^2)
    Cw: float # 水分含量 (g / cm^2)
    Cs: float # 结构化碳水化合物含量 (g / cm^2)
    Cca: float # 类胡萝卜素含量 (ug / cm^2)
    Cant: float # 花青素含量 (ug / cm^2)
    N: float # 叶片结构参数/有效层数 (无单位)
    PROT: float = 0.0 # 蛋白质含量 (g / cm^2)
    CBC: float = 0.0 # 褐色素含量 (g / cm^2)
    rho_thermal: float = 0.01 # 热反射率 (无单位)
    tau_thermal: float = 0.01 # 热透射率 (无单位)

@dataclass
class LeafOptics:
    """叶片光学输出 (来自 PROSPECT)"""
    refl: np.ndarray # 叶片半球反射率 (波长数组) (无单位)
    tran: np.ndarray # 叶片半球透射率 (波长数组) (无单位)
    kChlrel: np.ndarray # 叶绿素相对吸收系数 (波长数组) (无单位)
    par_absorption_coeff: float = 0.0 # PAR 波段平均叶片吸收系数 (0-1) (无单位)


class PROSPECT_Model:
    """封装 PROSPECT-PRO 模型及其辅助函数"""

    @staticmethod
    def calculate_tav(alpha, nr):
        """
        计算介电平面平均透射率 (t_av)。
        输入：alpha (度), nr (相对折射率数组); 输出：tav (平均透射率数组)
        """
        rd = np.pi / 180
        nr = np.atleast_1d(nr)
        n2 = nr ** 2; n_p = n2 + 1; nm = n2 - 1
        a = (nr + 1) * (nr + 1) / 2
        k = -(n2 - 1) * (n2 - 1) / 4
        sa = np.sin(alpha * rd)

        b1 = 0
        if alpha != 90:
            b1 = np.sqrt(np.clip((sa ** 2 - n_p / 2) ** 2 + k, 0, np.inf))
        b2 = sa ** 2 - n_p / 2
        b = b1 - b2

        a3 = a ** 3; b3 = b ** 3
        ts = np.zeros_like(a); mask_a = a > 1e-9
        ts[mask_a] = (k[mask_a] ** 2 / (6 * b3[mask_a]) + k[mask_a] / b[mask_a] - b[mask_a] / 2) - \
                     (k[mask_a] ** 2 / (6 * a3[mask_a]) + k[mask_a] / a[mask_a] - a[mask_a] / 2)

        tp1 = -2 * n2 * (b - a) / (n_p ** 2)
        tp2 = -2 * n2 * n_p * np.log(b / a) / (nm ** 2)
        tp3 = n2 * (1 / b - 1 / a) / 2

        denom4 = (2 * n_p * a - nm ** 2)
        denom5 = (2 * n_p * b - nm ** 2)

        tp4 = np.zeros_like(a); tp5 = np.zeros_like(a)
        mask_safe = (denom4 > 1e-9) & (denom5 > 1e-9)
        tp4[mask_safe] = (16 * n2[mask_safe] ** 2 * (n2[mask_safe] ** 2 + 1) * np.log(denom5[mask_safe] / denom4[mask_safe]) / (n_p[mask_safe] ** 3 * nm[mask_safe] ** 2))
        tp5[mask_safe] = (16 * n2[mask_safe] ** 3 * (1 / denom5[mask_safe] - 1 / denom4[mask_safe]) / n_p[mask_safe] ** 3)

        tp = tp1 + tp2 + tp3 + tp4 + tp5
        tav = (ts + tp) / (2 * sa ** 2)

        return np.clip(tav, 0.0, 1.0)


    @staticmethod
    def PROSPECT_5D(leafbio: LeafBiology, optical_params: Dict) -> LeafOptics:
        """核心 PROSPECT-5D/PRO 模型 (光谱计算)"""
        Cab, Cca, Cw, Cdm, Cs, Cant, N, PROT, CBC = leafbio.Cab, leafbio.Cca, \
            leafbio.Cw, leafbio.Cdm, leafbio.Cs, leafbio.Cant, leafbio.N, leafbio.PROT, leafbio.CBC


        # Run PROSPECT-PRO if PROT and/or CBC are non-zero (Cdm = PROT + CBC)
        if (PROT > 0.0 or CBC > 0.0) and Cdm > 0:
            Cdm = 0.0

        nr, Kdm, Kab, Kca, Kw, Ks, Kant, kcbc, kprot = optical_params["nr"], \
            optical_params["Kdm"], optical_params["Kab"], optical_params["Kca"], \
            optical_params["Kw"], optical_params["Ks"], optical_params["Kant"], optical_params["kcbc"], optical_params["kprot"]
        # PROSPECT-PRO optical parameters (Féret et al., 2021) kcbc and kprot

        Kall = (Cab * Kab + Cca * Kca + Cdm * Kdm + Cw * Kw + Cs * Ks + Cant * Kant + CBC * kcbc + PROT * kprot) / N

        j = np.where(Kall > 0)[0]
        t1 = (1 - Kall) * np.exp(-Kall)
        tau = np.ones_like(Kall)

        if j.size > 0:
            t2_approx = -Kall[j]**2 * np.log(Kall[j])
            tau[j] = t1[j] + t2_approx

        kChlrel = np.zeros_like(Kall)
        mask_positive_kall = Kall > 1e-9
        kChlrel[mask_positive_kall] = Cab * Kab[mask_positive_kall] / (Kall[mask_positive_kall] * N)

        # Stokes Equations (N layers)
        t_alph = PROSPECT_Model.calculate_tav(40, nr)
        r_alph = 1 - t_alph
        t12 = PROSPECT_Model.calculate_tav(90, nr)
        r12 = 1 - t12
        t21 = t12 / (nr ** 2)
        r21 = 1 - t21

        denom = 1 - r21 * r21 * tau ** 2
        Ta = t_alph * tau * t21 / denom
        Ra = r_alph + r21 * tau * Ta
        t = t12 * tau * t21 / denom
        r = r12 + r21 * tau * t

        D = np.sqrt(np.clip((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), 1e-9, np.inf))
        a = (1 + r ** 2 - t ** 2 + D) / (2 * r)
        b = (1 - r ** 2 + t ** 2 + D) / (2 * t)

        if N == 1:
            Rsub = np.zeros_like(r)
            Tsub = np.ones_like(t)
        else:
            N_float = float(N)
            bNm1 = b ** (N_float - 1); bN2 = bNm1 ** 2; a2 = a ** 2
            denom_stokes = a2 * bN2 - 1
            mask_denom_safe = np.abs(denom_stokes) > 1e-9
            Rsub, Tsub = np.zeros_like(b), np.zeros_like(b)
            Rsub[mask_denom_safe] = a[mask_denom_safe] * (bN2[mask_denom_safe] - 1) / denom_stokes[mask_denom_safe]
            Tsub[mask_denom_safe] = bNm1[mask_denom_safe] * (a2[mask_denom_safe] - 1) / denom_stokes[mask_denom_safe]

            j_zero_abs = np.where(r + t >= 1)[0]
            Tsub[j_zero_abs] = t[j_zero_abs] / (t[j_zero_abs] + (1 - t[j_zero_abs]) * (N_float - 1))
            Rsub[j_zero_abs] = 1 - Tsub[j_zero_abs]

        denom_full = 1 - Rsub * r
        tran = Ta * Tsub / denom_full
        refl = Ra + Ta * Rsub * t / denom_full

        return LeafOptics(refl=refl, tran=tran, kChlrel=kChlrel)

    @staticmethod
    def calculate_par_absorption(leafopt: LeafOptics, spectral_params: Dict) -> float:
        """计算 PAR 波段的平均叶片吸收系数"""
        Ipar = spectral_params['Ipar']
        refl_par = leafopt.refl[Ipar]
        tran_par = leafopt.tran[Ipar]
        alpha_par_spectral = 1.0 - refl_par - tran_par
        par_absorption_coeff = np.mean(alpha_par_spectral) if alpha_par_spectral.size > 0 else 0.0
        return float(np.clip(par_absorption_coeff, 0.0, 1.0))

    @staticmethod
    def calculate_leaf_properties_prospect(cab, cca, cdm, cw, N_leaf, PROT, CBC, spectral_params, optical_params):
        """运行 PROSPECT 并提取所需的 PAR 吸收系数 (封装)"""
        lb = LeafBiology(Cab=cab, Cca=cca, Cdm=cdm, Cw=cw, Cs=0.0, Cant=0.0, N=N_leaf, PROT=PROT, CBC=CBC)
        prospect_output = PROSPECT_Model.PROSPECT_5D(lb, optical_params)
        k_leaf = PROSPECT_Model.calculate_par_absorption(prospect_output, spectral_params)
        return k_leaf


# ============================================================
# 模块 B: BSM/土壤光学模型 (Soil Optics Module)
# ============================================================

@dataclass
class SoilOptics:
    """土壤反射率输出 (来自 BSM)"""
    refl: np.ndarray # 湿土半球反射率 (光谱数组) (无单位)
    refl_dry: np.ndarray # 干土半球反射率 (光谱数组) (无单位)

@dataclass
class SoilParameters:
    """BSM 输入的土壤特性"""
    B: float # 土壤亮度形状参数 (无单位)
    lat: float # 土壤色度纬度 (度)
    lon: float # 土壤色度经度 (度)
    SMp: float # 土壤含水量百分比 (%)
    SMC: float = 25.0 # 土壤含水饱和容量 (%)
    film: float = 0.0150 # 水膜厚度 (cm)


class BSM_Model:
    """封装 BSM 土壤模型及其辅助函数"""

    @staticmethod
    def soilwat(rdry, nw, kw, SMp, SMC, deleff):
        """BSM 辅助函数: 模拟土壤水膜对土壤反射率的影响。"""
        k_arr = np.array([0, 1, 2, 3, 4, 5, 6])
        nk = len(k_arr)
        mu = (SMp - 5) / SMC

        if mu <= 0:
            return rdry

        # Lekner & Dorf (1988) constants - depends on PROSPECT's calculate_tav
        rbac = 1 - (1 - rdry) * (
            rdry * PROSPECT_Model.calculate_tav(90, 2 / nw) / PROSPECT_Model.calculate_tav(90, 2) + 1 - rdry
        )

        p = 1 - PROSPECT_Model.calculate_tav(90, nw) / nw ** 2
        Rw = 1 - PROSPECT_Model.calculate_tav(40, nw)

        fmul = poisson.pmf(k_arr, mu)
        tw = np.exp(-2 * kw[:, np.newaxis] * deleff * k_arr)

        Rwet_k = Rw[:, np.newaxis] + (1 - Rw[:, np.newaxis]) * (1 - p[:, np.newaxis]) * tw * rbac[:, np.newaxis] / (
            1 - p[:, np.newaxis] * tw * rbac[:, np.newaxis]
        )

        rwet = (rdry * fmul[0]) + (Rwet_k[:, 1:nk] @ fmul[1:nk])
        return rwet

    @staticmethod
    def BSM(soilpar: SoilParameters, optical_params: Dict) -> SoilOptics:
        """运行 BSM (Brightness-Shape-Moisture) 土壤模型。"""
        SMp, SMC, film = soilpar.SMp, soilpar.SMC, soilpar.film
        GSV = optical_params["GSV"]
        kw = optical_params["kw"]
        nw = optical_params["nw"]

        B, lat, lon = soilpar.B, soilpar.lat, soilpar.lon
        lat_rad = lat * np.pi / 180
        lon_rad = lon * np.pi / 180

        f1 = B * np.sin(lat_rad)
        f2 = B * np.cos(lat_rad) * np.sin(lon_rad)
        f3 = B * np.cos(lat_rad) * np.cos(lon_rad)

        rdry = f1 * GSV[:, 0] + f2 * GSV[:, 1] + f3 * GSV[:, 2]
        rwet = BSM_Model.soilwat(rdry, nw, kw, SMp, SMC, film)

        return SoilOptics(refl=rwet, refl_dry=rdry)


# ============================================================
# 模块 C: SAILH/冠层辐射传输模型 (Canopy RTM Module)
# ============================================================

class CanopyReflectances:
    """存储 SAILH 模型计算得到的冠层反射率。"""
    def __init__(self, rso, rdo, rsd, rdd):
        self.rso = rso # 定向-定向反射率
        self.rdo = rdo # 定向-半球反射率
        self.rsd = rsd # 半球-定向反射率
        self.rdd = rdd # 半球-半球反射率

class Angles:
    """存储太阳天顶角、观测天顶角和相对方位角。"""
    def __init__(self, sol_angle, obs_angle, rel_angle):
        self.sol_angle = sol_angle # 太阳天顶角 (度)
        self.obs_angle = obs_angle # 观测天顶角 (度)
        self.rel_angle = rel_angle # 相对方位角 (度)

class CanopyStructure:
    """存储 SAILH 所需的冠层结构属性。"""
    def __init__(self, LAI, LIDFa, LIDFb, q, nlayers=60, nlazi=36):
        self.LAI = float(LAI)
        self.LIDFa = float(LIDFa)
        self.LIDFb = float(LIDFb)
        self.q = float(q) # Hotspot parameter
        self.nlayers = int(nlayers)
        self.nlincl = 13
        self.nlazi = int(nlazi)
        self.lidf = SAILH_Model.calculate_leafangles(LIDFa, LIDFb, self.nlincl)


class SAILH_Model:
    """封装 SAILH 辐射传输模型及其辅助函数"""

    @staticmethod
    def calculate_leafangles(LIDFa, LIDFb, nlincl=13):
        """计算叶倾角分布函数 (LIDF)"""
        def dcum(a, b, theta):
            rd = np.pi / 180
            if LIDFa > 1:
                f = 1 - np.cos(theta * rd)
            else:
                eps = 1e-8; delx = 1; x = 2 * rd * theta; theta2 = x
                while delx > eps:
                    y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
                    dx = 0.5 * (y - x + theta2)
                    x = x + dx; delx = abs(dx)
                f = (2 * y + theta2) / np.pi
            return f

        F = np.zeros(14)
        for i in range(1, 9): F[i] = dcum(LIDFa, LIDFb, i * 10)
        for i in range(9, 13): F[i] = dcum(LIDFa, LIDFb, 80 + (i - 8) * 2)
        F[13] = 1
        lidf = np.diff(F)
        return lidf[:, np.newaxis]

    @staticmethod
    def _volscatt(sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli):
        """几何因子计算 (消光和散射)"""
        nli = len(cos_ttli)
        psi_rad_col = psi_rad * np.ones((nli, 1)); cos_psi = np.cos(psi_rad_col)
        Cs = cos_ttli * cos_tts; Ss = sin_ttli * sin_tts
        Co = cos_ttli * cos_tto; So = sin_ttli * sin_tto
        As = np.maximum(Ss, np.abs(Cs)); Ao = np.maximum(So, np.abs(Co))

        bts = np.arccos(np.clip(-Cs / As, -1.0, 1.0))
        bto = np.arccos(np.clip(-Co / Ao, -1.0, 1.0))

        chi_o = 2 / np.pi * ((bto - np.pi / 2) * Co + np.sin(bto) * So)
        chi_s = 2 / np.pi * ((bts - np.pi / 2) * Cs + np.sin(bts) * Ss)

        delta1 = np.abs(bts - bto)
        delta2 = np.pi - np.abs(bts + bto - np.pi)
        Tot = psi_rad_col + delta1 + delta2

        bt1 = np.minimum(psi_rad_col, delta1)
        bt3 = np.maximum(psi_rad_col, delta2)
        bt2 = Tot - bt1 - bt3

        T1 = 2 * Cs * Co + Ss * So * cos_psi
        T2 = np.sin(bt2) * (2 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

        Jmin = bt2 * T1 - T2
        Jplus = (np.pi - bt2) * T1 + T2

        frho = Jplus / (2 * np.pi ** 2)
        ftau = -Jmin / (2 * np.pi ** 2)

        zeros = np.zeros((nli, 1))
        frho = np.maximum(zeros, frho); ftau = np.maximum(zeros, ftau)
        return chi_s, chi_o, frho, ftau

    @staticmethod
    def Psofunction(x, K, k, LAI, q, dso):
        """APPENDIX IV Pso 函数 (用于热点效应积分)"""
        if dso != 0:
            alpha = (dso / q) * 2 / (k + K)
            pso = np.exp(
                (K + k) * LAI * x
                + np.sqrt(K * k) * LAI / alpha * (1 - np.exp(x * alpha))
            )
        else:
            pso = np.exp((K + k) * LAI * x - np.sqrt(K * k) * LAI * x)
        return pso

    @staticmethod
    def SAILH(soil, leafopt, canopy, angles, constants=None):
        """核心 SAILH 解析模型 (计算反射率和 k_ext)"""
        if constants is None: constants = {'deg2rad': np.pi / 180}
        deg2rad = constants['deg2rad']

        nl = canopy.nlayers
        lidf = canopy.lidf
        LAI = canopy.LAI
        rho = leafopt.refl; tau = leafopt.tran; rs = soil.refl
        tts, tto, psi = angles.sol_angle, angles.obs_angle, angles.rel_angle

        litab = np.array([*range(5, 80, 10), *range(81, 91, 2)])[:, np.newaxis]
        cos_tts = np.cos(tts * deg2rad); cos_tto = np.cos(tto * deg2rad)

        psi = abs(psi - 360 * round(psi / 360)); psi_rad = psi * deg2rad
        sin_tts = np.sin(tts * deg2rad); sin_tto = np.sin(tto * deg2rad)
        tan_tts = np.tan(tts * deg2rad); tan_tto = np.tan(tto * deg2rad)
        dso = np.sqrt(tan_tts ** 2 + tan_tto ** 2 - 2 * tan_tts * tan_tto * np.cos(psi_rad))

        sin_ttli = np.sin(litab * deg2rad); cos_ttli = np.cos(litab * deg2rad)

        chi_s, chi_o, frho, ftau = SAILH_Model._volscatt(
            sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli
        )
        lidf_col = lidf

        k = chi_s.T @ lidf_col / cos_tts
        K = chi_o.T @ lidf_col / cos_tto

        bfli = cos_ttli ** 2; bf = bfli.T @ lidf_col
        sob = frho.T @ lidf_col * np.pi / (cos_tts * cos_tto)
        sof = ftau.T @ lidf_col * np.pi / (cos_tts * cos_tto)

        sdb, sdf = 0.5 * (k + bf), 0.5 * (k - bf)
        ddb, ddf = 0.5 * (1 + bf), 0.5 * (1 - bf)
        dob, dof = 0.5 * (K + bf), 0.5 * (K - bf)

        sigb = ddb * rho + ddf * tau; sigf = ddf * rho + ddb * tau
        sb = sdb * rho + sdf * tau; sf = sdf * rho + sdb * tau
        vb = dob * rho + dof * tau; vf = dof * rho + dob * tau
        w = sob * rho + sof * tau; a = 1 - sigf

        m = np.sqrt(np.clip(a ** 2 - sigb ** 2, 1e-9, np.inf))
        rinf = (a - m) / sigb; rinf2 = rinf ** 2
        e1 = np.exp(-m * LAI); e2 = e1 ** 2; re = rinf * e1
        denom_rhotau = 1 - rinf2 * e2

        tau_dd = (1 - rinf2) * e1 / denom_rhotau
        rho_dd = rinf * (1 - e2) / denom_rhotau

        s1 = sf + rinf * sb; s2 = sf * rinf + sb
        v1 = vf + rinf * vb; v2 = vf * rinf + vb

        k_minus_m = k.item() - m; k_plus_m = k.item() + m
        Pss_term = s1 * (1 - e1) / k_minus_m
        Qss_term = s2 * (1 - re) / k_plus_m

        tau_sd = (Pss_term - re * Qss_term) / denom_rhotau
        rho_sd = (Qss_term - re * Pss_term) / denom_rhotau

        tau_oo = np.exp(-K.item() * LAI)
        tau_do = (v1 * (1 - e1) / (K.item() - m) - re * (v2 * (1 - re) / (K.item() + m))) / denom_rhotau
        rho_do = (v2 * (1 - re) / (K.item() + m) - re * (v1 * (1 - e1) / (K.item() - m))) / denom_rhotau

        xl_vec = np.linspace(0, LAI, nl + 1)[:, np.newaxis]
        dx = LAI / nl; iLAI = LAI * (1 / nl)
        K_geo = K.item(); k_geo = k.item()

        Pso_prob = np.zeros_like(xl_vec)
        for idx, x_level in enumerate(xl_vec):
            x_val = x_level if idx == 0 else x_level - dx / 2
            Pso_prob[idx] = SAILH_Model.Psofunction(x_val, K_geo, k_geo, LAI, canopy.q, dso)

        Pso_clamped = np.minimum(Pso_prob.reshape(nl + 1, 1), 1.0)
        rho_so = w * np.sum(Pso_clamped[:nl]) * iLAI
        tau_ss_scalar = np.mean(np.exp(-k_geo * LAI * iLAI))

        denom_full = 1 - rs * rho_dd
        rso = rho_so + rho_dd * rs * tau_dd / denom_full
        rdo = rho_do + (tau_oo + tau_do) * rs * tau_dd / denom_full
        rsd = rho_sd + (tau_ss_scalar + tau_sd) * rs * tau_dd / denom_full
        rdd = rho_dd + tau_dd * rs * tau_dd / denom_full

        return CanopyReflectances(rso, rdo, rsd, rdd), k.item()


# ============================================================
# 模块 D: LayeredSCOPE 核心/生物物理通量 (LayeredSCOPE Core)
# ============================================================

class LayeredSCOPE_Core:
    """封装所有叶片级别的通量计算函数"""

    @staticmethod
    def par_from_sw(sw_down):
        """将短波辐射 (SWdown) 转换为 PAR。"""
        return sw_down * 4.6

    @staticmethod
    def farquhar_a(ci, tleaf, v_cmax25, j_max25, par_leaf, rd25=1.0):
        """Farquhar 光合作用模型 (FvCB 简化)"""
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
        A_net = min(Ac, Aj) - Rd
        return A_net

    @staticmethod
    def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
        """Ball-Berry 气孔导度模型"""
        A_mol = A_net * 1e-6
        gs = gs0 + m * (A_mol * np.clip(rh, 0, 1)) / max(cs * 1e-6, 1e-9)
        return max(gs, 1e-6)

    @staticmethod
    def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
        """叶片能量平衡方程求解 (fsolve)"""
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
            t = float(np.asarray(tleaf_array).ravel()[0])
            H = rho_air * cp * (t - tair) * gb
            LE = lambda_v * g_v * 1e-3
            return np.array(Rn - (H + LE))

        try:
            tleaf_solution, infodict, ier, mesg = fsolve(residual, np.array([tleaf_guess]), full_output=True)
            tleaf_final = float(np.asarray(tleaf_solution).ravel()[0]) if ier == 1 else tair + 1.0
        except Exception:
            tleaf_final = tair + 1.0

        H = rho_air * cp * (tleaf_final - tair) * gb
        LE = lambda_v * g_v * 1e-3
        return tleaf_final, H, LE

    @staticmethod
    def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
        """叶片太阳诱导叶绿素荧光 (SIF) 量子效率模型"""
        par_abs_leaf = float(par_abs_leaf)
        A_leaf = float(A_leaf)
        phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
        npq = 0.3
        phi_F = phi_fmax * (1 - phi_P - npq)
        phi_F = np.clip(phi_F, 0, phi_fmax)
        return phi_F * par_abs_leaf


class LayeredSCOPE:
    """分层 SCOPE 模型简化版 (主类)"""
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10,
                 cab=40, cca=10, cdm=0.01, cw=0.01, N_leaf=1.5, PROT=0.0, CBC=0.0,
                 LIDFa=0.0, LIDFb=0.0, q=0.1, sol_angle=30.0, obs_angle=0.0, rel_angle=180.0,
                 SMp=15.0, B=0.5, lat=0.0, lon=100.0,
                 spectral_params=None, optical_params=None):
        self.lai = float(lai); self.v_cmax25 = float(v_cmax25); self.j_max25 = float(j_max25); self.n_layers = int(n_layers)
        self.cab, self.cca, self.cdm, self.cw = float(cab), float(cca), float(cdm), float(cw)
        self.N_leaf, self.PROT, self.CBC = float(N_leaf), float(PROT), float(CBC)
        self.LIDFa, self.LIDFb, self.q = float(LIDFa), float(LIDFb), float(q)
        self.sol_angle, self.obs_angle, self.rel_angle = float(sol_angle), float(obs_angle), float(rel_angle)
        self.SMp, self.B, self.lat, self.lon = float(SMp), float(B), float(lat), float(lon)

        self.spectral_params = spectral_params if spectral_params is not None else self._mock_spectral()
        self.optical_params = optical_params if optical_params is not None else self._make_optical_params()

        self.canopy_structure = CanopyStructure(self.lai, self.LIDFa, self.LIDFb, self.q, self.n_layers)
        self.angles = Angles(self.sol_angle, self.obs_angle, self.rel_angle)

    def _mock_spectral(self):
        wl = np.linspace(400, 2400, 100)
        Ipar = np.where((wl >= 400) & (wl <= 700))[0]
        return {'wl_spectrum': wl, 'Ipar': Ipar}

    def _make_optical_params(self):
        nwl = self._mock_spectral()['wl_spectrum'].size
        return {
            "nr": np.linspace(1.2, 1.4, nwl), "Kdm": np.full(nwl, 0.01), "Kab": 0.5 * np.exp(-0.01 * np.arange(nwl)),
            "Kca": np.full(nwl, 0.05), "Kw": np.full(nwl, 0.01), "Ks": np.full(nwl, 0.001), "Kant": np.full(nwl, 0.1),
            "kcbc": np.full(nwl, 0.005), "kprot": np.full(nwl, 0.005), "GSV": np.ones((nwl, 3)) * 0.3,
            "nw": np.linspace(1.3, 1.33, nwl), "kw": np.zeros(nwl),
        }

    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        """运行分层 SCOPE 模型进行瞬时通量计算"""
        n = len(tair)
        H_series = np.zeros(n); LE_series = np.zeros(n); GPP_series = np.zeros(n); SIF_series = np.zeros(n)
        tleaf_prev = float(np.asarray(tair).ravel()[0]); lai_layer = self.lai / self.n_layers

        # 1. PROSPECT: Leaf absorption coefficient
        k_leaf_absorption = PROSPECT_Model.calculate_leaf_properties_prospect(
            self.cab, self.cca, self.cdm, self.cw, self.N_leaf, self.PROT, self.CBC,
            self.spectral_params, self.optical_params
        )

        # 2. BSM: Soil Reflectance
        soilpar = SoilParameters(self.B, self.lat, self.lon, self.SMp)
        soilopt_bsm = BSM_Model.BSM(soilpar, self.optical_params)

        # 3. SAILH: Canopy Extinction Coefficient
        leafopt_sail = LeafOptics(self.optical_params['Kab'], self.optical_params['Kw'], self.optical_params['Kab'])
        class MockSoil: # Mock class for SAILH input
             def __init__(self, refl): self.refl = refl
        soilopt_mock = MockSoil(refl=soilopt_bsm.refl)

        _, k_ext_solar = SAILH_Model.SAILH(soilopt_mock, leafopt_sail, self.canopy_structure, self.angles)
        k_ext = k_ext_solar

        for i in range(n):
            tair_i = float(np.asarray(tair[i]).ravel()[0]); sw_i = float(np.asarray(sw_down[i]).ravel()[0])
            rh_i = float(np.asarray(rh[i]).ravel()[0]); wind_i = float(np.asarray(wind[i]).ravel()[0])

            par_layer = LayeredSCOPE_Core.par_from_sw(sw_i)
            H_canopy = LE_canopy = GPP_canopy = SIF_canopy = 0.0

            for l in range(self.n_layers):
                par_abs_layer = par_layer * (1 - np.exp(-k_ext * lai_layer))
                par_abs_leaf = par_abs_layer * k_leaf_absorption
                tleaf = tleaf_prev

                for _ in range(3):
                    A_net = LayeredSCOPE_Core.farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = LayeredSCOPE_Core.ball_berry(A_net, rh_i, cs=co2)
                    tleaf, H_layer, LE_layer = LayeredSCOPE_Core.energy_balance_leaf(max(tleaf, tair_i+0.1), tair_i, par_abs_leaf, gs, wind_i)

                F_leaf = LayeredSCOPE_Core.leaf_sif(par_abs_leaf, A_net)

                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k_ext * lai_layer * (self.n_layers-l-1))
                H_canopy += H_layer * (lai_layer / max(self.lai, 1e-6))
                LE_canopy += LE_layer * (lai_layer / max(self.lai, 1e-6))

                par_layer -= par_abs_layer
                par_layer = max(0, par_layer)
                tleaf_prev = tleaf

            H_series[i], LE_series[i], GPP_series[i], SIF_series[i] = H_canopy, LE_canopy, GPP_canopy, SIF_canopy

        return {"H": H_series, "LE": LE_series, "GPP": GPP_series, "SIF": SIF_series}


# ============================================================
# 模块 E: DALEC/碳循环模型 (Carbon Cycle Module)
# ============================================================

class DALEC:
    """简化的 DALEC 碳循环模型"""
    def __init__(self,
                 C_leaf=200.0, C_wood=5000.0, C_root=500.0, C_litter=2000.0, C_soil=10000.0,
                 tau_leaf=365.0, tau_wood=1.0/0.0001, tau_root=1.0/0.002, tau_litter=1.0/0.008, tau_soil=1.0/0.00005,
                 f_a=0.3, f_f=0.4, f_r=0.2, f_l=0.1, LMA=50.0):
        self.P_RATES = {
            'f_a': f_a, 'f_f': f_f, 'f_r': f_r, 'f_l': f_l,
            't_litter_rate': 1.0/tau_litter, 't_som_rate': 1.0/tau_soil,
            't_wood_rate': 1.0/tau_wood, 't_root_rate': 1.0/tau_root, 'LMA': LMA
        }
        self.C = {'leaf': C_leaf, 'wood': C_wood, 'root': C_root, 'litter': C_litter, 'soil': C_soil, 'labile': 50.0}
        self.tau = {'leaf': tau_leaf, 'wood': tau_wood, 'root': tau_root, 'litter': tau_litter, 'soil': tau_soil, 'labile': 10.0}

    def update(self, GPP, Tair):
        """DALEC 模型的日步长更新"""
        deltat = 1.0
        GPP = float(GPP); Tair = float(Tair)
        GPP_gC = GPP * 12e-6 * 3600 * 24

        # 1. Ra and NPP
        Ra = self.P_RATES['f_a'] * GPP_gC
        NPP = GPP_gC - Ra

        # 2. Allocation
        P_leaf = NPP * self.P_RATES['f_f']
        P_root = (NPP - P_leaf) * self.P_RATES['f_r']
        P_labile = (NPP - P_leaf - P_root) * self.P_RATES['f_l']
        P_wood = NPP - P_leaf - P_root - P_labile

        # 3. Turnover Rates
        k_labile = 1.0 / self.tau['labile']; k_leaf = 1.0 / self.tau['leaf']
        k_wood = self.P_RATES['t_wood_rate']; k_root = self.P_RATES['t_root_rate']
        k_litter = self.P_RATES['t_litter_rate']; k_som = self.P_RATES['t_som_rate']

        # 4. Turnover Fluxes
        T_labile = self.C['labile'] * k_labile * deltat; T_leaf = self.C['leaf'] * k_leaf * deltat
        T_wood = self.C['wood'] * k_wood * deltat; T_root = self.C['root'] * k_root * deltat
        T_litter = self.C['litter'] * k_litter * deltat; T_som = self.C['soil'] * k_som * deltat

        # 5. Pool Update
        dC_labile = P_labile - T_labile
        dC_leaf = P_leaf + T_labile - T_leaf
        dC_root = P_root - T_root
        dC_wood = P_wood - T_wood
        dC_litter = T_leaf + T_root - T_litter
        dC_soil = T_litter + T_wood - T_som

        self.C['labile'] += dC_labile; self.C['leaf'] += dC_leaf; self.C['root'] += dC_root
        self.C['wood'] += dC_wood; self.C['litter'] += dC_litter; self.C['soil'] += dC_soil

        for k in self.C: self.C[k] = max(0.0, self.C[k])

        # 6. Reco and LAI output
        Reco = Ra + T_litter + T_som
        Reco_umol = Reco / (12e-6 * 3600 * 24)
        LAI = max(0.1, self.C['leaf'] / self.P_RATES['LMA'])

        return Reco_umol, LAI


# ============================================================
# 模块 F: SCOPE-DALEC 集成/同化 (Integration and Assimilation)
# ============================================================

class Assimilation_Tools:
    """封装同化模型中使用的辅助工具函数"""
    @staticmethod
    def LAI_to_NDVI(lai, a=0.8, b=0.5):
        """简单的 LAI 到 NDVI 转换模型"""
        return a * (1 - np.exp(-b * np.asarray(lai)))


class SCOPE_DALEC:
    """SCOPE-DALEC 集成模型 (主控类)"""
    def __init__(self, scope_params, dalec_params, optical_params = None):
        self.scope = LayeredSCOPE(**scope_params, optical_params = optical_params)

        # Initializing the DALEC model
        self.dalec = DALEC(
            C_leaf=scope_params.get('C_leaf', 200.0), C_wood=scope_params.get('C_wood', 5000.0),
            C_root=scope_params.get('C_root', 500.0), C_litter=scope_params.get('C_litter', 2000.0),
            C_soil=scope_params.get('C_soil', 10000.0),
            tau_leaf=365.0, tau_wood=1.0/0.0001, tau_root=1.0/0.002, tau_litter=1.0/0.008, tau_soil=1.0/0.00005,
            f_a=0.3, f_f=0.4, f_r=0.2, f_l=0.1, LMA=50.0
        )
        self.results = {'GPP': [], 'Reco': [], 'NEE': [], 'H': [], 'LE': [], 'SIF': [], 'LAI': []}

    def run(self, met):
        """运行 SCOPE-DALEC 耦合模型"""
        tair, sw_down, rh, vpd, wind = met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind']
        self.results = {k: [] for k in self.results}
        
        for i in range(len(tair)):
            # 1. Calculate instantaneous fluxes based on current LAI
            scope_out = self.scope.run_time_series(
                np.array([tair[i]]), np.array([sw_down[i]]), np.array([rh[i]]), np.array([vpd[i]]), np.array([wind[i]])
            )
            GPP = float(scope_out['GPP'][0]); H = float(scope_out['H'][0]); LE = float(scope_out['LE'][0]); SIF = float(scope_out['SIF'][0])

            # 2. DALEC update (daily step)
            Reco, LAI_new = self.dalec.update(GPP, tair[i])
            NEE = Reco - GPP

            # 3. Update SCOPE's LAI for the next iteration
            self.scope.lai = LAI_new
            self.scope.canopy_structure = CanopyStructure(LAI_new, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            for k, v in zip(['GPP','Reco','NEE','H','LE','SIF','LAI'], [GPP,Reco,NEE,H,LE,SIF,LAI_new]):
                self.results[k].append(v)
        return self.results

    def assimilate_ndvi(self, met, ndvi_obs, param_names=['cab','v_cmax25','j_max25'],
                          x0=None, B_inv=None, R_inv=None):
        """基于 4D-Var 的 NDVI 数据同化"""
        if x0 is None: x0 = np.array([self.scope.cab, self.scope.v_cmax25, self.scope.j_max25])
        if B_inv is None: B_inv = np.eye(len(x0))
        if R_inv is None: R_inv = np.eye(len(ndvi_obs))

        def cost_function(params):
            original_dalec_C = {k: v for k,v in self.dalec.C.items()}
            original_scope_lai = self.scope.lai

            for n,p in zip(param_names, params): setattr(self.scope, n, float(p))

            # 重新运行 SCOPE/DALEC 流程
            out_scope = self.scope.run_time_series(met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind'])
            self.dalec.C = original_dalec_C.copy()
            out_dalec_lai = []

            for i in range(len(met['Tair'])):
                GPP_i = float(out_scope['GPP'][i])
                _, LAI = self.dalec.update(GPP_i, met['Tair'][i])
                out_dalec_lai.append(LAI)
                self.scope.canopy_structure = CanopyStructure(LAI, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            ndvi_model = Assimilation_Tools.LAI_to_NDVI(out_dalec_lai)
            resid = ndvi_model - ndvi_obs
            J_obs = 0.5 * resid.T @ R_inv @ resid
            delta = np.array(params) - x0
            J_b = 0.5 * delta.T @ B_inv @ delta

            self.dalec.C = original_dalec_C
            self.scope.lai = original_scope_lai
            self.scope.canopy_structure = CanopyStructure(original_scope_lai, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            return float(J_obs + J_b)

        res = minimize(cost_function, x0, bounds=[(0,80),(20,150),(50,300)])
        for n,p in zip(param_names, res.x): setattr(self.scope, n, float(p))
        return res


# ============================================================
# 示例运行 (保持不变)
# ============================================================

if __name__ == "__main__":
    days = 30
    met = {
        "Tair": np.linspace(15,30,days), "SWdown": np.linspace(0,800,days),
        "RH": np.linspace(0.5,0.8,days), "VPD": np.linspace(0.5,2.0,days),
        "Wind": np.linspace(0.5,3.0,days)
    }

    # PROSPECT and SAILH parameters
    scope_params = {"lai":10.0, "v_cmax25":70, "j_max25":140, "n_layers":10,
                    "cab":40, "cca":10, "cdm":0.01, "cw":0.01, "N_leaf": 1.5, "PROT": 0.001, "CBC": 0.005,
                    "LIDFa": 0.0, "LIDFb": 0.0, "q": 0.1, "sol_angle": 30.0, "obs_angle": 0.0, "rel_angle": 180.0}
    dalec_params = {}

    model = SCOPE_DALEC(scope_params, dalec_params)

    print("--- 运行集成模型 (PROSPECT-PRO/SAILH/LayeredSCOPE) ---")
    out = model.run(met)

    print("\n优化前的模拟结果:")
    for k,v in out.items():
        print(f"{k}: {np.round(v[:5],2)}")

    # 4D-Var Assimilation Example
    ndvi_obs = np.linspace(0.3,0.7,days)
    R_inv = np.eye(days)/0.05**2
    B_inv = np.eye(3)/np.array([10,10,20])**2

    print("\n--- 启动 4D-Var NDVI 同化 ---")
    res = model.assimilate_ndvi(met, ndvi_obs, x0=[40,70,140], B_inv=B_inv, R_inv=R_inv)

    print("优化后的参数:")
    print(f"Cab: {model.scope.cab:.2f}, Vcmax25: {model.scope.v_cmax25:.2f}, Jmax25: {model.scope.j_max25:.2f}")