import numpy as np
from scipy.optimize import fsolve, minimize
import scipy.integrate as integrate
from dataclasses import dataclass
from typing import Dict, List, Any
from scipy.stats import poisson

# -------------------------------
# 核心常数定义
# -------------------------------
R = 8.314
KELVIN = 273.15
H_PLANCK = 6.626e-34
C_LIGHT = 2.998e8

# ============================================================
# PROSPECT-PRO 模型类
# ============================================================

@dataclass
class LeafBiology:
    """叶片生物参数 (用于 PROSPECT 输入)"""
    Cab: float
    Cdm: float
    Cw: float
    Cs: float
    Cca: float
    Cant: float
    N: float
    PROT: float = 0.0
    CBC: float = 0.0
    rho_thermal: float = 0.01
    tau_thermal: float = 0.01


@dataclass
class LeafOptics:
    """叶片光学输出 (来自 PROSPECT)"""
    refl: np.ndarray
    tran: np.ndarray
    kChlrel: np.ndarray
    # 增加属性以返回计算后的 PAR 吸收系数，供 LayeredSCOPE 使用
    par_absorption_coeff: float = 0.0


# --- PROSPECT 辅助函数 ---

def calculate_tav(alpha, nr):
    """计算介电平面平均透射率 (用于 PROSPECT)"""
    rd = np.pi / 180

    # --- FIX: 强制 nr 为 1D 数组，确保所有派生变量也是 1D 数组 ---
    nr = np.atleast_1d(nr)

    n2 = nr ** 2
    n_p = n2 + 1
    nm = n2 - 1
    a = (nr + 1) * (nr + 1) / 2
    k = -(n2 - 1) * (n2 - 1) / 4
    sa = np.sin(alpha * rd)

    # --- 计算 b, b1, b2 ---
    b1 = 0
    if alpha != 90:
        b1 = np.sqrt(np.clip((sa ** 2 - n_p / 2) ** 2 + k, 0, np.inf))
    b2 = sa ** 2 - n_p / 2
    b = b1 - b2

    # --- 计算 a³ 和 b³ (现已保证 b, a 为 1D) ---
    a3 = a ** 3
    b3 = b ** 3

    ts = np.zeros_like(a)
    mask_a = a > 1e-9

    # Calculation for ts: (k^2 / (6 * b^3) + k / b - b / 2) - (k^2 / (6 * a^3) + k / a - a / 2)
    # 所有的 k, b, a 等变量现在都是 1D 数组，可以安全地使用 mask_a 进行切片。
    ts[mask_a] = (k[mask_a] ** 2 / (6 * b3[mask_a]) + k[mask_a] / b[mask_a] - b[mask_a] / 2) - \
                 (k[mask_a] ** 2 / (6 * a3[mask_a]) + k[mask_a] / a[mask_a] - a[mask_a] / 2)

    # Calculation for tp (The rest of the complex logic)
    tp1 = -2 * n2 * (b - a) / (n_p ** 2)
    tp2 = -2 * n2 * n_p * np.log(b / a) / (nm ** 2)
    tp3 = n2 * (1 / b - 1 / a) / 2

    denom4 = (2 * n_p * a - nm ** 2)
    denom5 = (2 * n_p * b - nm ** 2)

    tp4 = np.zeros_like(a)
    tp5 = np.zeros_like(a)

    mask_safe = (denom4 > 1e-9) & (denom5 > 1e-9)

    tp4[mask_safe] = (16 * n2[mask_safe] ** 2 * (n2[mask_safe] ** 2 + 1) * np.log(denom5[mask_safe] / denom4[mask_safe]) / (n_p[mask_safe] ** 3 * nm[mask_safe] ** 2))
    tp5[mask_safe] = (16 * n2[mask_safe] ** 3 * (1 / denom5[mask_safe] - 1 / denom4[mask_safe]) / n_p[mask_safe] ** 3)

    tp = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa ** 2)

    return np.clip(tav, 0.0, 1.0)


def PROSPECT_5D(leafbio: LeafBiology, optical_params: Dict) -> LeafOptics:
    """
    PROSPECT-5D/PRO model (Spectrally based calculation).
    """
    Cab, Cca, Cw, Cdm, Cs, Cant, N = leafbio.Cab, leafbio.Cca, leafbio.Cw, leafbio.Cdm, leafbio.Cs, leafbio.Cant, leafbio.N
    PROT, CBC = leafbio.PROT, leafbio.CBC

    if PROT > 0.0 or CBC > 0.0:
        if Cdm > 0: Cdm = 0.0

    nr = optical_params["nr"]
    Kdm, Kab, Kca, Kw, Ks, Kant = optical_params["Kdm"], optical_params["Kab"], optical_params["Kca"], optical_params["Kw"], optical_params["Ks"], optical_params["Kant"]
    kcbc, kprot = optical_params["kcbc"], optical_params["kprot"]

    # Compact leaf layer absorption (Kall)
    Kall = (Cab * Kab + Cca * Kca + Cdm * Kdm + Cw * Kw + Cs * Ks + Cant * Kant + CBC * kcbc + PROT * kprot) / N

    # Non-conservative scattering (Tau - single layer transmittance)
    j = np.where(Kall > 0)[0]
    t1 = (1 - Kall) * np.exp(-Kall)

    tau = np.ones_like(Kall)

    if j.size > 0:
        # Mocking the numerical integration for t2: t2 = Kall^2 * Ei(Kall)
        t2_approx = -Kall[j]**2 * np.log(Kall[j])
        tau[j] = t1[j] + t2_approx

    kChlrel = np.zeros_like(Kall)
    mask_positive_kall = Kall > 1e-9
    kChlrel[mask_positive_kall] = Cab * Kab[mask_positive_kall] / (Kall[mask_positive_kall] * N)

    # --- Stokes Equations (N layers) ---
    t_alph = calculate_tav(40, nr)
    r_alph = 1 - t_alph
    t12 = calculate_tav(90, nr)
    r12 = 1 - t12
    t21 = t12 / (nr ** 2)
    r21 = 1 - t21

    # Single elementary layer (Ra, Ta, r, t)
    denom = 1 - r21 * r21 * tau ** 2
    Ta = t_alph * tau * t21 / denom
    Ra = r_alph + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    # Stokes equations for N-1 layers
    D = np.sqrt(np.clip((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), 1e-9, np.inf))
    a = (1 + r ** 2 - t ** 2 + D) / (2 * r)
    b = (1 - r ** 2 + t ** 2 + D) / (2 * t)

    if N == 1:
        Rsub = np.zeros_like(r)
        Tsub = np.ones_like(t)
    else:
        N_float = float(N)
        bNm1 = b ** (N_float - 1); bN2 = bNm1 ** 2; a2 = a ** 2
        denom = a2 * bN2 - 1
        mask_denom_safe = np.abs(denom) > 1e-9
        Rsub, Tsub = np.zeros_like(b), np.zeros_like(b)
        Rsub[mask_denom_safe] = a[mask_denom_safe] * (bN2[mask_denom_safe] - 1) / denom[mask_denom_safe]
        Tsub[mask_denom_safe] = bNm1[mask_denom_safe] * (a2[mask_denom_safe] - 1) / denom[mask_denom_safe]

        j_zero_abs = np.where(r + t >= 1)[0]
        Tsub[j_zero_abs] = t[j_zero_abs] / (t[j_zero_abs] + (1 - t[j_zero_abs]) * (N_float - 1))
        Rsub[j_zero_abs] = 1 - Tsub[j_zero_abs]

    # Reflectance and transmittance of the full leaf
    denom_full = 1 - Rsub * r
    tran = Ta * Tsub / denom_full
    refl = Ra + Ta * Rsub * t / denom_full

    leafopt_out = LeafOptics(refl=refl, tran=tran, kChlrel=kChlrel)
    return leafopt_out


def calculate_par_absorption(leafopt: LeafOptics, spectral_params: Dict) -> float:
    """Calculates the mean PAR absorption coefficient from PROSPECT output."""
    Ipar = spectral_params['Ipar']
    refl_par = leafopt.refl[Ipar]
    tran_par = leafopt.tran[Ipar]
    alpha_par_spectral = 1.0 - refl_par - tran_par

    par_absorption_coeff = np.mean(alpha_par_spectral) if alpha_par_spectral.size > 0 else 0.0

    return float(np.clip(par_absorption_coeff, 0.0, 1.0))


def calculate_leaf_properties_prospect(cab, cca, cdm, cw, N_leaf, PROT, CBC, spectral_params, optical_params):
    """Runs PROSPECT and extracts the required PAR absorption coefficient."""
    lb = LeafBiology(Cab=cab, Cca=cca, Cdm=cdm, Cw=cw, Cs=0.0, Cant=0.0, N=N_leaf, PROT=PROT, CBC=CBC)
    prospect_output = PROSPECT_5D(lb, optical_params)
    k_leaf = calculate_par_absorption(prospect_output, spectral_params)
    return k_leaf


# ============================================================
# BSM SOIL MODEL INTEGRATION (PART 1: STRUCTURES & LOGIC)
# ============================================================

@dataclass
class SoilOptics:
    """Soil Reflectance Output (from BSM)"""
    refl: np.ndarray
    refl_dry: np.ndarray


@dataclass
class SoilParameters:
    """Soil Characteristics for BSM Input"""
    B: float
    lat: float
    lon: float
    SMp: float
    SMC: float = 25.0
    film: float = 0.0150


def soilwat(rdry, nw, kw, SMp, SMC, deleff):
    """BSM helper: Model soil water effects on soil reflectance (soilwat.m)."""
    k_arr = np.array([0, 1, 2, 3, 4, 5, 6])
    nk = len(k_arr)
    mu = (SMp - 5) / SMC

    if mu <= 0:
        return rdry

    # Lekner & Dorf (1988) constants
    # rbac: Reflectance at bottom of film surface (r-back)
    rbac = 1 - (1 - rdry) * (
        rdry * calculate_tav(90, 2 / nw) / calculate_tav(90, 2) + 1 - rdry
    )

    # total reflectance at bottom of water film surface
    p = 1 - calculate_tav(90, nw) / nw ** 2

    # reflectance of water film top surface
    Rw = 1 - calculate_tav(40, nw)

    # Poisson Distribution for k films
    fmul = poisson.pmf(k_arr, mu)

    # Tw: Transmittance through k films
    # FIX: Ensure kw is a column vector (100, 1) for broadcasting with k_arr (7,) -> Result (100, 7)
    tw = np.exp(-2 * kw[:, np.newaxis] * deleff * k_arr)

    # R(wet, k): Total reflectance for k water films (from top, reflected off dry soil, transmitted back)
    Rwet_k = Rw[:, np.newaxis] + (1 - Rw[:, np.newaxis]) * (1 - p[:, np.newaxis]) * tw * rbac[:, np.newaxis] / (
        1 - p[:, np.newaxis] * tw * rbac[:, np.newaxis]
    )

    # Final wet reflectance: R_dry * P(k=0) + Sum(Rwet_k * P(k>0))
    rwet = (rdry * fmul[0]) + (Rwet_k[:, 1:nk] @ fmul[1:nk])

    return rwet

def BSM(soilpar: SoilParameters, optical_params: Dict) -> SoilOptics:
    """Run the BSM (Brightness-Shape-Moisture) soil model."""

    SMp, SMC, film = soilpar.SMp, soilpar.SMC, soilpar.film

    GSV = optical_params["GSV"]
    kw = optical_params["Kw"]
    nw = optical_params["nw"]

    # 1. Calculate dry soil reflectance (rdry) from GSV
    B, lat, lon = soilpar.B, soilpar.lat, soilpar.lon
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    f1 = B * np.sin(lat_rad)
    f2 = B * np.cos(lat_rad) * np.sin(lon_rad)
    f3 = B * np.cos(lat_rad) * np.cos(lon_rad)

    # GSV is [nwl, 3]. dot product requires GSV to be [nwl, 3] and factors [3]
    rdry = f1 * GSV[:, 0] + f2 * GSV[:, 1] + f3 * GSV[:, 2]

    # 2. Apply water effects
    rwet = soilwat(rdry, nw, kw, SMp, SMC, film)

    return SoilOptics(refl=rwet, refl_dry=rdry)


# ============================================================
# SAILH MODEL INTEGRATION (PART 1: STRUCTURES & HELPERS)
# ============================================================

class CanopyReflectances:
    """Class to hold canopy reflectances computed by the SAILH model."""
    def __init__(self, rso, rdo, rsd, rdd):
        self.rso = rso
        self.rdo = rdo
        self.rsd = rsd
        self.rdd = rdd


class Angles:
    """Class to hold solar zenith, observation zenith, and relative azimuth angles."""
    def __init__(self, sol_angle, obs_angle, rel_angle):
        self.sol_angle = sol_angle
        self.obs_angle = obs_angle
        self.rel_angle = rel_angle


def calculate_leafangles(LIDFa, LIDFb, nlincl=13):
    """
    Calculate the Leaf Inclination Distribution Function (LIDF)
    using the method from the original SCOPE model.
    """
    def dcum(a, b, theta):
        # Calculate cumulative distribution
        rd = np.pi / 180
        if LIDFa > 1:
            f = 1 - np.cos(theta * rd)
        else:
            eps = 1e-8
            delx = 1
            x = 2 * rd * theta
            theta2 = x
            while delx > eps:
                y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
                dx = 0.5 * (y - x + theta2)
                x = x + dx
                delx = abs(dx)
            f = (2 * y + theta2) / np.pi
        return f

    # Define the 13 standard SAIL inclination angles
    # F sized to 14 entries so diff for actual LIDF becomes 13 entries
    F = np.zeros(14)
    for i in range(1, 9):
        theta = i * 10
        F[i] = dcum(LIDFa, LIDFb, theta)
    for i in range(9, 13):
        theta = 80 + (i - 8) * 2
        F[i] = dcum(LIDFa, LIDFb, theta)
    F[13] = 1

    lidf = np.diff(F)
    # Ensure it returns the correct shape (column vector if needed for dot product)
    return lidf[:, np.newaxis]


class CanopyStructure:
    """Class to hold canopy properties for SAILH."""
    def __init__(self, LAI, LIDFa, LIDFb, q, nlayers=60, nlazi=36):
        self.LAI = float(LAI)
        self.LIDFa = float(LIDFa)
        self.LIDFb = float(LIDFb)
        self.q = float(q) # Hotspot parameter
        self.nlayers = int(nlayers)
        self.nlincl = 13 # Fixed by SAIL definition
        self.nlazi = int(nlazi)
        self.lidf = calculate_leafangles(LIDFa, LIDFb, self.nlincl)


def _volscatt(sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli):
    """Geometric factors (extinction and scattering) for SAIL."""

    nli = len(cos_ttli)
    psi_rad_col = psi_rad * np.ones((nli, 1))
    cos_psi = np.cos(psi_rad_col)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    # As, Ao: max([Ss, Cs], [], 2) in MATLAB -> max across the two columns/values
    As = np.maximum(Ss, np.abs(Cs))
    Ao = np.maximum(So, np.abs(Co))

    # Avoid division by zero/NaN if As or Ao is zero (clip input to acos)
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

    # Enforce non-negativity (max([zeros(nli,1), frho], [], 2))
    zeros = np.zeros((nli, 1))
    frho = np.maximum(zeros, frho)
    ftau = np.maximum(zeros, ftau)

    # Return column vectors
    return chi_s, chi_o, frho, ftau


def Psofunction(x, K, k, LAI, q, dso):
    """APPENDIX IV function Pso (used for Hot Spot integration)."""
    if dso != 0:
        alpha = (dso / q) * 2 / (k + K)
        pso = np.exp(
            (K + k) * LAI * x
            + np.sqrt(K * k) * LAI / alpha * (1 - np.exp(x * alpha))
        )
    else:
        pso = np.exp((K + k) * LAI * x - np.sqrt(K * k) * LAI * x)
    return pso


def SAILH(soil, leafopt, canopy, angles, constants=None):
    """
    Core SAILH Analytical Model. Calculates analytical canopy reflectances
    (rso, rdo, rsd, rdd) and the extinction coefficient (k).
    """
    if constants is None: constants = {'deg2rad': np.pi / 180}

    deg2rad = constants['deg2rad']

    # Input structures assumed to be instances of the classes defined above
    nl = canopy.nlayers
    lidf = canopy.lidf
    LAI = canopy.LAI

    rho = leafopt.refl
    tau = leafopt.tran
    rs = soil.refl # Spectral soil reflectance

    tts, tto, psi = angles.sol_angle, angles.obs_angle, angles.rel_angle

    # --- 0. Geometry Setup ---
    litab = np.array([*range(5, 80, 10), *range(81, 91, 2)])[:, np.newaxis] # SAIL standard angles [13, 1]

    cos_tts = np.cos(tts * deg2rad)
    cos_tto = np.cos(tto * deg2rad)

    # Ensures symmetry at 90 and 270 deg
    psi = abs(psi - 360 * round(psi / 360))
    psi_rad = psi * deg2rad

    sin_tts = np.sin(tts * deg2rad)
    sin_tto = np.sin(tto * deg2rad)
    # FIX: Corrected typo from deg2deg to deg2rad
    tan_tts = np.tan(tts * deg2rad)
    tan_tto = np.tan(tto * deg2rad)

    dso = np.sqrt(tan_tts ** 2 + tan_tto ** 2 - 2 * tan_tts * tan_tto * np.cos(psi_rad))

    # 1. Extinction and Scattering Factors (chi_s, chi_o, frho, ftau are [13, 1])
    sin_ttli = np.sin(litab * deg2rad)
    cos_ttli = np.cos(litab * deg2rad)

    chi_s, chi_o, frho, ftau = _volscatt(
        sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli
    )

    lidf_col = lidf

    # 2. Integrated Extinction & Scattering (scalars/spectra)
    k = chi_s.T @ lidf_col / cos_tts # Extinction coefficient in solar direction (scalar)
    K = chi_o.T @ lidf_col / cos_tto # Extinction coefficient in viewing direction (scalar)

    bfli = cos_ttli ** 2
    bf = bfli.T @ lidf_col

    sob = frho.T @ lidf_col * np.pi / (cos_tts * cos_tto)
    sof = ftau.T @ lidf_col * np.pi / (cos_tts * cos_tto)

    # 3. Geometric Factors for SAIL H-Matrix (Scalars)
    sdb, sdf = 0.5 * (k + bf), 0.5 * (k - bf)
    ddb, ddf = 0.5 * (1 + bf), 0.5 * (1 - bf)
    dob, dof = 0.5 * (K + bf), 0.5 * (K - bf)

    # 4. Scattering Coefficients (Spectrally dependent, [nwl] size)
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau
    a = 1 - sigf # Attenuation

    # --- 5. Analytical Solution (Infinite Canopy and Finite Layers) ---

    m = np.sqrt(np.clip(a ** 2 - sigb ** 2, 1e-9, np.inf)) # Clamping for numerical stability
    rinf = (a - m) / sigb
    rinf2 = rinf ** 2

    e1 = np.exp(-m * LAI)
    e2 = e1 ** 2
    re = rinf * e1

    denom_rhotau = 1 - rinf2 * e2

    # Analytical solution for diffuse reflectance/transmittance (finite)
    tau_dd = (1 - rinf2) * e1 / denom_rhotau
    rho_dd = rinf * (1 - e2) / denom_rhotau

    # Shortened terms for directional solution
    s1 = sf + rinf * sb
    s2 = sf * rinf + sb
    v1 = vf + rinf * vb
    v2 = vf * rinf + vb

    # Analytical solutions for full canopy r_sd and t_sd
    k_minus_m = k.item() - m
    k_plus_m = k.item() + m

    # Analytical solution for Pss_term (Specular to diffuse transmission term)
    Pss_term = s1 * (1 - e1) / k_minus_m
    Qss_term = s2 * (1 - re) / k_plus_m

    # Analytical solutions for full canopy r_sd and t_sd
    tau_sd = (Pss_term - re * Qss_term) / denom_rhotau
    rho_sd = (Qss_term - re * Pss_term) / denom_rhotau # Analytical solution for rho_sd (Diffuse Reflectance for Specular Incidence)

    # Analytical solution for observer direction (tau_oo, tau_do)
    tau_oo = np.exp(-K.item() * LAI) # tau_oo = exp(-K*LAI)
    tau_do = (v1 * (1 - e1) / (K.item() - m) - re * (v2 * (1 - re) / (K.item() + m))) / denom_rhotau

    # Analytical solution for observer directional reflectance (rho_do)
    rho_do = (v2 * (1 - re) / (K.item() + m) - re * (v1 * (1 - e1) / (K.item() - m))) / denom_rhotau

    # rho_so (hotspot summation term)
    xl_vec = np.linspace(0, LAI, nl + 1)[:, np.newaxis]
    dx = LAI / nl
    iLAI = LAI * (1 / nl)
    K_geo = K.item(); k_geo = k.item()

    Pso_prob = np.zeros_like(xl_vec)
    for idx, x_level in enumerate(xl_vec):
        if idx == 0:
            Pso_prob[idx] = Psofunction(x_level, K_geo, k_geo, LAI, canopy.q, dso)
        else:
             x_center = x_level - dx / 2
             Pso_prob[idx] = Psofunction(x_center, K_geo, k_geo, LAI, canopy.q, dso)

    Pso_clamped = np.minimum(Pso_prob.reshape(nl + 1, 1), 1.0) # Simplified clamping

    # rho_so = w * sum(Pso(1:nl)) * iLAI
    rho_so = w * np.sum(Pso_clamped[:nl]) * iLAI # Mock summation

    # Mocking the Pss, Qss, Poo, Qoo terms based on the analytical result form
    # Note: tau_ss used here must be scalar, but we calculated it as spectral: Use mean of spectrum
    tau_ss_scalar = np.mean(np.exp(-k_geo * LAI * iLAI))

    # --- 6. Final Reflectance (Rso, Rdo, Rsd, Rdd) ---

    denom = 1 - rs * rho_dd

    # rso: Bidirectional Reflectance (TOC view)
    rso = rho_so + rho_dd * rs * tau_dd / denom
    rdo = rho_do + (tau_oo + tau_do) * rs * tau_dd / denom
    rsd = rho_sd + (tau_ss_scalar + tau_sd) * rs * tau_dd / denom # Using scalar tau_ss
    rdd = rho_dd + tau_dd * rs * tau_dd / denom

    # Return the extinction coefficient 'k' (needed for LayeredSCOPE) along with Reflectances
    return CanopyReflectances(rso, rdo, rsd, rdd), k.item()


# ============================================================
# SAILH MODEL INTEGRATION (PART 2: LayeredSCOPE Update)
# ============================================================

# --- Original LayeredSCOPE Module (Updated) ---

def par_from_sw(sw_down):
    return sw_down * 4.6

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

def ball_berry(A_net, rh, cs=400.0, gs0=0.01, m=9.0):
    A_mol = A_net * 1e-6
    gs = gs0 + m * (A_mol * np.clip(rh, 0, 1)) / max(cs * 1e-6, 1e-9)
    return max(gs, 1e-6)

def energy_balance_leaf(tleaf_guess, tair, par_abs_leaf, gs, wind, gb0=0.2):
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

def leaf_sif(par_abs_leaf, A_leaf, phi_fmax=0.03):
    par_abs_leaf = float(par_abs_leaf)
    A_leaf = float(A_leaf)
    phi_P = np.clip(A_leaf / (par_abs_leaf + 1e-9), 0, 1)
    npq = 0.3
    phi_F = phi_fmax * (1 - phi_P - npq)
    phi_F = np.clip(phi_F, 0, phi_fmax)
    return phi_F * par_abs_leaf


class LayeredSCOPE:
    # Adding SAILH parameters
    def __init__(self, lai=3.0, v_cmax25=60.0, j_max25=120.0, n_layers=10,
                 cab=40, cca=10, cdm=0.01, cw=0.01, N_leaf=1.5, PROT=0.0, CBC=0.0,
                 # SAILH parameters:
                 LIDFa=0.0, LIDFb=0.0, q=0.1, sol_angle=30.0, obs_angle=0.0, rel_angle=180.0,
                 SMp=15.0, B=0.5, lat=0.0, lon=100.0,
                 spectral_params=None, optical_params=None):
        self.lai = float(lai)
        self.v_cmax25 = float(v_cmax25)
        self.j_max25 = float(j_max25)
        self.n_layers = int(n_layers)
        self.cab, self.cca, self.cdm, self.cw = float(cab), float(cca), float(cdm), float(cw)
        self.N_leaf, self.PROT, self.CBC = float(N_leaf), float(PROT), float(CBC)

        # SAILH/BSM parameters
        self.LIDFa, self.LIDFb, self.q = float(LIDFa), float(LIDFb), float(q)
        self.sol_angle, self.obs_angle, self.rel_angle = float(sol_angle), float(obs_angle), float(rel_angle)
        self.SMp, self.B, self.lat, self.lon = float(SMp), float(B), float(lat), float(lon)

        self.spectral_params = spectral_params if spectral_params is not None else self._mock_spectral()
        self.optical_params = optical_params if optical_params is not None else self._mock_optical_params()

        # Pre-calculate SAIL structure for geometry (lidf, etc.)
        self.canopy_structure = CanopyStructure(self.lai, self.LIDFa, self.LIDFb, self.q, self.n_layers)
        self.angles = Angles(self.sol_angle, self.obs_angle, self.rel_angle)

    def _mock_spectral(self):
        wl = np.linspace(400, 2400, 100)
        Ipar = np.where((wl >= 400) & (wl <= 700))[0]
        return {'wl_spectrum': wl, 'Ipar': Ipar}

    def _mock_optical_params(self):
        nwl = self._mock_spectral()['wl_spectrum'].size
        return {
            "nr": np.linspace(1.2, 1.4, nwl),
            "Kdm": np.full(nwl, 0.01),
            "Kab": 0.5 * np.exp(-0.01 * np.arange(nwl)),
            "Kca": np.full(nwl, 0.05),
            "Kw": np.full(nwl, 0.01),
            "Ks": np.full(nwl, 0.001),
            "Kant": np.full(nwl, 0.1),
            "kcbc": np.full(nwl, 0.005),
            "kprot": np.full(nwl, 0.005),
            "GSV": np.ones((nwl, 3)) * 0.3, # Mock Global Soil Vectors
            "nw": np.linspace(1.3, 1.33, nwl), # Mock Water Refractive Index
            "kw": np.zeros(nwl), # Mock Water Absorption Coeff
        }


    def run_time_series(self, tair, sw_down, rh, vpd, wind, co2=410.0):
        n = len(tair)
        H_series = np.zeros(n); LE_series = np.zeros(n); GPP_series = np.zeros(n); SIF_series = np.zeros(n)
        tleaf_prev = float(np.asarray(tair).ravel()[0])
        lai_layer = self.lai / self.n_layers

        # 1. PROSPECT: Calculate leaf-level absorption coefficient (alpha_leaf)
        k_leaf_absorption = calculate_leaf_properties_prospect(
            self.cab, self.cca, self.cdm, self.cw, self.N_leaf, self.PROT, self.CBC,
            self.spectral_params, self.optical_params
        )

        # 2. BSM: Calculate Soil Reflectance (rwet)
        soilpar = SoilParameters(self.B, self.lat, self.lon, self.SMp)
        soilopt_bsm = BSM(soilpar, self.optical_params)

        # 3. SAILH: Calculate Canopy Extinction Coefficient (k_ext)
        # LeafOptics required by SAILH
        leafopt_sail = LeafOptics(self.optical_params['Kab'], self.optical_params['Kw'], self.optical_params['Kab'])

        # Mock SoilOptics class instance required by SAILH input (refl is rwet)
        class MockSoil:
            def __init__(self, refl): self.refl = refl
        soilopt_mock = MockSoil(refl=soilopt_bsm.refl)

        # k_ext_solar is the solar beam extinction coefficient k (a scalar)
        _, k_ext_solar = SAILH(soilopt_mock, leafopt_sail, self.canopy_structure, self.angles)
        k_ext = k_ext_solar

        for i in range(n):
            tair_i = float(np.asarray(tair[i]).ravel()[0])
            sw_i = float(np.asarray(sw_down[i]).ravel()[0])
            rh_i = float(np.asarray(rh[i]).ravel()[0])
            wind_i = float(np.asarray(wind[i]).ravel()[0])

            par_layer = par_from_sw(sw_i)
            H_canopy = LE_canopy = GPP_canopy = SIF_canopy = 0.0

            for l in range(self.n_layers):
                # Light Penetration (Beer-Lambert Law using SAIL k_ext)
                # Net absorbed light by the layer = Incoming * (1 - np.exp(-k_ext * dLAI))
                par_abs_layer = par_layer * (1 - np.exp(-k_ext * lai_layer))

                # Absorbed by leaf chlorophyll (uses alpha_leaf determined by PROSPECT)
                par_abs_leaf = par_abs_layer * k_leaf_absorption

                tleaf = tleaf_prev

                # Energy Balance Iteration
                for _ in range(3):
                    A_net = farquhar_a(co2*0.7, tleaf, self.v_cmax25, self.j_max25, par_abs_leaf)
                    gs = ball_berry(A_net, rh_i, cs=co2)
                    tleaf, H_layer, LE_layer = energy_balance_leaf(max(tleaf, tair_i+0.1), tair_i, par_abs_leaf, gs, wind_i)

                F_leaf = leaf_sif(par_abs_leaf, A_net)

                GPP_canopy += A_net * lai_layer
                SIF_canopy += F_leaf * np.exp(-k_ext * lai_layer * (self.n_layers-l-1)) # SIF uses SAIL k_ext
                H_canopy += H_layer * (lai_layer / max(self.lai, 1e-6))
                LE_canopy += LE_layer * (lai_layer / max(self.lai, 1e-6))

                par_layer -= par_abs_layer # Deplete incoming PAR for next layer
                par_layer = max(0, par_layer) # Ensure PAR doesn't go negative
                tleaf_prev = tleaf

            H_series[i], LE_series[i], GPP_series[i], SIF_series[i] = H_canopy, LE_canopy, GPP_canopy, SIF_canopy

        return {"H": H_series, "LE": LE_series, "GPP": GPP_series, "SIF": SIF_series}


# --- DALEC and SCOPE_DALEC Classes (Original) ---

class DALEC:
    def __init__(self,
                 C_leaf=200.0, C_wood=5000.0, C_root=500.0, C_litter=2000.0, C_soil=10000.0,
                 tau_leaf=365.0, tau_wood=1.0/0.0001, tau_root=1.0/0.002, tau_litter=1.0/0.008, tau_soil=1.0/0.00005,
                 f_a=0.3, f_f=0.4, f_r=0.2, f_l=0.1, LMA=50.0):
        # Fortran DALEC Parameters (simplified subset)
        self.P_RATES = {
            'f_a': f_a, 'f_f': f_f, 'f_r': f_r, 'f_l': f_l, 't_litter_rate': 1.0/tau_litter, 't_som_rate': 1.0/tau_soil,
            't_wood_rate': 1.0/tau_wood, 't_root_rate': 1.0/tau_root, 'LMA': LMA
        }

        # Carbon Pools (gC/m2)
        self.C = {'leaf': C_leaf, 'wood': C_wood, 'root': C_root, 'litter': C_litter, 'soil': C_soil, 'labile': 50.0}

        # Lifetimes (days) - Used for pool turnover calculation
        self.tau = {'leaf': tau_leaf, 'wood': tau_wood, 'root': tau_root,
                    'litter': tau_litter, 'soil': tau_soil, 'labile': 10.0} # Labile life set to p[14]=10 days

    def update(self, GPP, Tair):
        # Note: Time step (deltat) is assumed to be 1 day for this simplified daily DALEC model
        deltat = 1.0

        GPP = float(GPP); Tair = float(Tair)

        # GPP conversion (umol C/m2/s -> gC/m2/day)
        GPP_gC = GPP * 12e-6 * 3600 * 24

        # --- Fortran DALEC Core Fluxes (Simplified Allocation/Turnover) ---

        # 1. Autotrophic Respiration (Ra) and NPP
        Ra = self.P_RATES['f_a'] * GPP_gC # f_a is the fraction of GPP respired (p[2])
        NPP = GPP_gC - Ra

        # 2. Allocation (NPP -> P) - based on p[3], p[4], p[12]
        P_leaf = NPP * self.P_RATES['f_f']
        P_root = (NPP - P_leaf) * self.P_RATES['f_r']
        P_labile = (NPP - P_leaf - P_root) * self.P_RATES['f_l'] # P_labile (f_l) is p[12]
        P_wood = NPP - P_leaf - P_root - P_labile # Residual allocation to wood

        # 3. Turnover Rates (Simplified: k = 1/tau or k = P_rate)
        # Note: Phenology factors are ignored; using constant turnover rates
        k_labile = 1.0 / self.tau['labile']
        k_leaf = 1.0 / self.tau['leaf']
        k_wood = self.P_RATES['t_wood_rate']
        k_root = self.P_RATES['t_root_rate']
        k_litter = self.P_RATES['t_litter_rate']
        k_som = self.P_RATES['t_som_rate']

        # 4. Turnover Fluxes (T) - T = C * (1 - (1 - k_rate)^deltat) / deltat (Simplified to C * k_rate * deltat)
        T_labile = self.C['labile'] * k_labile * deltat
        T_leaf = self.C['leaf'] * k_leaf * deltat
        T_wood = self.C['wood'] * k_wood * deltat
        T_root = self.C['root'] * k_root * deltat
        T_litter = self.C['litter'] * k_litter * deltat
        T_som = self.C['soil'] * k_som * deltat

        # 5. Pool Update (C_new = C_old + (Production - Turnover) * deltat)

        # Labile (P_labile -> T_labile)
        dC_labile = P_labile - T_labile

        # Foliar (P_leaf + T_labile -> T_leaf)
        dC_leaf = P_leaf + T_labile - T_leaf

        # Root (P_root -> T_root)
        dC_root = P_root - T_root

        # Wood (P_wood -> T_wood)
        dC_wood = P_wood - T_wood

        # Litter (T_leaf + T_root -> T_litter)
        dC_litter = T_leaf + T_root - T_litter

        # SOM (T_litter + T_wood -> T_som)
        dC_soil = T_litter + T_wood - T_som

        # Update pools
        self.C['labile'] += dC_labile; self.C['leaf'] += dC_leaf; self.C['root'] += dC_root
        self.C['wood'] += dC_wood; self.C['litter'] += dC_litter; self.C['soil'] += dC_soil

        # Clamp pools at zero
        for k in self.C: self.C[k] = max(0.0, self.C[k])

        # 6. Reco and LAI output
        Reco = Ra + T_litter + T_som # Heterotrophic respiration is T_litter + T_som
        Reco_umol = Reco / (12e-6 * 3600 * 24)

        # LAI is derived from Foliar pool and LMA (p[17]=LMA)
        LAI = max(0.1, self.C['leaf'] / self.P_RATES['LMA'])

        return Reco_umol, LAI

class SCOPE_DALEC:
    def __init__(self, scope_params, dalec_params):
        self.scope = LayeredSCOPE(**scope_params)

        # Initializing the enhanced DALEC model with Fortran-derived initial pools/rates
        self.dalec = DALEC(
            C_leaf=scope_params.get('C_leaf', 200.0), C_wood=scope_params.get('C_wood', 5000.0),
            C_root=scope_params.get('C_root', 500.0), C_litter=scope_params.get('C_litter', 2000.0),
            C_soil=scope_params.get('C_soil', 10000.0),
            tau_leaf=365.0, tau_wood=1.0/0.0001, tau_root=1.0/0.002, tau_litter=1.0/0.008, tau_soil=1.0/0.00005,
            f_a=0.3, f_f=0.4, f_r=0.2, f_l=0.1, LMA=50.0
        )
        self.results = {'GPP': [], 'Reco': [], 'NEE': [], 'H': [], 'LE': [], 'SIF': [], 'LAI': []}

    def run(self, met):
        tair, sw_down, rh, vpd, wind = met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind']
        self.results = {k: [] for k in self.results}
        for i in range(len(tair)):
            # Step 1: Calculate instantaneous fluxes based on current LAI
            scope_out = self.scope.run_time_series(
                np.array([tair[i]]), np.array([sw_down[i]]), np.array([rh[i]]), np.array([vpd[i]]), np.array([wind[i]])
            )
            GPP = float(scope_out['GPP'][0]); H = float(scope_out['H'][0]); LE = float(scope_out['LE'][0]); SIF = float(scope_out['SIF'][0])

            # Step 2: DALEC update (daily step)
            Reco, LAI_new = self.dalec.update(GPP, tair[i])
            NEE = Reco - GPP

            # Step 3: Update SCOPE's LAI for the next iteration
            self.scope.lai = LAI_new
            self.canopy_structure = CanopyStructure(LAI_new, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            for k, v in zip(['GPP','Reco','NEE','H','LE','SIF','LAI'], [GPP,Reco,NEE,H,LE,SIF,LAI_new]):
                self.results[k].append(v)
        return self.results

    def assimilate_ndvi(self, met, ndvi_obs, param_names=['cab','v_cmax25','j_max25'],
                         x0=None, B_inv=None, R_inv=None):
        if x0 is None: x0 = np.array([self.scope.cab, self.scope.v_cmax25, self.scope.j_max25])
        if B_inv is None: B_inv = np.eye(len(x0))
        if R_inv is None: R_inv = np.eye(len(ndvi_obs))

        def cost_function(params):
            # Save original DALEC and SCOPE state
            original_dalec_C = {k: v for k,v in self.dalec.C.items()}
            original_scope_lai = self.scope.lai

            # 1. Update Parameters
            for n,p in zip(param_names, params): setattr(self.scope, n, float(p))

            # 2. Run SCOPE/DALEC with new parameters
            out_scope = self.scope.run_time_series(met['Tair'], met['SWdown'], met['RH'], met['VPD'], met['Wind'])

            # Need to re-run DALEC sequentially to get the LAI evolution
            self.dalec.C = original_dalec_C.copy() # Reset DALEC C pools
            out_dalec_lai = []

            for i in range(len(met['Tair'])):
                GPP_i = float(out_scope['GPP'][i])
                _, LAI = self.dalec.update(GPP_i, met['Tair'][i])
                out_dalec_lai.append(LAI)
                # Note: SCOPE's LAI must also be updated sequentially during cost calculation
                self.scope.canopy_structure = CanopyStructure(LAI, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            # 3. Calculate Cost
            ndvi_model = LAI_to_NDVI(out_dalec_lai)
            resid = ndvi_model - ndvi_obs
            J_obs = 0.5 * resid.T @ R_inv @ resid
            delta = np.array(params) - x0
            J_b = 0.5 * delta.T @ B_inv @ delta

            # Restore SCOPE/DALEC state for the next minimization step
            self.dalec.C = original_dalec_C
            self.scope.lai = original_scope_lai
            self.scope.canopy_structure = CanopyStructure(original_scope_lai, self.scope.LIDFa, self.scope.LIDFb, self.scope.q, self.scope.n_layers)

            return float(J_obs + J_b)

        res = minimize(cost_function, x0, bounds=[(0,80),(20,150),(50,300)])
        for n,p in zip(param_names, res.x): setattr(self.scope, n, float(p))
        return res

def LAI_to_NDVI(lai, a=0.8, b=0.5):
    return a * (1 - np.exp(-b * np.asarray(lai)))


# ============================================================
# 示例运行
# ============================================================

if __name__ == "__main__":
    hours = 24
    met = {
        "Tair": np.linspace(15,30,hours), "SWdown": np.linspace(0,800,hours),
        "RH": np.linspace(0.5,0.8,hours), "VPD": np.linspace(0.5,2.0,hours),
        "Wind": np.linspace(0.5,3.0,hours)
    }

    # PROSPECT and SAILH parameters
    scope_params = {"lai":3.0, "v_cmax25":70, "j_max25":140, "n_layers":10,
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
    ndvi_obs = np.linspace(0.3,0.7,hours)
    R_inv = np.eye(hours)/0.05**2
    B_inv = np.eye(3)/np.array([10,10,20])**2

    print("\n--- 启动 4D-Var NDVI 同化 ---")
    res = model.assimilate_ndvi(met, ndvi_obs, x0=[40,70,140], B_inv=B_inv, R_inv=R_inv)

    print("优化后的参数:")
    print(f"Cab: {model.scope.cab:.2f}, Vcmax25: {model.scope.v_cmax25:.2f}, Jmax25: {model.scope.j_max25:.2f}")
