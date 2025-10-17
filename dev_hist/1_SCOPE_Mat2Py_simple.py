import numpy as np
from typing import Dict, Tuple, Any, Callable

'''
A very simplied SCOPE, and only RTMt_sb (thermal radiative transfer)
'''

# ==============================================================================
# 1. CORE MATH HELPERS (Equivalent to MATLAB's nested functions)
# ==============================================================================

def _sel_root(a, b, c, dsign):
    """Quadratic formula root selection: ax^2 + bx + c = 0."""
    if np.isscalar(a) and a == 0:
        x = -c / b
    else:
        dsign = np.where(dsign == 0, -1, dsign)
        discriminant = np.sqrt(b**2 - 4 * a * c)
        x = (-b + dsign * discriminant) / (2 * a)
    return x

def satvap(T_c: np.ndarray) -> np.ndarray:
    """Saturated vapour pressure (hPa) at temperature T (Celsius)."""
    return 6.107 * 10**(7.5 * T_c / (237.3 + T_c))

# --- Biochemical Helpers ---

# Persistent variable for counting calls to _compute_a (used in the iterative Ci solver)
_compute_a_fcount = [0]

def _temperature_function_c3(Tref, R, T, deltaHa):
    """Temperature function for C3 (exponential factor only)."""
    tempfunc1 = (1 - Tref / T)
    fTv = np.exp(deltaHa / (Tref * R) * tempfunc1)
    return fTv

def _high_temp_inhibtion_c3(Tref, R, T, deltaS, deltaHd):
    """High Temperature Inhibition Function for C3."""
    hightempfunc_num = (1 + np.exp((Tref * deltaS - deltaHd) / (Tref * R)))
    hightempfunc_deno = (1 + np.exp((deltaS * T - deltaHd) / (R * T)))
    fHTv = hightempfunc_num / hightempfunc_deno
    return fHTv

def _fluorescencemodel(ps, x, Kp, Kf, Kd, Knparams):
    """Calculates fluorescence yields and quenching terms."""
    Kno, alpha, beta = Knparams[0], Knparams[1], Knparams[2]

    x_alpha = np.power(x, alpha)
    Kn = Kno * (1 + beta) * x_alpha / (beta + x_alpha)

    fo0 = Kf / (Kf + Kp + Kd)
    fo = Kf / (Kf + Kp + Kd + Kn)
    fm = Kf / (Kf + Kd + Kn)
    fm0 = Kf / (Kf + Kd)
    fs = fm * (1 - ps)

    eta = np.where(fo0 > 0, fs / fo0, 0.0)
    qQ = 1 - (fs - fo) / (fm - fo)
    qE = 1 - (fm - fo) / (fm0 - fo0)

    return eta, qE, qQ, fs, fo, fm, fo0, fm0, Kn

def _gs_fun(Cs, RH, A, BallBerrySlope, BallBerry0):
    """Helper function for BallBerry to compute stomatal conductance gs."""
    gs = np.maximum(BallBerry0, BallBerrySlope * A * RH / (Cs + 1e-9) + BallBerry0)
    gs = np.where(np.isnan(Cs), np.nan, gs)
    return gs

def _ball_berry(Cs, RH, A, BallBerrySlope, BallBerry0, minCi, Ci_input=None):
    """Ball-Berry model for Ci/gs."""
    if Ci_input is not None and Ci_input.size > 0:
        Ci = Ci_input
        gs = None
        if A is not None and A.size > 0 and A.ndim == Ci_input.ndim:
            gs = _gs_fun(Cs, RH, A, BallBerrySlope, BallBerry0)
    elif np.all(BallBerry0 == 0) or A is None or A.size == 0:
        Ci = np.maximum(minCi * Cs, Cs * (1 - 1.6 / (BallBerrySlope * RH)))
        gs = None
    else:
        gs = _gs_fun(Cs, RH, A, BallBerrySlope, BallBerry0)
        Ci = np.maximum(minCi * Cs, Cs - 1.6 * A / gs)
    return Ci, gs

def _ci_next(Ci_in, Cs, RH, minCi, BallBerrySlope, BallBerry0, A_fun, ppm2bar):
    """Error function for Ci fixed-point iteration."""
    A, _ = A_fun(Ci_in)
    A_bar = A * ppm2bar
    Ci_out, _ = _ball_berry(Cs, RH, A_bar, BallBerrySlope, BallBerry0, minCi)
    err = Ci_out - Ci_in
    return err, Ci_out

def _fixedp_brent_ari(fun, x0, corner, tol):
    """Mock Fixed-Point Solver for Ci iteration."""
    Ci_in = x0
    for _ in range(100):
        err, Ci_next_val = fun(Ci_in)
        if np.max(np.abs(err)) < tol:
            return Ci_next_val
        Ci_in = Ci_next_val
    return Ci_next_val

def _compute_a(Ci, Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, kpepcase):
    """Computes Gross (Ag) and Net (A) Assimilation."""
    global _compute_a_fcount

    # Handle reset call
    if Ci is None:
        _compute_a_fcount[0] = 0
        return None, None

    fcount = _compute_a_fcount[0]

    if Type.lower() == 'c3':
        Vs = Vs_C3
        if np.any(g_m < np.inf):
            a_vc, b_vc = 1.0 / g_m, -(MM_consts + Ci + (Rd + Vcmax) / g_m)
            c_vc = Vcmax * (Ci - Gamma_star + Rd / g_m)
            Vc = _sel_root(a_vc, b_vc, c_vc, -1)

            a_ve, b_ve = 1.0 / g_m, -(Ci + 2 * Gamma_star + (Rd + Je * effcon) / g_m)
            c_ve = Je * effcon * (Ci - Gamma_star + Rd / g_m)
            Ve = _sel_root(a_ve, b_ve, c_ve, -1)
            CO2_per_electron = Ve / Je
        else:
            Vc = Vcmax * (Ci - Gamma_star) / (MM_consts + Ci)
            CO2_per_electron = (Ci - Gamma_star) / (Ci + 2 * Gamma_star) * effcon
            Ve = Je * CO2_per_electron
    else: # C4
        Vc, Vs = Vcmax, kpepcase * Ci
        CO2_per_electron = effcon
        Ve = Je * CO2_per_electron

    # Smoothing min(Vc, Ve) -> V and min(V, Vs) -> Ag
    V = _sel_root(atheta, -(Vc + Ve), Vc * Ve, np.sign(Gamma_star - Ci))
    Ag = _sel_root(0.98, -(V + Vs), V * Vs, -1)

    A = Ag - Rd
    _compute_a_fcount[0] = fcount + 1

    biochem_out = {'A': A, 'Ag': Ag, 'CO2_per_electron': CO2_per_electron}
    return A, biochem_out

# --- Resistance Helpers ---

def _psim(z, L, unst, st, x):
    """Stability correction for momentum."""
    pm = np.zeros_like(z)
    if np.any(unst):
        pm = np.where(unst,
                      (2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2),
                      pm)
    if np.any(st):
        pm = np.where(st, -5 * z / L, pm)
    return pm

def _psih(z, L, unst, st, x):
    """Stability correction for heat."""
    ph = np.zeros_like(z)
    if np.any(unst):
        ph = np.where(unst, 2 * np.log((1 + x**2) / 2), ph)
    if np.any(st):
        ph = np.where(st, -5 * z / L, ph)
    return ph

def _phstar(z, zR, d, L, st, unst, x):
    """Stability correction function for the canopy layer."""
    phs = np.zeros_like(z)
    if np.any(unst):
        phs = np.where(unst, (z - d) / (zR - d) * (x**2 - 1) / (x**2 + 1), phs)
    if np.any(st):
        phs = np.where(st, -5 * z / L, phs)
    return phs

# ==============================================================================
# 2. FLUX AND RESISTANCE MODULES
# ==============================================================================

def biochemical(leafbio: Dict, meteo: Dict, options: Dict, constants: Dict, fV: np.ndarray) -> Dict:
    """
    SCOPE's core biochemical model (Farquhar/Collatz).
    Calculates A, Ci, and fluorescence parameters (eta, Kn, etc.).
    """

    # --- Setup ---
    tempcor = options['apply_T_corr']
    rhoa, Mair, R = constants['rhoa'], constants['Mair'], constants['R']
    Q, Cs, eb, O, p = meteo['Q'], meteo['Cs'], meteo['eb'], meteo['Oa'], meteo['p']
    T_celsius = meteo['T']
    T = T_celsius + 273.15 * (T_celsius < 200) # [K]
    Type, Vcmax25_base = leafbio['Type'], leafbio['Vcmax25']
    BallBerrySlope, RdPerVcmax25, BallBerry0 = leafbio['BallBerrySlope'], leafbio['RdPerVcmax25'], leafbio['BallBerry0']
    Tref = 25 + 273.15
    Kc25, Ko25, spfy25 = 405, 279, 2444

    ppm2bar = 1e-6 * (p * 1e-3)
    Cs_bar = Cs * ppm2bar
    O_bar = (O * 1e-3) * (p * 1e-3) * (1 if Type.lower() == 'c3' else 0)
    Vcmax25, Rd25 = fV * Vcmax25_base, RdPerVcmax25 * Vcmax25_base
    g_m = leafbio.get('g_m', np.inf) * 1e6 if 'g_m' in leafbio else np.inf

    # --- T Corrections ---
    Vcmax, Rd, Kc, Ko, Gamma_star = Vcmax25, Rd25, Kc25 * 1e-6, Ko25 * 1e-3, 0.5 * O_bar / spfy25
    Ke = 1

    if tempcor:
        TDP = leafbio['TDP']
        # T corrections logic (simplified)
        if Type.lower() == 'c3':
            fTv = _temperature_function_c3(Tref, R, T, TDP['delHaV'])
            fHTv = _high_temp_inhibtion_c3(Tref, R, T, TDP['delSV'], TDP['delHdV'])
            Vcmax = Vcmax25 * fTv * fHTv * leafbio['stressfactor']
            Rd = Rd25 * _temperature_function_c3(Tref, R, T, TDP['delHaR']) * leafbio['stressfactor']
            Kc = Kc25 * 1e-6 * _temperature_function_c3(Tref, R, T, TDP['delHaKc'])
            Ko = Ko25 * 1e-3 * _temperature_function_c3(Tref, R, T, TDP['delHaKo'])
            Gamma_star = (0.5 * O_bar / spfy25) * _temperature_function_c3(Tref, R, T, TDP['delHaT'])
        # C4 logic is complex and omitted for core structure (using C3 fallback for mock)

    # --- Je & Photosynthesis constants ---
    po0 = 4.0 / (0.05 + np.maximum(0.8738, 0.0301 * T_celsius + 0.0773) + 4.0)
    Je = 0.5 * po0 * Q

    if Type.lower() == 'c3':
        MM_consts, Vs_C3, minCi = (Kc * (1 + O_bar / Ko)), (Vcmax / 2), 0.3
        effcon, kpepcase = 1/5, None
    else:
        MM_consts, Vs_C3, minCi = 0, 0, 0.1
        effcon, kpepcase = 1/6, 1.0 # Mock kpepcase since it's missing in input

    # --- Ci Iteration ---
    RH = np.minimum(1, eb / satvap(T_celsius))
    _compute_a(None, None, None, None, None, None, None, None, None, None, None, None) # Reset fcount

    computeA_fun = lambda x: _compute_a(x, Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, 0.8, kpepcase)

    if np.all(BallBerry0 == 0):
        Ci_bar, _ = _ball_berry(Cs_bar, RH, None, BallBerrySlope, BallBerry0, minCi)
    else:
        Ci_next_fun = lambda x: _ci_next(x, Cs_bar, RH, minCi, BallBerrySlope, BallBerry0, computeA_fun, ppm2bar)
        Ci_bar = _fixedp_brent_ari(Ci_next_fun, Cs_bar, None, 1e-7)

    # --- Final Fluxes ---
    A, _ = computeA_fun(Ci_bar)

    Ag = A + Rd
    CO2_per_electron = Ag / Je
    gs = np.maximum(0, 1.6 * A * ppm2bar / (Cs_bar - Ci_bar))
    Ja = Ag / CO2_per_electron
    rcw = (rhoa / (Mair * 1e-3)) / gs

    Kd = np.maximum(0.8738, 0.0301 * T_celsius + 0.0773)
    ps = po0 * Ja / Je
    ps = np.where(np.isnan(ps), np.broadcast_to(po0, ps.shape), ps)
    ps_rel = np.maximum(0, 1 - ps / po0)

    Knparams = np.array([leafbio['Kn0'], leafbio['Knalpha'], leafbio['Knbeta']])
    eta, qE, qQ, fs, fo, fm, fo0, fm0, Kn = _fluorescencemodel(ps, ps_rel, 4.0, 0.05, Kd, Knparams)

    # --- Output ---
    return {
        'A': A, 'Ag': Ag, 'Ci': Ci_bar / ppm2bar, 'rcw': rcw, 'gs': gs, 'RH': RH,
        'Ja': Ja, 'ps': ps, 'ps_rel': ps_rel, 'Kn': Kn, 'eta': eta, 'fs': fs,
        'qE': qE, 'qQ': qQ, 'SIF': fs * Q
    }

def heatfluxes(ra: np.ndarray, rs: np.ndarray, Tc: np.ndarray, ea: np.ndarray, Ta: np.ndarray,
               e_to_q: float, Ca: np.ndarray, Ci: np.ndarray, constants: Dict,
               es_fun: Callable, s_fun: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates latent (lE) and sensible (H) heat flux."""
    rhoa, cp = constants['rhoa'], constants['cp']

    lambda_val = (2.501 - 0.002361 * Tc) * 1e6  # Latent heat of vaporization [J kg-1]
    ei = es_fun(Tc)
    s = s_fun(ei, Tc)

    qi, qa = ei * e_to_q, ea * e_to_q # Absolute humidity

    lE = rhoa / (ra + rs) * lambda_val * (qi - qa)   # Latent heat flux [W m-2]
    H = (rhoa * cp) / ra * (Tc - Ta)           # Sensible heat flux [W m-2]
    ec = ea + (ei - ea) * ra / (ra + rs)         # Vapour pressure at surface
    Cc = Ca - (Ca - Ci) * ra / (ra + rs)        # CO2 concentration at surface

    return lE, H, ec, Cc, lambda_val, s

def resistances(constants: Dict, soil: Dict, canopy: Dict, meteo: Dict) -> Dict:
    """Calculates aerodynamic and boundary resistances."""
    kappa = constants['kappa']
    Cd, LAI, rwc, z0m, d, h = canopy['Cd'], canopy['LAI'], canopy['rwc'], canopy['zo'], canopy['d'], canopy['hc']
    z, u, L = meteo['z'], np.maximum(0.3, meteo['u']), meteo['L']
    rbs = soil['rbs']

    # Derived parameters
    zr, n = 2.5 * h, Cd * LAI / (2 * kappa**2)
    unst, st = (L < 0) & (L > -500), (L > 0) & (L < 500)
    x = np.where(unst, (1 - 16 * z / L)**(1 / 4), 1)

    # Stability corrections and ustar
    pm_z = _psim(z - d, L, unst, st, x)
    pm_h = _psim(h - d, L, unst, st, x)
    ph_z = _psih(z - d, L, unst, st, x)
    ustar = np.maximum(0.001, kappa * u / (np.log((z - d) / z0m) - pm_z))

    # Kh calculation
    Kh_zr = kappa * ustar * (zr - d)
    Kh = Kh_zr
    Kh = np.where(unst, Kh_zr * (1 - 16 * (h - d) / L)**0.5, Kh)

    # Resistances
    ph_zr = np.where(z >= zr, _psih(zr - d, L, unst, st, x), ph_z)
    rai = np.where(z > zr, 1 / (kappa * ustar) * (np.log((z - d) / (zr - d)) - ph_z + ph_zr), 0)
    rar = 1 / (kappa * ustar) * ((zr - h) / (zr - d) - _phstar(zr, zr, d, L, st, unst, x) + _phstar(h, zr, d, L, st, unst, x))

    log_rac = (np.log((np.exp(n) - 1) / (np.exp(n) + 1)) -
               np.log((np.exp(n * (z0m + d) / h) - 1) / (np.exp(n * (z0m + d) / h) + 1)))
    rac = h * np.sinh(n) / (n * Kh) * log_rac

    raa, rawc, raws = rai + rar + rac, rwc, _phstar(0.01, zr, d, L, st, unst, x) + rbs

    return {'ustar': ustar, 'raa': raa, 'rawc': rawc, 'raws': raws}

# ==============================================================================
# 3. MOCK EXTERNAL FUNCTIONS (RTM, Aggregation, Stability)
# ==============================================================================

def RTMt_sb(constants: Dict, rad: Dict, soil: Dict, leafbio: Dict, canopy: Dict, gap: Dict,
            Tcu: np.ndarray, Tch: np.ndarray, Tsu: float, Tsh: float, option: int) -> Dict:
    """
    MOCK RTMt_sb: Generates plausible net radiation components for the EBal loop.
    Ensures non-zero PAR/Net Rad for biochemical/EBal to work.
    """
    nl = canopy['nlayers']
    # Net radiation calculation based on leaf/soil temperature difference
    base_Rn = 50.0

    # Net Radiances (W m^-2)
    Rnhc = base_Rn * np.ones(nl) + 1.0 * (Tch - meteo_in['Ta'])
    Rnuc = (base_Rn * 2) * np.ones_like(Tcu) + 1.0 * (Tcu - meteo_in['Ta'])

    # Soil Net Radiances (W m^-2)
    Rnhs = base_Rn * 0.5 + 1.0 * (Tsh - meteo_in['Ta'])
    Rnus = base_Rn * 1.5 + 1.0 * (Tsu - meteo_in['Ta'])

    return {
        'Rnhc': Rnhc, 'Rnhct': np.zeros_like(Rnhc),
        'Rnuc': Rnuc, 'Rnuct': np.zeros_like(Rnuc),
        'Rnhs': Rnhs, 'Rnhst': 0.0,
        'Rnus': Rnus, 'Rnust': 0.0,
        'Eoutte': 1.0,
        'Pnh_Cab': 100 * np.ones(nl), # Absorbed PAR for shaded
        'Pnu_Cab': 200 * np.ones_like(Tcu), # Absorbed PAR for sunlit
    }

def Monin_Obukhov(constants: Dict, meteo: Dict, H: float) -> float:
    """
    Translation of Monin_Obukhov.m: Calculates the Monin-Obukhov length L.
    """
    rhoa, cp = constants['rhoa'], constants['cp']
    kappa, g = constants['kappa'], constants['g']
    C2K = constants.get('C2K', 273.15)

    ustar = meteo['ustar']
    Ta_K = meteo['Ta'] + C2K

    L = -rhoa * cp * ustar**3 * Ta_K / (kappa * g * H)
    L = np.where(np.isnan(L), -1E6, L)

    return L.item() if isinstance(L, np.ndarray) and L.ndim == 0 else L


def meanleaf(canopy: Dict, F: np.ndarray, choice: str, Ps: np.ndarray) -> np.ndarray:
    """
    Translation of meanleaf.m: Calculates the layer average and the canopy average
    of leaf properties (F).
    """
    nl = canopy['nlayers']
    nli = canopy.get('nlincl')
    nlazi = canopy.get('nlazi')
    lidf = canopy['lidf']

    Fout = F

    if choice == 'angles':
        # integration over leaf angles (result is [nl] vector)
        F_weighted = F * lidf[:nli, np.newaxis, np.newaxis]
        Fout = np.sum(F_weighted, axis=(0, 1)) / nlazi

    elif choice == 'layers':
        # integration over layers only (result is scalar average)
        Fout = np.dot(Ps, F) / nl

    elif choice == 'angles_and_layers':
        # integration over both leaf angles and layers (result is scalar average)

        F_weighted = F * lidf[:nli, np.newaxis, np.newaxis]
        F_weighted_Ps = F_weighted * Ps[np.newaxis, np.newaxis, :nl]
        Fout = np.sum(F_weighted_Ps) / nlazi / nl

    else:
        if F.ndim > 1:
            Fout = np.mean(F, axis=tuple(range(F.ndim - 1)))
        else:
            Fout = np.mean(F)

    return Fout

def aggregator(LAI: float, sunlit_flux: np.ndarray, shaded_flux: np.ndarray, Fs: np.ndarray, canopy: Dict, integr: str) -> np.ndarray:
    """Aggregates fluxes over sunlit/shaded fractions and layers (SCOPE default)."""
    mean_sunlit = meanleaf(canopy, sunlit_flux, integr, Fs)
    mean_shaded = meanleaf(canopy, shaded_flux, 'layers', 1 - Fs)

    flux_tot = LAI * (mean_sunlit + mean_shaded)
    return flux_tot


# ==============================================================================
# 4. CORE SCOPE SOLVER (ebal.m equivalent)
# ==============================================================================

def ebal_scope(constants: Dict, options: Dict, rad: Dict, gap: Dict, meteo: Dict,
               soil: Dict, canopy: Dict, leafbio: Dict, integr: str) -> Dict[str, Any]:
    """
    Core SCOPE Energy Balance Solver (Simplified ebal.m implementation).
    Iteratively solves leaf/soil temperatures and fluxes.
    """
    # --- Setup ---
    nl, LAI = canopy['nlayers'], canopy['LAI']
    Ta, ea, Ca, p = meteo['Ta'], meteo['ea'], meteo['Ca'], meteo['p']
    rhoa, cp = constants['rhoa'], constants['cp']

    es_fun = lambda T: satvap(T)
    s_fun = lambda es, T: es * 2.3026 * 7.5 * 237.3 / (237.3 + T)**2

    # Initial T guesses (oC)
    Tsh, Tsu = Ta + 1.0, Ta + 3.0
    Ts = np.array([Tsh, Tsu]) # [Tsh, Tsu]
    Tch, Tcu = (Ta + 0.1) * np.ones(nl), (Ta + 0.3) * np.ones((13, 36, nl)) # Sunlit is multi-dimensional

    # Initial boundary conditions
    ech, Cch = ea * np.ones(nl), Ca * np.ones(nl)
    ecu, Ccu = ea * np.ones_like(Tcu), Ca * np.ones_like(Tcu)

    e_to_q = constants['MH2O'] / constants['Mair'] / p
    Fc = gap['Ps'][:-1] # Canopy layer gap fraction (Ps[1:end-1])
    Fs = np.array([1 - gap['Ps'][-1], gap['Ps'][-1]]) # Soil fractions
    fV = np.exp(canopy['kV'] * canopy['xl'][:nl])
    fVu = np.ones_like(Tcu) * fV[np.newaxis, np.newaxis, :]

    # Iteration control
    maxEBer, maxit, Wc, counter = 1.0, 100, 1.0, 0
    CONT = True

    # --- Main Iteration Loop ---
    while CONT and counter < maxit:
        # 1. Net Radiation
        rad_out = RTMt_sb(constants, rad, soil, leafbio, canopy, gap, Tcu, Tch, Tsu, Tsh, 0)
        Rnhc = rad_out['Rnhc'] + rad_out.get('Rnhct', 0)
        Rnuc = rad_out['Rnuc'] + rad_out.get('Rnuct', 0)
        Rns = np.array([rad_out['Rnhs'] + rad_out.get('Rnhst', 0),
                        rad_out['Rnus'] + rad_out.get('Rnust', 0)])

        # 2. Biochemical & Stomatal Resistance
        meteo_h, meteo_u = meteo.copy(), meteo.copy()
        meteo_h.update({'T': Tch, 'eb': ech, 'Cs': Cch, 'Q': rad_out['Pnh_Cab']}) # Pnh_Cab: net PAR Cab shaded leaves (photons)
        meteo_u.update({'T': Tcu, 'eb': ecu, 'Cs': Ccu, 'Q': rad_out['Pnu_Cab']})

        bch = biochemical(leafbio, meteo_h, options, constants, fV)
        bcu = biochemical(leafbio, meteo_u, options, constants, fVu)

        # 3. Aerodynamic Resistances
        resist_out = resistances(constants, soil, canopy, meteo)
        meteo['L'] = 100.0 # Mock initial L for first step

        rac = (LAI + 1) * (resist_out['raa'] + resist_out['rawc']) # Aerodynamic resistance in canopy layer (above z0+d) [s m-1]
        ras = (LAI + 1) * (resist_out['raa'] + resist_out['raws'])

        # 4. Heat Fluxes
        lEch, Hch, ech, Cch, lambdah, sh = heatfluxes(rac, bch['rcw'], Tch, ea, Ta, e_to_q, Ca, bch['Ci'], constants, es_fun, s_fun)
        lEcu, Hcu, ecu, Ccu, lambdau, su = heatfluxes(rac, bcu['rcw'], Tcu, ea, Ta, e_to_q, Ca, bcu['Ci'], constants, es_fun, s_fun)
        lEs, Hs, _, _, lambdas, ss = heatfluxes(ras, soil['rss'], Ts, ea, Ta, e_to_q, Ca, Ca, constants, es_fun, s_fun)

        # 5. Aggregate Fluxes
        G = 0.35 * Rns # Mock Soil Heat Flux
        Htot = Fs @ Hs + aggregator(LAI, Hcu, Hch, Fc, canopy, integr)
        meteo['L'] = Monin_Obukhov(constants, meteo, Htot) # Update stability

        # 6. Energy Balance Errors
        EBerch = Rnhc - lEch - Hch
        EBercu = Rnuc - lEcu - Hcu
        EBers = Rns - lEs - Hs - G

        maxEBerch = np.max(np.abs(EBerch))
        maxEBercu = np.max(np.abs(EBercu))
        maxEBers = np.max(np.abs(EBers))

        CONT = (maxEBerch > maxEBer) or (maxEBercu > maxEBer) or (maxEBers > maxEBer)
        if not CONT: break

        # 7. Update Temperatures
        sigmaSB = constants['sigmaSB']

        # Shaded leaf temperature update
        denom_ch = (rhoa * cp / rac + rhoa * lambdah * e_to_q * sh / (rac + bch['rcw']) + 4 * leafbio['emis'] * sigmaSB * (Tch + 273.15)**3)
        Tch += Wc * EBerch / denom_ch

        # Sunlit leaf temperature update
        denom_cu = (rhoa * cp / rac + rhoa * lambdau * e_to_q * su / (rac + bcu['rcw']) + 4 * leafbio['emis'] * sigmaSB * (Tcu + 273.15)**3)
        Tcu += Wc * EBercu / denom_cu

        # Soil temperature update (mocking dG=0 for simplicity)
        denom_s = (rhoa * cp / ras + rhoa * lambdas * e_to_q * ss / (ras + soil['rss']) + 4 * (1 - soil['rs_thermal']) * sigmaSB * (Ts + 273.15)**3)
        Ts += Wc * EBers / denom_s
        Tsh, Tsu = Ts[0], Ts[1] # Update for next RTM call

        # Temperature bounds (omitted for brevity)

        counter += 1
        if counter % 10 == 0: Wc *= 0.9 # Damping

    # --- Final Aggregation and Output ---
    fluxes = {}
    fluxes['Rnctot'] = aggregator(LAI, Rnuc, Rnhc, Fc, canopy, integr)
    fluxes['Actot'] = aggregator(LAI, bcu['A'], bch['A'], Fc, canopy, integr)
    fluxes['Htot'] = Htot
    fluxes['SIF'] = aggregator(LAI, bcu['SIF'], bch['SIF'], Fc, canopy, integr)
    fluxes['GPP'] = canopy['LAI'] * (meanleaf(canopy, bch['Ag'],'layers', Fc) + meanleaf(canopy, bcu['Ag'],integr, 1 - Fc)) # gross photosynthesis

    return {'fluxes': fluxes, 'Tch': Tch, 'Tsu': Tsu, 'iterations': counter}

# ==============================================================================
# 5. EXAMPLE EXECUTION WITH MOCK DATA
# ==============================================================================

# Define mock input dictionaries
NL = 5 # Number of layers
NLI = 13 # Number of leaf inclination angles
NLAZI = 36 # Number of leaf azimuth angles

meteo_in = {
    'Ta': 25.0, 'ea': 15.0, 'Ca': 400.0, 'p': 1013.25, 'Oa': 210.0,
    'Q': 800.0 * np.ones((NLI, NLAZI, NL)), 'u': 2.0, 'z': 20.0, 'L': -10.0, 'ustar': 0.3
}
constants_in = {
    'rhoa': 1.2, 'Mair': 28.97, 'MH2O': 18.015, 'cp': 1004.0,
    'sigmaSB': 5.67e-8, 'R': 8.314, 'kappa': 0.4, 'g': 9.81
}
canopy_in = {
    'nlayers': NL, 'LAI': 3.0, 'kV': -0.5, 'xl': np.linspace(0, 3.0, NL + 1),
    'Cd': 0.2, 'rwc': 10.0, 'zo': 0.5, 'd': 1.0, 'hc': 2.0,
    'nlincl': NLI,
    'nlazi': NLAZI,
    # Spherical (Uniform) Leaf Angle Distribution: all 13 classes are equally likely
    'lidf': np.full(NLI, 1.0 / NLI)
}

soil_in = {
    'rs_thermal': 0.9, 'rss': 100.0, 'rbs': 5.0, 'GAM': 100.0 # Placeholder
}
leafbio_in = {
    'Type': 'C3', 'Vcmax25': 50.0, 'RdPerVcmax25': 0.01, 'BallBerrySlope': 9.0,
    'BallBerry0': 0.01, 'Kn0': 0.05, 'Knalpha': 1.0, 'Knbeta': 1.0,
    'emis': 0.98, 'stressfactor': 1.0,
    'TDP': {'delHaV': 70e3, 'delSV': 600, 'delHdV': 200e3, 'delHaR': 50e3,
            'delSR': 100, 'delHdR': 150e3, 'delHaKc': 60e3, 'delHaKo': 30e3, 'delHaT': 40e3}
}
options_in = {'apply_T_corr': True, 'MoninObukhov': True}

# Mock RTM input (minimal setup)
rad_in = {'Rnuc': meteo_in['Q'], 'Rnhc': meteo_in['Q'][..., 0]}
gap_in = {'Ps': np.linspace(1.0, 0.1, NL + 1)}

# Run the solver
# integr='angles_and_layers' is typically used for sunlit integration
results = ebal_scope(constants_in, options_in, rad_in, gap_in, meteo_in, soil_in, canopy_in, leafbio_in, integr='angles_and_layers')

print("--- SCOPE Core Solver Results (Mock Execution) ---")
print(f"Iterations completed: {results['iterations']}")
print(f"Final Net Assimilation (A_tot) [umol m^-2 s^-1]: {results['fluxes']['Actot']}")
print(f"Final Total Sensible Heat (H_tot) [W m^-2]: {results['fluxes']['Htot']}")
print(f"Final Mean Canopy Temperature (Tch_mean) [oC]: {np.mean(results['Tch']):.2f}")
print(f"Final Sunlit Soil Temperature (Tsu) [oC]: {results['Tsu']:.2f}")
print(f"Final GPP [gC m-2 d01]: {results['fluxes']['GPP']:.2f}")
print(f"Final SIF: {results['fluxes']['SIF']:.2f}")
