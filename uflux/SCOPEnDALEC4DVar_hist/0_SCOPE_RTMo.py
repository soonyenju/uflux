# RTMo_python.py
import numpy as np

def Sint(y, x):
    """
    Spectral integration (trapezoidal rule approximation, as in SCOPE's Sint.m).

    Parameters
    ----------
    y : array_like
        Values to integrate. Can be 1D or 2D (e.g., spectra or profiles × wavelengths).
    x : array_like
        Monotonically increasing vector of x-values (e.g., wavelengths).

    Returns
    -------
    int : ndarray
        Integrated values over x.
    """
    y = np.array(y, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    # Ensure x and y have compatible orientation
    if y.ndim == 1:
        y = y[np.newaxis, :]  # make it 2D: 1 × n

    if x.ndim > 1:
        x = x.flatten()

    nx = len(x)
    step = x[1:nx] - x[:nx-1]                    # differences along x
    mean_vals = 0.5 * (y[:, :nx-1] + y[:, 1:nx]) # trapezoidal mean
    integral = np.sum(mean_vals * step, axis=1)   # integrate along wavelength

    return integral.squeeze()  # return 1D if possible

def Planck(wl, Tb, em=None):
    """
    Compute blackbody spectral radiance using Planck's law.

    Parameters
    ----------
    wl : array_like
        Wavelength in nanometers (nm)
    Tb : array_like
        Blackbody temperature in Kelvin (K)
    em : array_like, optional
        Emissivity (default = 1 for all elements)

    Returns
    -------
    Lb : ndarray
        Spectral radiance (W m⁻² sr⁻¹ nm⁻¹)
    """
    c1 = 1.191066e-22  # W·m²·sr⁻¹
    c2 = 14388.33      # μm·K

    wl = np.array(wl, dtype=np.float64)
    Tb = np.array(Tb, dtype=np.float64)

    if em is None:
        em = np.ones_like(Tb, dtype=np.float64)
    else:
        em = np.array(em, dtype=np.float64)

    # MATLAB uses wl in nm, but converts to m (×1e-9) and mm (×1e-3)
    Lb = em * c1 * (wl * 1e-9) ** (-5) / (np.exp(c2 / (wl * 1e-3 * Tb)) - 1)

    return Lb

def ephoton(lambda_m, constants):
    h = constants['h']
    c = constants['c']
    return h * c / lambda_m

def e2phot(lambda_m, E, constants):
    """Convert energy flux E (W m-2 per wavelength bin) at wavelength lambda (m)
       to mol photons (mol) corresponding to that E (per same spectral bin).
       Returns mols (not per second conversion factors) consistent with usage in RTMo.
    """
    e = ephoton(lambda_m, constants)
    photons = E / e
    molphotons = photons / constants['A']  # Avogadro
    return molphotons

def volscat(tts, tto, psi, ttli):
    """Translated volscat (vectorized). tts,tto,psi are degrees. ttli: array degrees."""
    deg2rad = np.pi/180.0
    nli = len(ttli)
    psi_rad = psi * deg2rad * np.ones(nli)
    # OR: psi_rad = psi * deg2rad
    cos_psi = np.cos(psi*deg2rad)
    cos_ttli = np.cos(ttli*deg2rad)
    sin_ttli = np.sin(ttli*deg2rad)
    cos_tts = np.cos(tts*deg2rad)
    sin_tts = np.sin(tts*deg2rad)
    cos_tto = np.cos(tto*deg2rad)
    sin_tto = np.sin(tto*deg2rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts
    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = np.maximum(Ss, Cs)
    Ao = np.maximum(So, Co)

    # avoid division by zero issues:
    bts = np.arccos(np.clip(-Cs / (As + 1e-20), -1, 1))
    bto = np.arccos(np.clip(-Co / (Ao + 1e-20), -1, 1))

    chi_o = 2/np.pi * ((bto - np.pi/2) * Co + np.sin(bto) * So)
    chi_s = 2/np.pi * ((bts - np.pi/2) * Cs + np.sin(bts) * Ss)

    delta1 = np.abs(bts - bto)
    delta2 = np.pi - np.abs(bts + bto - np.pi)
    Tot = psi_rad + delta1 + delta2

    bt1 = np.minimum(psi_rad, delta1)
    bt3 = np.maximum(psi_rad, delta2)
    bt2 = Tot - bt1 - bt3

    T1 = 2 * Cs * Co + Ss * So * cos_psi  # NOTE: cos_psi is scalar in original, used as constant
    # The MATLAB version used cos(bt1) and cos(bt3) with angular arrays; replicate roughly:
    T2 = np.sin(bt2) * (2 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

    Jmin = (bt2) * T1 - T2
    Jplus = (np.pi - bt2) * T1 + T2

    frho = Jplus / (2 * np.pi**2)
    ftau = -Jmin / (2 * np.pi**2)

    frho = np.maximum(0.0, frho)
    ftau = np.maximum(0.0, ftau)
    return chi_s, chi_o, frho, ftau

def Psofunction(K, k, LAI, q, dso, xl):
    if dso != 0:
        alf = (dso / q) * 2.0 / (k + K)
        return np.exp((K + k) * LAI * xl + np.sqrt(K * k) * LAI / (alf + 1e-20) * (1 - np.exp(xl * (alf))))
    else:
        return np.exp((K + k) * LAI * xl - np.sqrt(K * k) * LAI * xl)

def calc_reflectances(tau_ss, tau_sd, tau_dd, rho_dd, rho_sd, rs, nl, nwl):
    R_sd = np.zeros((nl+1, nwl))
    R_dd = np.zeros((nl+1, nwl))
    Xsd = np.zeros((nl, nwl))
    Xdd = np.zeros((nl, nwl))
    Xss = np.zeros(nl)
    R_sd[nl, :] = rs
    R_dd[nl, :] = rs
    for j in range(nl-1, -1, -1):
        Xss_j = tau_ss[j]
        dnorm = 1.0 - rho_dd[j, :] * R_dd[j+1, :]
        Xsd[j, :] = (tau_sd[j, :] + tau_ss[j] * R_sd[j+1, :] * rho_dd[j, :]) / (dnorm + 1e-20)
        Xdd[j, :] = tau_dd[j, :] / (dnorm + 1e-20)
        R_sd[j, :] = rho_sd[j, :] + tau_dd[j, :] * (R_sd[j+1, :] * Xss_j + R_dd[j+1, :] * Xsd[j, :])
        R_dd[j, :] = rho_dd[j, :] + tau_dd[j, :] * R_dd[j+1, :] * Xdd[j, :]
    return R_sd, R_dd, Xss, Xsd, Xdd

def calc_fluxprofile(Esun_, Esky_, rs, Xss, Xsd, Xdd, R_sd, R_dd, nl, nwl):
    Es_ = np.zeros((nl+1, nwl))
    Emin_ = np.zeros((nl+1, nwl))
    Eplu_ = np.zeros((nl+1, nwl))
    Es_[0, :] = Esun_.copy()
    Emin_[0, :] = Esky_.copy()
    for j in range(0, nl):
        Es_[j+1, :] = Xss[j] * Es_[j, :]
        Emin_[j+1, :] = Xsd[j, :] * Es_[j, :] + Xdd[j, :] * Emin_[j, :]
        Eplu_[j, :] = R_sd[j, :] * Es_[j, :] + R_dd[j, :] * Emin_[j, :]
    # bottom upward from soil
    Eplu_[nl, :] = rs * (Es_[nl, :] + Emin_[nl, :])
    return Es_, Emin_, Eplu_

def calcTOCirr(atmo, meteo, rdd, rsd, wl, nwl):
    """Simple version: if atmo contains Esun_ & Esky_, use them; otherwise create a
       placeholder using small constants.
    """
    if 'Esun_' in atmo and 'Esky_' in atmo:
        return atmo['Esun_'], atmo['Esky_']
    else:
        # Very simple fallback: Esun as small peaked solar shape; Esky constant low
        Esun_ = np.maximum(1e-6, np.pi * 0.8 * np.ones(nwl))
        Esky_ = np.maximum(1e-6, np.pi * 0.2 * np.ones(nwl))
        # If meteo overrides incoming totals, scale fractionally (approx)
        if meteo.get('Rin', -999) != -999:
            # split optical/thermal by threshold (here use 3 µm)
            J_o = wl < 3.0
            Esun_total = 0.001 * Sint(Esun_[J_o], wl[J_o]) if np.any(J_o) else 1.0
            Esky_total = 0.001 * Sint(Esky_[J_o], wl[J_o]) if np.any(J_o) else 1.0
            Etot = Esun_total + Esky_total
            if Etot > 0:
                fEsun = Esun_[J_o] / (Esun_[J_o] + Esky_[J_o] + 1e-20)
                Esun_[J_o] = fEsun * meteo['Rin']
                Esky_[J_o] = (1 - fEsun) * meteo['Rin']
        return Esun_, Esky_

def RTMo(spectral, atmo, soil, leafopt, canopy, angles, constants, meteo, options):
    """Python translation of RTMo (core parts implemented).
       Returns rad (dict), gap (dict), profiles (dict).
    """
    deg2rad = constants['deg2rad']
    wl = spectral['wlS']  # micrometers
    nwl = len(wl)
    wlPAR = spectral['wlPAR']
    minPAR = wlPAR.min()
    maxPAR = wlPAR.max()
    Ipar = np.where((wl >= minPAR) & (wl <= maxPAR))[0]
    tts = angles['tts']
    tto = angles['tto']
    psi = angles['psi']

    nl = canopy['nlayers']
    litab = canopy['litab']
    lazitab = canopy['lazitab']
    nlazi = canopy['nlazi']
    LAI = canopy['LAI']
    lidf = canopy['lidf']
    xl = canopy['xl']
    dx = 1.0 / nl

    rho = leafopt['refl']   # shape (nl, nwl)
    tau = leafopt['tran']   # shape (nl, nwl)
    kChlrel = leafopt.get('kChlrel', np.zeros_like(rho))
    kCarrel = leafopt.get('kCarrel', np.zeros_like(rho))
    rs = soil['refl']       # shape (nwl,)
    epsc = 1.0 - rho - tau
    epss = 1.0 - rs
    iLAI = LAI / nl

    # initializations
    Rndif = np.zeros(nl)
    Pdif = np.zeros(nl)
    Pndif = np.zeros(nl)
    Rndif_ = np.zeros((nl, nwl))
    Pndif_ = np.zeros((nl, len(wlPAR)))
    Rndif_PAR_ = np.zeros((nl, len(wlPAR)))
    # ... (other arrays as needed)

    # 1. geometric quantities
    cos_tts = np.cos(np.deg2rad(tts))
    tan_tto = np.tan(np.deg2rad(tto))
    cos_tto = np.cos(np.deg2rad(tto))
    sin_tts = np.sin(np.deg2rad(tts))
    tan_tts = np.tan(np.deg2rad(tts))
    psi = abs(psi - 360.0 * np.round(psi / 360.0))
    dso = np.sqrt(tan_tts**2 + tan_tto**2 - 2 * tan_tts * tan_tto * np.cos(np.deg2rad(psi)))

    chi_s, chi_o, frho, ftau = volscat(tts, tto, psi, litab)
    cos_ttlo = np.cos(np.deg2rad(lazitab))
    cos_ttli = np.cos(np.deg2rad(litab))
    sin_ttli = np.sin(np.deg2rad(litab))
    ksli = chi_s / (cos_tts + 1e-20)
    koli = chi_o / (cos_tto + 1e-20)
    sobli = frho * np.pi / (cos_tts * (cos_tto + 1e-20))
    sofli = ftau * np.pi / (cos_tts * (cos_tto + 1e-20))
    bfli = cos_ttli**2

    # integrate over leaf angles using lidf
    k = ksli @ lidf
    K = koli @ lidf
    bf = bfli @ lidf
    sob = sobli @ lidf
    sof = sofli @ lidf

    sdb = 0.5 * (k + bf)
    sdf = 0.5 * (k - bf)
    ddb = 0.5 * (1 + bf)
    ddf = 0.5 * (1 - bf)
    dob = 0.5 * (K + bf)
    dof = 0.5 * (K - bf)

    Css = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts
    cos_deltas = np.outer(Css, np.ones(nlazi)) + np.outer(Ss, np.cos(np.deg2rad(lazitab)))
    fs = np.abs(cos_deltas / (cos_tts + 1e-20))

    # 2 reflectance/transmittance factors (vectors shape [nl, nwl])
    # We allow broadcasting: ddb,d df, sdb etc are scalars -> combine with rho,tau arrays
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau
    a = 1.0 - sigf

    # 3 flux calculation: thin layer direct transmittances etc.
    tau_ss = np.repeat((1.0 - k * iLAI), nl, axis=0) if np.isscalar(k) else np.tile(1.0 - k * iLAI, (nl, 1))
    # tau_dd shape [nl, nwl]
    tau_dd = (1.0 - a * iLAI)
    tau_sd = sf * iLAI
    rho_sd = sb * iLAI
    rho_dd = sigb * iLAI
    # calc reflectances
    R_sd, R_dd, Xss, Xsd, Xdd = calc_reflectances(tau_ss, tau_sd, tau_dd, rho_dd, rho_sd, rs, nl, nwl)
    rdd = R_dd[0, :].copy()
    rsd = R_sd[0, :].copy()

    Esun_, Esky_ = calcTOCirr(atmo, meteo, rdd, rsd, wl, nwl)

    Es_, Emins_, Eplus_ = calc_fluxprofile(Esun_, np.zeros_like(Esky_), rs, Xss, Xsd, Xdd, R_sd, R_dd, nl, nwl)
    Es_d, Emind_, Eplud_ = calc_fluxprofile(np.zeros_like(Esun_), Esky_, rs, Xss, Xsd, Xdd, R_sd, R_dd, nl, nwl)
    Emin_ = Emins_ + Emind_
    Eplu_ = Eplus_ + Eplud_

    # 1.5 probabilities Ps, Po, Pso
    Ps = np.exp(k * xl * LAI)
    Po = np.exp(K * xl * LAI)
    Ps[:-0] = Ps[:-0] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx + 1e-20)
    Po[:-0] = Po[:-0] * (1 - np.exp(-K * LAI * dx)) / (K * LAI * dx + 1e-20)
    q = canopy.get('hot', 1.0)
    Pso = np.zeros_like(Po)
    for j in range(len(xl)):
        Pso[j] = Psofunction(K, k, LAI, q, dso, xl[j])  # note: original used quad - approximated by direct eval
    Pso = np.minimum(Pso, np.minimum(Ps, Po))

    # 3.3 outgoing fluxes in viewing dir due to diffuse
    piLocd_ = (np.sum(vb * Po[:nl][:, None] * Emind_[:nl, :], axis=0) + np.sum(vf * Po[:nl][:, None] * Eplud_[:nl, :], axis=0)) * iLAI
    piLosd_ = rs * (Emind_[nl, :].T * Po[-1])
    piLocu_ = (np.sum(vb * Po[:nl][:, None] * Emins_[:nl, :], axis=0) + np.sum(vf * Po[:nl][:, None] * Eplus_[:nl, :], axis=0) + np.sum(w * Pso[:nl][:, None] * Esun_.T[None, :], axis=0)) * iLAI
    piLosu_ = rs * (Emins_[nl, :].T * Po[-1] + Esun_ * Pso[-1])

    piLod_ = piLocd_ + piLosd_
    piLou_ = piLocu_ + piLosu_
    piLoc_ = piLocu_ + piLocd_
    piLos_ = piLosu_ + piLosd_
    piLo_ = piLoc_ + piLos_
    Lo_ = piLo_ / np.pi
    rso = piLou_ / (Esun_ + 1e-20)
    rdo = piLod_ / (Esky_ + 1e-20)
    Refl = piLo_ / (Esky_ + Esun_ + 1e-20)
    smallE = Esky_ < 2e-4 * np.max(Esky_)
    Refl[smallE] = rso[smallE]

    # 4 net fluxes and PAR
    P_ = e2phot(wl[Ipar] * 1e-6, (Esun_[Ipar] + Esky_[Ipar]), constants)  # convert µm->m
    P = 0.001 * Sint(P_, wl[Ipar])
    EPAR_ = Esun_[Ipar] + Esky_[Ipar]
    EPAR = 0.001 * Sint(EPAR_, wl[Ipar])

    # absorbed solar by leaves per layer
    Asun = np.zeros(nl)
    Pnsun = np.zeros(nl)
    Rnsun_Cab = np.zeros(nl)
    Rnsun_Car = np.zeros(nl)
    Rnsun_PAR = np.zeros(nl)
    Pnsun_Cab = np.zeros(nl)
    Pnsun_Car = np.zeros(nl)
    for j in range(nl):
        Asun[j] = 0.001 * Sint(Esun_ * epsc[j, :], wl)
        Pnsun[j] = 0.001 * Sint(e2phot(wl[Ipar] * 1e-6, Esun_[Ipar] * epsc[j, Ipar], constants), wlPAR)
        # for spectral PAR splits use kChlrel/kCarrel if provided
        Rnsun_Cab[j] = 0.001 * Sint(Esun_[spectral['IwlP']] * epsc[j, spectral['IwlP']] * kChlrel[j, :], spectral['wlP'])
        Rnsun_Car[j] = 0.001 * Sint(Esun_[spectral['IwlP']] * epsc[j, spectral['IwlP']] * kCarrel[j, :], spectral['wlP'])
        Rnsun_PAR[j] = 0.001 * Sint(Esun_[Ipar] * epsc[j, Ipar], wlPAR)
        Pnsun_Cab[j] = 0.001 * Sint(e2phot(np.array(spectral['wlP']) * 1e-6, kChlrel[j, :] * Esun_[spectral['IwlP']] * epsc[j, spectral['IwlP']], constants), spectral['wlP'])
        Pnsun_Car[j] = 0.001 * Sint(e2phot(np.array(spectral['wlP']) * 1e-6, kCarrel[j, :] * Esun_[spectral['IwlP']] * epsc[j, spectral['IwlP']], constants), spectral['wlP'])

    # 4.3 total direct radiation per leaf area (fs times Asun) - options.lite handling simplified
    if options.get('lite', True):
        fs_mean = lidf @ np.mean(fs, axis=1)
        Rndir = fs_mean * Asun[-1]  # simplified
        Pndir = fs_mean * Pnsun[-1]
        # minimal outputs
    else:
        Rndir = None
        Pndir = None

    # 4.4 diffuse radiation per layer and PAR conversions
    Rndif_ = np.zeros((nl, nwl))
    Pndif_ = np.zeros((nl, nwl))
    # Rndif_PAR_ = np.zeros((nl, len(wlPAR)))
    Rndif_PAR_ = np.zeros((nl, len(Ipar)))
    Rndif = np.zeros(nl)
    Pndif = np.zeros(nl)
    for j in range(nl):
        E_ = 0.5 * (Emin_[j, :] + Emin_[j+1, :] + Eplu_[j, :] + Eplu_[j+1, :])
        Pdif[j] = 0.001 * Sint(e2phot(wl[Ipar] * 1e-6, E_[Ipar], constants), wl[Ipar])
        Rndif_[j, :] = E_ * epsc[j, :]
        Pndif_[j, Ipar] = 0.001 * e2phot(wl[Ipar] * 1e-6, Rndif_[j, Ipar], constants)
        Rndif_PAR_[j, :] = Rndif_[j, Ipar]
        Rndif[j] = 0.001 * Sint(Rndif_[j, :], wl)
        Pndif[j] = Sint(Pndif_[j, Ipar], wlPAR)

    Rndirsoil = 0.001 * Sint(Esun_ * epss, wl)
    Rndifsoil = 0.001 * Sint(Emin_[nl, :] * epss, wl)

    Rnhc = Rndif.copy()
    Pnhc = Pndif.copy()

    # profiles (simplified)
    profiles = {}
    if options.get('calc_vert_profiles', False):
        profiles['Pn1d'] = (1 - Ps[:nl]) * Pnhc + Ps[:nl] * (Pndif if options.get('lite', True) else Pndif)
    else:
        profiles = {}

    # assemble outputs
    rad = {}
    gap = {}
    gap['k'] = k
    gap['K'] = K
    gap['Ps'] = Ps
    gap['Po'] = Po
    gap['Pso'] = Pso
    rad['rsd'] = rsd
    rad['rdd'] = rdd
    rad['rdo'] = rdo
    rad['rso'] = rso
    rad['refl'] = Refl
    rad['rho_dd'] = rho_dd
    rad['tau_dd'] = tau_dd
    rad['rho_sd'] = rho_sd
    rad['tau_ss'] = tau_ss
    rad['tau_sd'] = tau_sd
    rad['R_sd'] = R_sd
    rad['R_dd'] = R_dd
    rad['vb'] = vb
    rad['vf'] = vf
    rad['sigf'] = sigf
    rad['sigb'] = sigb
    rad['Esun_'] = Esun_
    rad['Esky_'] = Esky_
    rad['PAR'] = P * 1e6
    rad['EPAR'] = EPAR
    rad['Eplu_'] = Eplu_
    rad['Emin_'] = Emin_
    rad['Emins_'] = Emins_
    rad['Emind_'] = Emind_
    rad['Eplus_'] = Eplus_
    rad['Eplud_'] = Eplud_
    rad['Lo_'] = Lo_
    rad['Eout_'] = Eplu_[0, :]
    rad['Eouto'] = 0.001 * Sint(rad['Eout_'][spectral['IwlP']], spectral['wlP'])
    rad['Eoutt'] = 0.001 * Sint(rad['Eout_'][spectral['IwlT']], spectral['wlT'])
    rad['Lot'] = 0.001 * Sint(Lo_[spectral['IwlT']], spectral['wlT'])
    rad['Rnhs'] = Rndifsoil
    rad['Rnus'] = Rndirsoil
    rad['Rnhc'] = Rnhc
    rad['Rnuc'] = None
    rad['Pnh'] = 1e6 * Pnhc
    rad['Pnu'] = None

    return rad, gap, profiles

# -------------------------
# Example usage (small toy case)
# small wavelength set (micrometers)
wlS = np.array([0.45, 0.55, 0.65, 1.5, 3.5])  # µm (optical + some thermal)
nwl = len(wlS)
wlPAR = np.array([0.4, 0.7])  # PAR range min,max (µm)
# spectral helper indices for P and T bands (toy)
IwlP = np.arange(nwl)  # pretend all wavelengths included in P spectral vector
wlP = wlS.copy()
IwlT = np.where(wlS >= 3.0)[0] if np.any(wlS >= 3.0) else np.array([nwl-1])
wlT = wlS[IwlT]

spectral = {
    'wlS': wlS,
    'wlPAR': wlPAR,
    'IwlP': IwlP,
    'wlP': wlP,
    'IwlT': IwlT,
    'wlT': wlT,
    'IwlP': IwlP
}

# tiny atmosphere: provide Esun_ and Esky_ to skip MODTRAN path
atmo = {
    'Esun_': np.array([1.0, 1.2, 1.0, 0.2, 0.05]),
    'Esky_': np.array([0.1, 0.12, 0.1, 0.05, 0.02])
}

soil = {'refl': np.array([0.2, 0.25, 0.22, 0.3, 0.4])}  # nwl

nl = 3
nli = 5
nlazi = 8
canopy = {
    'nlayers': nl,
    'litab': np.linspace(10, 70, nli),
    'lazitab': np.linspace(0, 350, nlazi),
    'nlazi': nlazi,
    'LAI': 2.0,
    'lidf': np.ones(nli) / nli,
    'xl': np.linspace(0, 1, nl),
    'hot': 0.5
}

# make leaf optics small arrays (nl x nwl)
rho = 0.1 * np.ones((nl, nwl))
tau = 0.05 * np.ones((nl, nwl))
kChlrel = np.zeros((nl, nwl))
kCarrel = np.zeros((nl, nwl))
leafopt = {'refl': rho, 'tran': tau, 'kChlrel': kChlrel, 'kCarrel': kCarrel}

angles = {'tts': 30.0, 'tto': 20.0, 'psi': 10.0}

constants = {
    'deg2rad': np.pi/180.0,
    'h': 6.62607015e-34,
    'c': 2.99792458e8,
    'A': 6.02214076e23
}

meteo = {'Rin': 1000, 'Rli': 300, 'Ta': 15.0}
options = {'lite': True, 'calc_vert_profiles': True}

rad, gap, profiles = RTMo(spectral, atmo, soil, leafopt, canopy, angles, constants, meteo, options)

# show some key outputs
print("gap k, K:", gap['k'], gap['K'])
print("Top-of-canopy reflectance (refl):", np.round(rad['refl'], 4))
print("Directional directional reflectance rso:", np.round(rad['rso'], 4))
print("Incident PAR (micromol m-2 s-1):", rad['PAR'])
print("Profiles keys:", list(profiles.keys()))
