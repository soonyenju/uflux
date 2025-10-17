import numpy as np

def calc_aerodynamic_resistance(u, z=2.0, method='log', hc=None, z0m=None, z0h=None, d=None, k=0.41):
    """
    Calculate aerodynamic resistance for heat ra [s m^-1].

    Parameters
    ----------
    u : float or array-like
        Wind speed at reference height z [m s^-1]. Must be > 0.
    z : float
        Reference measurement height [m] (default 2.0 m).
    method : {'log', 'empirical'}
        'log'     -> use log-law (neutral conditions) with momentum & heat roughness lengths.
        'empirical' -> simple empirical formula ra = C / u (C in s).
    hc : float or None
        Canopy height [m]. If provided and z0m/z0h/d not supplied, defaults are computed:
            d   = 0.67 * hc
            z0m = 0.123 * hc
            z0h = z0m / 10
    z0m : float or None
        Aerodynamic roughness length for momentum [m] (overrides hc derived value).
    z0h : float or None
        Roughness length for heat [m]; if None the code uses z0m/10.
    d : float or None
        Zero-plane displacement height [m]; if None and hc given, uses 0.67*hc; else 0.
    k : float
        von Kármán constant (default 0.41).

    Returns
    -------
    ra : float or np.ndarray
        Aerodynamic resistance for heat [s m^-1].

    Example
    -------
    # Example 1: log law for a 1 m canopy, wind measured at 2 m (u=2 m/s)
    ra1 = calc_aerodynamic_resistance(u=2.0, z=2.0, method='log', hc=1.0)
    print("ra (log, hc=1 m, u=2 m/s):", ra1, "s m^-1")

    # Example 2: bare soil with small roughness (z0m=0.001 m), u=3 m/s
    ra2 = calc_aerodynamic_resistance(u=3.0, z=2.0, method='log', z0m=0.001, z0h=0.0001, d=0.0)
    print("ra (log, bare soil, u=3 m/s):", ra2, "s m^-1")

    # Example 3: empirical ra = 208 / u
    ra3 = calc_aerodynamic_resistance(u=np.array([0.5, 1.0, 2.0, 5.0]), method='empirical')
    print("ra (empirical 208/u):", ra3)

    # Example 4: custom
    # ra = calc_aerodynamic_resistance(u=1, z=2.0, method='log')
    # print(ra)

    Notes
    -----
    - 'log' method assumes neutral stratification (Monin-Obukhov length large).
    - Be careful when u is very small (near-zero): ra -> large. The function will raise for u<=0.
    - Typical vegetation defaults (if hc provided): d=0.67*hc, z0m=0.123*hc, z0h=z0m/10.
    - Empirical constant examples: C=208 (for some surface types) gives ra ~ 208/u.
    Users should choose an empirical C based on literature or calibration.
    """
    u = np.asarray(u, dtype=float)
    if np.any(u <= 0):
        raise ValueError("Wind speed u must be > 0 m/s for ra calculation.")

    if method == 'log':
        # set defaults from canopy height if provided
        if hc is not None and (z0m is None):
            z0m = 0.123 * hc
        if d is None:
            if hc is not None:
                d = 0.67 * hc
            else:
                d = 0.0
        if z0m is None:
            # fallback small roughness for bare/open land
            z0m = 0.001  # m, smooth bare soil / short grass (user can override)
        if z0h is None:
            z0h = max(z0m / 10.0, 1e-6)  # often z0h ~= z0m/10, avoid zero

        # ensure arguments make sense: (z - d) must be > z0m and > z0h
        z_minus_d = z - d
        if np.any(z_minus_d <= 0):
            raise ValueError("Reference height z must be greater than zero-plane displacement d.")

        # log-law formula for neutral conditions (resistance for heat)
        # ra = ( ln((z - d)/z0m) * ln((z - d)/z0h) ) / (k^2 * u)
        ln_m = np.log(z_minus_d / z0m)
        ln_h = np.log(z_minus_d / z0h)
        ra = (ln_m * ln_h) / (k**2 * u)
        return ra

    elif method == 'empirical':
        # Simple empirical formula: ra = C / u
        # choose default C ~ 208 s (example). User can change C by passing z0m as C if desired.
        # (We reuse z0m argument as container for C if provided)
        C = 208.0 if z0m is None else float(z0m)
        ra = C / u
        return ra

    else:
        raise ValueError("method must be 'log' or 'empirical'.")


def calc_sensible_heat_flux_air(Ta, u, Rn=None, G=0.1, beta=None, P=101325):
    """
    Estimate sensible heat flux (W/m²) using simple bulk or energy balance method.

    Parameters
    ----------
    Ta : float
        Air temperature (°C)
    u : float
        Wind speed (m/s)
    Rn : float, optional
        Net radiation (W/m²). Required if using Bowen ratio method.
    G : float, optional
        Fraction of Rn going into soil heat (default 0.1)
    beta : float, optional
        Bowen ratio (H/LE). If None, uses bulk transfer approximation.
    P : float, optional
        Air pressure (Pa). Default = 101325 Pa.

    Returns
    -------
    H : float
        Sensible heat flux (W/m²)

    Examples
    --------
    # Example 1: Using bulk transfer
    Ta = 11     # °C
    u = 0.73       # m/s
    H_bulk = calc_sensible_heat_flux_air(Ta, u)
    print(f"Sensible heat flux (bulk): {H_bulk:.1f} W/m²")

    # Example 2: Using Bowen ratio + net radiation
    Rn = 1000    # W/m²
    beta = 0.1  # typical for vegetated surfaces
    H_bowen = calc_sensible_heat_flux_air(Ta, u, Rn=Rn, beta=beta)
    print(f"Sensible heat flux (Bowen): {H_bowen:.1f} W/m²")
    """
    # Constants
    cp = 1005.0  # specific heat of air (J/kg/K)
    R = 287.05   # gas constant for dry air (J/kg/K)
    Tk = Ta + 273.15

    # Air density (kg/m³)
    rho = P / (R * Tk)

    if beta is not None and Rn is not None:
        # Energy balance approach: H = beta / (1 + beta) * (Rn - G*Rn)
        H = (beta / (1 + beta)) * (Rn * (1 - G))
    else:
        # Bulk transfer method: approximate (Ts - Ta) from empirical 1–3°C
        delta_T = 2.0  # assumed temperature difference in K
        gh = 0.0025 * u  # conductance for heat (m/s)
        H = rho * cp * gh * delta_T

    return H


def calc_latent_heat_flux_air(Ta, VPD_hPa, u, P=101325):
    """
    Calculate latent heat flux (W/m²) using simplified bulk transfer,
    using vapor pressure deficit (VPD) instead of RH.

    Parameters
    ----------
    Ta : float
        Air temperature in °C
    VPD_hPa : float
        Vapor pressure deficit in hPa
    u : float
        Wind speed at reference height (m/s)
    P : float
        Air pressure in Pa (default 101325 Pa)

    Returns
    -------
    LE : float
        Latent heat flux in W/m²
    
    Example
    -------
    Ta = 11          # °C
    VPD = 3.53   # hPa
    ws = 0.73          # m/s
    LE = calc_latent_heat_flux_air(Ta, VPD, ws)
    print(f"Latent heat flux: {LE:.1f} W/m²")
    """
    # Constants
    Lv = 2.45e6  # Latent heat of vaporization (J/kg)
    
    # Convert temperature to Kelvin
    Tk = Ta + 273.15

    # Saturation vapor pressure (Pa) over water
    es = 610.78 * np.exp(Ta * 17.27 / (Ta + 237.3))  # Pa

    # Convert VPD from hPa to Pa
    VPD_Pa = VPD_hPa * 100.0  # Pa

    # Actual vapor pressure (Pa)
    ea = es - VPD_Pa  # ea = es - VPD

    # Saturation specific humidity (kg/kg)
    qs = 0.622 * es / (P - 0.378 * es)

    # Actual specific humidity (kg/kg)
    qa = 0.622 * ea / (P - 0.378 * ea)

    # Bulk transfer coefficient for water vapor (simplified)
    g_v = 0.0025 * u  # m/s

    # Latent heat flux (W/m²)
    LE = Lv * g_v * (qs - qa)

    return LE
    

def calc_RH_from_VPD(T_air, VPD):
    """
    Calculate relative humidity (RH) from air temperature (°C) and VPD in hPa.

    Parameters
    ----------
    T_air : float or np.array
        Air temperature in Celsius.
    VPD : float or np.array
        Vapor pressure deficit in hPa.

    Returns
    -------
    RH : float or np.array
        Relative humidity (0-1)

    Example
    -------
    T_air = 25.0    # °C
    VPD_hPa = 12.0  # hPa

    RH = calc_RH_hPa(T_air, VPD_hPa)
    print(f"Relative Humidity: {RH*100:.1f}%")
    """
    # Saturation vapor pressure in hPa
    e_s = 6.108 * np.exp(17.27 * T_air / (T_air + 237.3))  # hPa

    # Actual vapor pressure
    e_a = e_s - VPD

    # RH as fraction
    RH = np.clip(e_a / e_s, 0, 1)

    return RH
