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

# Example
# -------
# ra = calc_aerodynamic_resistance(u=1, z=2.0, method='log')
# print(ra)