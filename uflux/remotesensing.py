import numpy as np

def campbell_k(lad=1.0, sza_deg=30.0):
    """
    Campbell (1986)-style approximation for extinction coefficient k(θ).
    lad: leaf angle distribution parameter (1 = spherical; <1 planophile; >1 erectophile)
    sza_deg: solar zenith angle in degrees
    """
    theta = np.radians(sza_deg)
    num = np.sqrt(lad**2 + np.tan(theta)**2)
    denom = lad + 1.774 * (lad + 1.182)**-0.733
    return num / denom

def LAI_to_FAPAR(lai, k=None, lad=None, sza_deg=None, clumping_index=1.0):
    """
    Calculate FAPAR from LAI using Beer–Lambert law.

    Parameters
    ----------
    lai : float or np.ndarray
        Leaf area index (m² m⁻²)
    k : float, optional
        Extinction coefficient. If None, computed from lad and sza_deg.
    lad : float, optional
        Leaf angle distribution parameter for Campbell model.
    sza_deg : float, optional
        Solar zenith angle (degrees) for Campbell model.
    clumping_index : float, optional
        Clumping correction factor (Ω, 0<Ω≤1). Default 1.0 (no clumping).

    Returns
    -------
    fapar : float or np.ndarray
        Fraction of absorbed PAR (0–1)

    Example
    -------
    fapar = LAI_to_FAPAR(1, k = canopy_model.canopopt['ext']['k_ext_obs'])
    print(fapar)
    """
    lai = np.asarray(lai, dtype=float)
    lai_eff = clumping_index * lai

    # Compute k if not provided
    if k is None:
        if lad is None or sza_deg is None:
            k = 0.5  # typical spherical case
        else:
            k = campbell_k(lad, sza_deg)

    fapar = 1.0 - np.exp(-k * lai_eff)
    return np.clip(fapar, 0.0, 1.0)

# 