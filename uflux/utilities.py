import numpy as np

# ========================================================================================================================
# Function shortwave_down_to_APAR
# ========================================================================================================================

def shortwave_down_to_APAR(shortwave_down: float, fAPAR: float) -> tuple:
    """
    Convert incoming shortwave radiation (SW) to absorbed PAR (APAR)
    in both µmol and mol photon units.

    Parameters
    ----------
    shortwave_down : float
        Incoming shortwave radiation (W m⁻²).
    fAPAR : float
        Fraction of absorbed Photosynthetically Active Radiation (dimensionless, 0–1).

    Returns
    -------
    tuple
        (I_abs, I_abs_mol)
        - I_abs : Absorbed PAR (µmol photons m⁻² s⁻¹)
        - I_abs_mol : Absorbed PAR (mol photons m⁻² s⁻¹)

    Notes
    -----
    - Assumes ~50% of total shortwave radiation is within the PAR range (400–700 nm).
    - Uses the standard conversion: 1 W m⁻² ≈ 4.57 µmol photons m⁻² s⁻¹ for sunlight at ~550 nm.

    Examples
    --------
    shortwave_down = 1651  # W m⁻²
    fAPAR = 0.3            # fraction (0–1)

    I_abs_umol, I_abs_mol = shortwave_down_to_APAR(shortwave_down, fAPAR)

    print("=== Shortwave to Absorbed PAR ===")
    print(f"Absorbed PAR (µmol m⁻² s⁻¹): {I_abs_umol:.2f}")
    print(f"Absorbed PAR (mol m⁻² s⁻¹): {I_abs_mol:.6f}")
    """

    # -----------------------------
    # Step 1: Convert shortwave (W m⁻²) to total PAR (µmol photons m⁻² s⁻¹)
    # -----------------------------
    # Multiply by 0.5 because ~50% of solar shortwave energy is within the PAR region.
    # Multiply by 4.57 to convert W m⁻² to µmol photons m⁻² s⁻¹.
    PAR_total = shortwave_down * 0.5 * 4.57

    # Absorbed PAR (µmol photons m⁻² s⁻¹)
    I_abs_umol = fAPAR * PAR_total

    # -----------------------------
    # Step 2: Convert from µmol to mol photons per m² per second
    # -----------------------------
    I_abs_mol = I_abs_umol * 1e-6  # 1 µmol = 1e⁻⁶ mol

    return I_abs_umol, I_abs_mol


# ========================================================================================================================
# Function calc_GPP_LUE
# ========================================================================================================================


def calc_GPP_LUE(APAR: float, LUE: float, day_length: float = 43200) -> float:
    """
    Calculate daily Gross Primary Productivity (GPP) using the Light Use Efficiency (LUE) approach.

    Parameters
    ----------
    APAR : float
        Absorbed Photosynthetically Active Radiation (mol photons m⁻² s⁻¹).
        Typically derived from `shortwave_down_to_APAR`.
    LUE : float
        Light Use Efficiency (g C mol⁻¹ photons).
        Represents the amount of carbon fixed per mol of absorbed photons.
    day_length : float, optional
        Day length (s). Default = 43,200 s (12 hours).

    Returns
    -------
    float
        GPP in grams of carbon per square meter per day (g C m⁻² day⁻¹).

    Notes
    -----
    - This function assumes steady daytime photosynthesis rate throughout the day.
    - GPP = APAR × LUE × day_length
      where:
        APAR (mol photons m⁻² s⁻¹)
        LUE  (g C mol⁻¹ photons)
        day_length (s)

    Example
    -------
    APAR = 0.001   # mol photons m⁻² s⁻¹ (from shortwave_down_to_APAR)
    LUE = 0.276       # g C mol⁻¹ photons
    day_length = 12 * 3600  # 12 hours = 43200 s

    GPP = calc_GPP_LUE(APAR, LUE, day_length)

    print("=== Gross Primary Productivity (LUE model) ===")
    print(f"GPP = {GPP:.2f} g C m⁻² day⁻¹")
    """

    # Multiply absorbed PAR by efficiency (carbon fixed per photon)
    # and integrate over daylight duration to get daily GPP.
    GPP = APAR * LUE * day_length  # g C m⁻² day⁻¹

    return GPP
