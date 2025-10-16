def calc_latent_energy_flux(gs):
    """
    Calculating latent heat fluxes.

    Parameters
    ----------
    gs : float
        Stomatal conductance to water vapor (mol m⁻² s⁻¹).
    wind : float
        Wind speed (m s⁻¹).
    Returns
    -------
    LE : float
        Latent heat flux (W m⁻²).
    Notes
    -----
        - latent heat flux LE = λ_v * g_v
    Simplifications:
        - Vapor pressure deficit effects on LE are lumped into a constant scaling.
    """
    lambda_v = 2.45e6    # Latent heat of vaporization of water (J kg⁻¹)
    g_v = 1.6 * gs       # Convert stomatal conductance for CO₂ to H₂O (mol m⁻² s⁻¹)
    LE = lambda_v * g_v * 1e-3  # Convert from mol to kg units approximately
    return LE

def calc_sensible_heat_flux(T_leaf, Ta, ws, gb0=0.2):
    """
    Calculating sensible heat fluxes.

    Parameters
    ----------
    T_leaf : float
        Leaf temperature (°C).
    Ta : float
        Air (ambient) temperature (°C)
    ws : float
        Wind speed (m s⁻¹).
    gb0 : float, optional
        Baseline boundary-layer conductance (mol m⁻² s⁻¹), default is 0.2.
    Returns
    -------
    H : float
        Sensible heat flux (W m⁻²).
    Notes
    -----
        - H : sensible heat flux = ρ_air * c_p * (T_leaf - T_air) * g_b
    """
    rho_air = 1.2       # Air density (kg m⁻³)
    cp = 1010            # Specific heat of air (J kg⁻¹ K⁻¹)
    gb = gb0 + 0.01 * ws  # Boundary layer conductance increases with wind
    H = rho_air * cp * (T_leaf - Ta) * gb
    return H

def leaf_energy_balance(T_leaf, Ta, APAR_leaf, gs, ws, gb0=0.2):
    """
    Solve for leaf temperature (Tleaf) by balancing absorbed radiation, sensible heat,
    and latent heat fluxes under given environmental conditions.

    Parameters
    ----------
    T_leaf : float
        Initial guess for leaf temperature (°C).
    Ta : float
        Air (ambient) temperature (°C).
    APAR_leaf : float
        Absorbed photosynthetically active radiation by the leaf (W m⁻²).
    gs : float
        Stomatal conductance to water vapor (mol m⁻² s⁻¹).
    ws : float
        Wind speed (m s⁻¹).
    gb0 : float, optional
        Baseline boundary-layer conductance (mol m⁻² s⁻¹), default is 0.2.

    Returns
    -------
    tleaf_final : float
        Solved leaf temperature (°C) that satisfies energy balance.
    H : float
        Sensible heat flux (W m⁻²).
    LE : float
        Latent heat flux (W m⁻²).

    Notes
    -----
    The energy balance for the leaf is given by:

        Rn = H + LE

    where:
        - Rn: net radiation absorbed by the leaf (W m⁻²)
        - H : sensible heat flux = ρ_air * c_p * (T_leaf - T_air) * g_b
        - LE: latent heat flux = λ_v * g_v

    The solution is obtained iteratively using `scipy.optimize.fsolve`.

    Simplifications:
        - Longwave exchange with the environment is approximated within (1 - 0.15) factor.
        - Boundary layer conductance increases linearly with wind speed.
        - Vapor pressure deficit effects on LE are lumped into a constant scaling.
    """
    # Assume 15% reflectance and thermal losses already approximated
    Rn = APAR_leaf * (1 - 0.15)

    # --- Define the energy balance residual function ---
    def residual(T_leaf):
        """
        Residual energy (W m⁻²) to be minimized:
            f(T_leaf) = Rn - (H + LE)
        where positive means net surplus of energy (leaf should warm up).
        """
        H = calc_sensible_heat_flux(T_leaf, Ta, ws, gb0=0.2)
        LE = calc_latent_energy_flux(gs)
        return np.array(Rn - (H + LE))

    # --- Solve for leaf temperature using fsolve ---
    try:
        T_leaf_final, infodict, ier, mesg = fsolve(
            residual, T_leaf, full_output=True
        )
        # Use solution only if solver converged successfully (ier == 1)
        T_leaf_final = T_leaf_final if ier == 1 else Ta + 1.0
    except Exception:
        # Fallback if solver fails — assume leaf is slightly warmer than air
        T_leaf_final = Ta + 1.0

    # --- Compute final fluxes using solved leaf temperature ---
    H = calc_sensible_heat_flux(T_leaf_final, Ta, ws, gb0=0.2)
    LE = calc_latent_energy_flux(gs)

    return T_leaf_final, H, LE


# # ===============================
# # 🌿 Example usage
# # ===============================

# # Environmental inputs
# Ta = 25.0           # Air temperature (°C)
# APAR_leaf = 400.0  # Absorbed PAR (W m⁻²)
# gs = 0.3              # Stomatal conductance (mol m⁻² s⁻¹)
# ws = 1.0            # Wind speed (m s⁻¹)
# T_leaf_guess = 26.0    # Initial guess for leaf temperature (°C)

# # Solve energy balance
# T_leaf_final, H, LE = leaf_energy_balance(T_leaf_guess, Ta, APAR_leaf, gs, ws)

# # Print results
# print("=== Leaf Energy Balance Results ===")
# print(f"Air temperature (°C): {Ta:.2f}")
# print(f"Solved leaf temperature (°C): {T_leaf_final}")
# print(f"Sensible heat flux H (W m⁻²): {H}")
# print(f"Latent heat flux LE (W m⁻²): {LE}")
