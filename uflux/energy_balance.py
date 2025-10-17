import numpy as np
from scipy.optimize import fsolve

class EnergyBalance:
    """
    A class to compute canopy or leaf-level energy fluxes and solve for
    leaf temperature at energy balance (steady-state conditions).

    This class provides methods to calculate:
        - Sensible heat flux (H)
        - Latent heat flux (LE)
        - Aerodynamic and stomatal resistances
        - Canopy net radiation (R_net)
        - Equilibrium leaf temperature (T_leaf_final) by solving the
          energy balance equation: Rn = H + LE

    The model can be used for simulating energy partitioning between
    radiative, sensible, and latent components for a vegetation canopy
    or individual leaf under given meteorological and biophysical
    conditions.

    Parameters
    ----------
    T_leaf : float
        Initial or measured leaf/canopy temperature [°C].
    Ta : float
        Air temperature [°C].
    VPD : float
        Vapour pressure deficit [hPa].
    shortwave_down : float
        Incoming shortwave radiation (global solar irradiance) [W m⁻²].
    longwave_down : float
        Incoming longwave radiation from the atmosphere [W m⁻²].
    rad : dict
        Dictionary of canopy reflectance components containing:
            - 'rso': Direct-to-observed reflectance
            - 'rdo': Diffuse-to-observed reflectance
            - 'rsd': Direct-to-diffuse reflectance
            - 'rdd': Diffuse-to-diffuse reflectance
    gs : float
        Stomatal conductance [mol m⁻² s⁻¹].

    Attributes
    ----------
    ra : float
        Aerodynamic resistance for heat transfer [s m⁻¹].
    rs : float
        Stomatal resistance [s m⁻¹].
    T_leaf_final : float
        (Optional, computed) Equilibrium leaf temperature [°C] from
        solving the energy balance.

    Methods
    -------
    sensible_heat_flux(T_leaf, Ta, ra)
        Computes the sensible heat flux (H) [W m⁻²].
    latent_heat_flux(T_leaf, VPD, ra, rs)
        Computes the latent heat flux (LE), latent heat of vaporization,
        and slope of the saturation vapour pressure curve.
    calc_aerodynamic_resistance(u, z=2.0, method='log', hc=None, ...)
        Calculates aerodynamic resistance using log-law or empirical formula.
    stomatal_conductance_to_resistance(gs_mol, Ta=25, Patm=101325)
        Converts stomatal conductance (mol m⁻² s⁻¹) to resistance (s m⁻¹).
    calculate_canopy_net_radiation(shortwave_down, longwave_down, T_canopy, rad, ...)
        Computes net shortwave, longwave, and total net radiation.
    _residual(T_leaf)
        Residual energy function f(T_leaf) = Rn - (H + LE), used by solver.
    _solve_equilibrium(T_guess)
        Solves for the equilibrium leaf temperature (°C) where Rn = H + LE.

    Notes
    -----
    - All fluxes are expressed as positive when directed away from the surface.
    - Units are consistent with micrometeorological convention (W m⁻², s m⁻¹).
    - The equilibrium temperature solver (`_solve_equilibrium`) uses
      `scipy.optimize.fsolve` and assumes steady-state energy balance.
    - Typical use case: modelling canopy energy partitioning in ecosystem
      flux studies, remote sensing validation, or vegetation-climate coupling.

    Example
    -------
    from scipy.optimize import fsolve
    rad = {'rso': 0.15, 'rdo': 0.15, 'rsd': 0.17, 'rdd': 0.19}
    eb = EnergyBalance(
        T_leaf=25.0, Ta=20.0, VPD=20.0,
        shortwave_down=750.0, longwave_down=400.0,
        rad=rad, gs=0.33, ra = 448
    )
    T_eq = eb._solve_equilibrium(T_guess=25.0)
    print(f"Equilibrium leaf temperature: {float(T_eq):.2f} °C")
    # Equilibrium leaf temperature: 18.45 °C
    """
    def __init__(self, T_leaf, Ta, VPD, shortwave_down, longwave_down, rad, gs, ra) -> None:
        self.T_leaf = T_leaf
        self.Ta = Ta
        self.VPD = VPD
        self.shortwave_down = shortwave_down
        self.longwave_down = longwave_down
        self.rad = rad
        self.ra = ra
        self.gs = gs

        self.rs = self.stomatal_conductance_to_resistance(self.gs, self.Ta)

    def sensible_heat_flux(self, T_leaf, Ta, ra):
        """
        Compute sensible heat flux (H) [W m⁻²] from temperatures and aerodynamic resistance.

        Parameters
        ----------
        T_leaf : float or np.ndarray
            Leaf/canopy temperature [°C]
        Ta : float or np.ndarray
            Air temperature [°C]
        ra : float or np.ndarray
            Aerodynamic resistance [s m⁻¹]
        physical constants :
            'rhoa' : air density [kg m⁻³]
            'cp'   : specific heat of air [J kg⁻¹ K⁻¹]

        Returns
        -------
        H : float or np.ndarray
            Sensible heat flux [W m⁻²]

        Example:
        -------
        T_leaf = 30.0   # °C
        Ta = 25.0   # °C
        ra = 100.0  # s m⁻¹

        H = sensible_heat_flux(T_leaf, Ta, ra)
        print(f"Sensible heat flux: {H:.2f} W/m²")
        """

        # --------------------------------------------------------------------------------
        # Default constants
        # --------------------------------------------------------------------------------
        rhoa = 1.2047     # kg m⁻³ - air density
        cp = 1004.0        # J kg⁻¹ K⁻¹ - specific heat capacity of dry air

        H = (rhoa * cp / ra) * (T_leaf - Ta)
        return {'H': H}

    def stomatal_conductance_to_resistance(self, gs_mol, Ta, Patm=101325):
        """
        Convert stomatal conductance (mol m⁻² s⁻¹) to stomatal resistance (s m⁻¹).

        Parameters
        ----------
        gs_mol : float or np.ndarray
            Stomatal conductance [mol m⁻² s⁻¹]
        Ta : float
            Air temperature [°C], default 25°C (298.15 K)
        Patm : float
            Air pressure [Pa], default 101325 Pa

        Returns
        -------
        rs : float or np.ndarray
            Stomatal resistance [s m⁻¹]

        Example
        -------
        rs = stomatal_conductance_to_resistance(0.3, 25)
        print(rs)
        """
        Ta_K = Ta + 273.15
        R = 8.314  # J mol⁻¹ K⁻¹
        gs_ms = gs_mol * R * Ta_K / Patm  # convert molar to m s⁻¹
        rs = 1 / gs_ms
        return rs

    def latent_heat_flux(self, T_leaf, VPD, ra, rs, p=101.325):
        """
        Compute latent heat flux (LE) from leaf surface to atmosphere.

        Parameters
        ----------
        T_leaf : float or np.ndarray
            Leaf (canopy) temperature [°C]
        VPD : vapour pressure deficit [hPa]
        ra : float or np.ndarray
            Aerodynamic resistance for heat [s m⁻¹]
        rs : float or np.ndarray
            Stomatal resistance [s m⁻¹]
        p : float, optional
            Atmospheric pressure [hPa], default = 101.325 hPa

        Returns
        -------
        LE : float or np.ndarray
            Latent heat flux [W m⁻²]
        lambda_v : float or np.ndarray
            Latent heat of vaporization [J kg⁻¹]
        s : float or np.ndarray
            Slope of saturation vapour pressure curve [hPa °C⁻¹]

        Example
        -------
        T_leaf = 25.0        # °C
        VPD = 32.0        # hPa
        ra = 100.0       # s m⁻¹
        rs = 200.0       # s m⁻¹
        LE, lambda_v, s = latent_heat_flux(T_leaf, VPD, ra, rs)

        print(f"Latent heat flux (LE): {LE:.2f} W/m²")
        print(f"Latent heat of vaporization (λ): {lambda_v:.2f} J/kg")
        print(f"Slope of sat. vapour curve (s): {s:.4f} hPa/°C")
        """

        # --------------------------------------------------------------------------------
        # Default constants
        # --------------------------------------------------------------------------------
        MH2O = 18.0       # g mol⁻¹ - molecular mass of water
        Mair = 28.96      # g mol⁻¹ - molecular mass of dry air
        rhoa = 1.2047     # kg m⁻³ - air density

        # --------------------------------------------------------------------------------
        # Convert parameters and compute intermediate quantities
        # --------------------------------------------------------------------------------
        ei = self.calc_temperature_to_vapour_pressure(T_leaf)                        # saturation vapour pressure [hPa]
        ea = ei - VPD                                                           # ambient vapour pressure [hPa]
        s = self.calc_slope_saturation_vapour_pressure(ei, T_leaf)                      # slope of saturation curve [hPa °C⁻¹]
        e_to_q = (MH2O / Mair) / p             # conversion factor from vapour pressure [Pa] to specific humidity [kg kg⁻¹ hPa⁻¹]

        qi = ei * e_to_q                       # specific humidity at leaf surface [kg kg⁻¹]
        qa = ea * e_to_q                       # specific humidity of ambient air [kg kg⁻¹]

        lambda_v = (2.501 - 0.002361 * T_leaf) * 1e6   # latent heat of vaporization [J kg⁻¹]

        # --------------------------------------------------------------------------------
        # Latent heat flux [W m⁻²]
        # --------------------------------------------------------------------------------
        LE = rhoa / (ra + rs) * lambda_v * (qi - qa)

        return {'LE': LE, 'lambda_v': lambda_v, 's': s}

    # --------------------------------------------------------------------------------
    # Helper functions for saturation vapour pressure and its slope
    # --------------------------------------------------------------------------------
    def calc_temperature_to_vapour_pressure(self, T):
        """Saturation vapour pressure [hPa]"""
        return 6.107 * 10.0 ** (7.5 * T / (237.3 + T))

    def calc_slope_saturation_vapour_pressure(self, es, T):
        """Slope of saturation vapour pressure curve [hPa °C⁻¹]"""
        return es * 2.3026 * 7.5 * 237.3 / (237.3 + T) ** 2

    def calculate_canopy_net_radiation(self,
            shortwave_down: float, longwave_down: float, T_canopy: float, rad: dict,
            canopy_emissivity: float = 0.98, direct_fraction: float = None
        ) -> dict:
        """
        Calculates the Net Radiation (Rn) for a canopy surface (W m⁻²)
        by summing the Net Shortwave (R_s_net) and Net Longwave (R_l_net) radiation fluxes.

        R_s_net is calculated using the four canopy reflectance components based on
        direct and diffuse incoming radiation partitioning.

        Args:
        -----
            shortwave_down (float): Total incoming shortwave radiation (R_s_down, W m⁻²).
            direct_fraction (float): Fraction of R_s_down that is direct beam (tau_s, dimensionless).
                                    (Diffuse fraction tau_d = 1 - direct_fraction)
            longwave_down (float): Incoming atmospheric longwave radiation (L_down, W m⁻²).
            T_canopy (float): Canopy temperature (°C).
            rso (float): Bidirectional reflectance (Specular-to-Observed).
            rdo (float): Directional reflectance for diffuse incidence (Diffuse-to-Observed).
            rsd (float): Diffuse reflectance for specular incidence (Specular-to-Diffuse).
            rdd (float): Diffuse reflectance for diffuse incidence (Diffuse-to-Diffuse).
            canopy_emissivity (float, optional): Longwave emissivity of the canopy (dimensionless).

        Returns:
        --------
            dict: A dictionary containing R_net, R_s_net, and R_l_net.

        Example:
        --------

        # Typical Midday Inputs
        shortwave_down = 800.0  # Total incoming shortwave (W m⁻²)
        longwave_down = 350.0    # Incoming atmospheric longwave (W m⁻²)
        T_canopy = 28.0   # Canopy temperature (°C)
        # direct_fraction = 0.75   # 75% direct, 25% diffuse

        # Reflectance Components (Note: these should be less than 1.0)
        rad = {
            'rso': 0.05,
            'rdo': 0.15,
            'rsd': 0.10,
            'rdd': 0.25
        }

        # Run the calculation
        results = calculate_canopy_net_radiation(
            shortwave_down=shortwave_down,
            longwave_down=longwave_down,
            T_canopy=T_canopy,
            rad = rad,
            canopy_emissivity=0.98
        )

        print("--- Canopy Net Radiation Calculation ---")
        print(f"Canopy Temperature: {T_canopy:.2f} °C")
        print(f"Canopy Albedo: {results['canopy_albedo']:.3f}")
        print(f"Net Shortwave (R_s_net): {results['R_s_net']:.2f} W m⁻²")
        print(f"Net Longwave (R_l_net): {results['R_l_net']:.2f} W m⁻²")
        print(f"Total Net Radiation (R_net): {results['R_net']:.2f} W m⁻²")
        """

        # --- Constants ---
        # Stefan-Boltzmann constant (W m⁻² K⁻⁴)
        SIGMA = 5.67e-8
        # Convert T_canopy to Kelvin for radiation calculations
        T_canopy_K = T_canopy + 273.15

        # --- 1. Net Shortwave Radiation (R_s_net) ---

        # Calculate Direct (S) and Diffuse (D) components of incoming shortwave
        if not direct_fraction:
            direct_beam = rad['rso'] + rad['rsd']
            diffuse_beam = rad['rdo'] + rad['rdd']
            direct_fraction = direct_beam / (direct_beam + diffuse_beam)
        tau_s = direct_fraction
        tau_d = 1.0 - direct_fraction

        # Canopy Albedo (rho_canopy)
        # The albedo is the weighted average of the four reflectances, where the weighting
        # is based on the source (direct/diffuse) and the measured output (total reflected)

        # Reflection of Direct beam (S -> Observed)
        rho_direct = tau_s * rad['rso'] + tau_d * rad['rsd']

        # Reflection of Diffuse beam (D -> Observed)
        rho_diffuse = tau_s * rad['rdo'] + tau_d * rad['rdd']

        # Total Canopy Albedo (weighted by incoming fractions)
        rho_canopy = tau_s * rho_direct + tau_d * rho_diffuse

        # Net Shortwave Absorbed (R_s_net)
        R_s_net = shortwave_down * (1.0 - rho_canopy)

        # --- 2. Net Longwave Radiation (R_l_net) ---

        # Outgoing Longwave Radiation (L_up)
        # The canopy radiates as a gray body at T_canopy
        L_up = canopy_emissivity * SIGMA * (T_canopy_K**4)

        # Net Longwave Radiation
        R_l_net = longwave_down - L_up

        # --- 3. Total Net Radiation (R_net) ---

        R_net = R_s_net + R_l_net

        # =============================================================================
        # Sanity Check and Approx. Net Radiation (Rn)
        # =============================================================================
        # Definition:
        #   Net radiation (Rn) is the net balance between incoming and outgoing
        #   shortwave and longwave radiation at the surface:
        #
        #       Rn = (Rs_down - Rs_up) + (Rl_down - Rl_up)
        #
        # Typical daytime values (W/m²):
        #   - Clear midday:        600–900
        #   - Normal sunny day:    400–700
        #   - Cloudy day:          200–400
        #   - Overcast/evening:    <200
        #   - Nighttime cooling:   −20 to −100
        #
        # Typical daily mean Rn (W/m²):
        #   - Tropical forest:      150–250
        #   - Temperate grassland:  100–200
        #   - Desert/arid land:      80–180
        #   - Water/irrigated crop: 120–220
        #   - Winter high-latitude:  30–100
        #
        # Approximate empirical formula:
        #   Rn ≈ (1 - α) * Rs - ϵ * σ * (Ta + 273.15)**4 * (1 - 0.26 * ea_kPa**(1/7))
        #
        # Units:
        #   Rn : Net radiation [W/m²]
        #   Rs : Incoming shortwave radiation [W/m²]
        #   α  : Surface albedo (shortwave reflectance) [dimensionless]
        #        Typical 0.15–0.25 for vegetation
        #   ϵ  : Surface longwave emissivity [dimensionless]
        #        Typical ≈ 0.98
        #   σ  : Stefan–Boltzmann constant [5.67 × 10⁻⁸ W·m⁻²·K⁻⁴]
        #   Ta : Air temperature [°C]
        #   ea : Actual vapor pressure [kPa]
        #
        # IMPORTANT:
        #   If your vapor pressure (ea) is in hPa, convert to kPa before use:
        #       ea_kPa = ea_hPa / 10
        #
        # Example:
        #   Rn = (1 - 0.23) * Rs - 0.98 * 5.67e-8 * (Ta + 273.15)**4 * (1 - 0.26 * (ea_hPa / 10)**(1/7))
        # =============================================================================
        if R_net > 1000:
            ea = self.calc_temperature_to_vapour_pressure(T_canopy) - self.VPD
            R_net = (1 - rho_canopy) * shortwave_down - 0.98 * 5.67e-8 * (T_canopy + 273.15)**4 * (1 - 0.26 * (ea / 10)**(1/7))
            R_s_net = R_net * R_s_net / (R_s_net + R_l_net)
            R_l_net = R_net - R_s_net

        return {
            "R_net": R_net,
            "R_s_net": R_s_net,
            "R_l_net": R_l_net,
            "canopy_albedo": rho_canopy
        }

    def _residual(self, T_leaf):
        """
        Residual energy (W m⁻²) to be minimized:
            f(T_leaf) = Rn - (H + LE)
        where positive means net surplus of energy (leaf should warm up).
        """
        H = self.sensible_heat_flux(T_leaf, self.Ta, self.ra)['H']
        LE = self.latent_heat_flux(T_leaf, self.VPD, self.ra, self.rs)['LE']
        self.Rn = self.calculate_canopy_net_radiation(self.shortwave_down, self.longwave_down, T_leaf, self.rad)['R_net']
        if np.abs(self.Ta - T_leaf) > 10:
            # Return a large residual so fsolve moves away from this value
            return 1e6
        return np.array(self.Rn - (H + LE))

    def _solve_equilibrium(self, T_guess):
        """
        Solve for equilibrium leaf temperature (°C).
        """
        try:
            T_leaf_final, infodict, ier, mesg = fsolve(
                self._residual, T_guess, full_output=True
            )
            # # Use solution only if solver converged successfully (ier == 1)
            # T_leaf_final = T_leaf_final if ier == 1 else self.Ta + 1.0
            if np.abs(T_leaf_final - self.Ta) > 30:
                print(f'Sanity check failed, T_leaf_final is {T_leaf_final} degC.')
                # =============================================================================
                # Approximate empirical relation (for sanity check):
                # =============================================================================
                #     T_leaf = Ta + a*Rn - b*gs - c*ws
                #   where:
                #       a ≈ 0.01  (K·m²/W)      → radiative heating term
                #       b ≈ 2.0   (K·(mol m⁻² s⁻¹)⁻¹) → stomatal cooling term
                #       c ≈ 0.3   (K·s/m)       → wind cooling term
                #
                # Summary:
                #   - Clear, calm, dry day:       +3 to +7 °C
                #   - Cloudy or breezy day:       0 to +2 °C
                #   - Humid, high transpiration:  −2 to 0 °C
                #   - Nighttime radiative loss:   −2 to −5 °C
                T_leaf_final = self.Ta + 0.01* self.Rn - 2 * self.gs - 0.3 * 1 # defual wind speed as 1 
        except Exception as e:
            print(e)
            # Fallback if solver fails — assume leaf is slightly warmer than air
            T_leaf_final = self.Ta + 0.01* self.Rn - 2 * self.gs - 0.3 * 1 # defual wind speed as 1 
        return T_leaf_final