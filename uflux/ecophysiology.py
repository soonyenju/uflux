import numpy as np

# ========================================================================================================================
# Module FvCB
# ========================================================================================================================

class FvCB:
    """
    Farquhar-von Caemmerer-Berry photosynthesis model supporting both C3 and C4.

    Parameters
    ----------
    mode : str
        'C3' or 'C4'
    Vcmax : float
        Maximum carboxylation rate (µmol m⁻² s⁻¹)
    Jmax : float
        Maximum electron transport rate (µmol m⁻² s⁻¹)
    Rd : float
        Day respiration (µmol m⁻² s⁻¹)
    # C4-specific
    Vpmax : float, optional
        Maximum PEP carboxylation rate (µmol m⁻² s⁻¹)
    Kp : float, optional
        Michaelis-Menten constant for CO₂ in PEPc (µmol mol⁻¹)
    gbs : float, optional
        Bundle sheath conductance (mol m⁻² s⁻¹)
    Kc : float
        Michaelis-Menten constant for CO₂ in Rubisco (µmol mol⁻¹)
    Ko : float
        Michaelis-Menten constant for O₂ in Rubisco (mmol mol⁻¹)
    O : float
        Oxygen concentration (mmol mol⁻¹)
    Gamma_star : float
        CO₂ compensation point (µmol mol⁻¹)
    """

    def __init__(self, mode, Vcmax, Jmax, Rd, Kc, Ko, O, Gamma_star, Vpmax=None, Kp=None, gbs=None):
        self.mode = mode.upper()
        self.Vcmax = Vcmax
        self.Jmax = Jmax
        self.Rd = Rd
        self.Kc = Kc
        self.Ko = Ko
        self.O = O
        self.Gamma_star = Gamma_star
        # C4 optional
        self.Vpmax = Vpmax
        self.Kp = Kp
        self.gbs = gbs

        if self.mode not in ['C3', 'C4']:
            raise ValueError("mode must be 'C3' or 'C4'")
        if self.mode == 'C4' and (Vpmax is None or Kp is None or gbs is None):
            raise ValueError("C4 mode requires Vpmax, Kp, and gbs parameters")

    # ------------------- Internal methods -------------------

    def _rubisco_rate(self, Ci):
        """Rubisco-limited assimilation (C3 or C4)."""
        return self.Vcmax * (Ci - self.Gamma_star) / (Ci + self.Kc * (1 + self.O / self.Ko))

    def _electron_transport(self, PAR, Ci=None):
        """Electron transport rate J (µmol m⁻² s⁻¹) using non-rectangular hyperbola."""
        if self.mode == 'C3':
            alpha = 0.24
        else:
            alpha = 0.4
        theta = 0.7
        J = (alpha * PAR + self.Jmax - np.sqrt((alpha * PAR + self.Jmax)**2 - 4 * theta * alpha * PAR * self.Jmax)) / (2 * theta)
        return J

    def _PEPc_rate(self, Ci):
        """PEP carboxylase rate for C4 only."""
        return self.Vpmax * Ci / (Ci + self.Kp)

    def _A_C3(self, Ci, PAR):
        J = self._electron_transport(PAR, Ci)
        Ac = self._rubisco_rate(Ci)
        Aj = J * (Ci - self.Gamma_star) / (4 * (Ci + 2 * self.Gamma_star))
        return np.minimum(Ac, Aj) - self.Rd

    def _A_C4(self, Ci, PAR):
        Vp = self._PEPc_rate(Ci)
        Cs = Vp / self.gbs + Ci
        Ac = self._rubisco_rate(Cs)
        J = self._electron_transport(PAR)
        Aj = J / 2
        return np.minimum(Ac, Aj) - self.Rd

    # ------------------- Public method -------------------

    def A_net(self, Ci, PAR):
        """
        Compute net assimilation rate (A_net).

        Parameters
        ----------
        Ci : float
            Intercellular CO2 concentration (µmol mol⁻¹)
        PAR : float
            Photosynthetically active radiation (µmol m⁻² s⁻¹)

        Returns
        -------
        float
            Net photosynthesis rate (µmol m⁻² s⁻¹)
        """
        if self.mode == 'C3':
            return self._A_C3(Ci, PAR)
        else:
            return self._A_C4(Ci, PAR)

# ------------------------------------------------------------------------------------------------------------------------
# # Example

# # C3
# c3_model = FvCB(mode='C3', Vcmax=60, Jmax=120, Rd=1, Kc=404, Ko=248, O=210, Gamma_star=42)
# print("C3 A_net:", c3_model.A_net(Ci=200, PAR=1500))

# # C4
# c4_model = FvCB(mode='C4', Vcmax=50, Jmax=120, Rd=1, Vpmax=100, Kp=80, gbs=0.003,
#                 Kc=404, Ko=248, O=210, Gamma_star=42)
# print("C4 A_net:", c4_model.A_net(Ci=200, PAR=1500))

# ========================================================================================================================
# Module BallBerry
# ========================================================================================================================

class BallBerry:
    """
    Ball-Berry stomatal conductance model.

    g_s = g0 + g1 * (A * RH / Cs)

    Parameters
    ----------
    g0 : float
        Residual stomatal conductance (mol m⁻² s⁻¹)
    g1 : float
        Empirical slope (unitless)
    """

    def __init__(self, g0=0.01, g1=9.0):
        self.g0 = g0
        self.g1 = g1

    def gs(self, A, RH, Cs):
        """
        Calculate stomatal conductance.

        Parameters
        ----------
        A : float or np.ndarray
            Net assimilation rate (µmol m⁻² s⁻¹)
        RH : float or np.ndarray
            Relative humidity at leaf surface (0–1)
        Cs : float or np.ndarray
            CO2 concentration at leaf surface (µmol mol⁻¹)

        Returns
        -------
        g_s : float or np.ndarray
            Stomatal conductance (mol m⁻² s⁻¹)
        """
        return self.g0 + self.g1 * (A * RH / Cs)

# ------------------------------------------------------------------------------------------------------------------------
# # Example

# A_net = 15.0   # µmol m⁻² s⁻¹
# RH = 0.7       # 70% relative humidity
# Cs = 400       # CO₂ concentration (µmol mol⁻¹)

# bb_model = BallBerry(g0=0.01, g1=9.0)
# gs_leaf = bb_model.gs(A_net, RH, Cs)
# print("Stomatal conductance:", gs_leaf)

# ========================================================================================================================
# Module WaterDensity
# ========================================================================================================================

class WaterDensity:
    """
    Calculate water density (kg/m³) from temperature (°C) and atmospheric pressure (Pa).

    Supports multiple formulations:
    - 'Fisher' (default): Fisher & Dial (1975) — accurate over a wide temperature and pressure range.
    - 'Chen' : Chen et al. (2008) — alternative high-accuracy empirical fit.
    - 'Approx' : Simple polynomial approximation valid for 0–100°C at 1 atm.

    Parameters
    ----------
    Ta : float or np.ndarray
        Water temperature (°C).
    Patm : float or np.ndarray
        Atmospheric pressure (Pa).
    method : str, optional
        Calculation method ('Fisher', 'Chen', or 'Approx'), default = 'Fisher'.

    Raises
    ------
    ValueError
        If temperature is below -30°C (outside empirical model validity) or method is unknown.

    Returns
    -------
    rho : np.ndarray
        Water density (kg/m³).
    """

    def __init__(self, Ta, Patm, method = 'Fisher') -> None:
        if np.nanmin(Ta) < np.array([-30]):
            raise ValueError("Water density calculations below about -30°C are unstable")

        if method == "Fisher":
            self.rho = WaterDensity._calc_water_density_Fisher(Ta, Patm)

        elif method == "Chen":
            self.rho = WaterDensity._calc_water_density_Chen(Ta, Patm)
        
        elif method == "Approx":
            self.rho = WaterDensity._calc_water_density_approx(Ta)

        else:
            raise ValueError("Unknown method provided to calculate water density")

    @staticmethod
    def _evaluate_horner_polynomial(x, cf):
        """Evaluates a polynomial with coefficients `cf` at `x` using Horner's method."""
        y = np.zeros_like(x)
        for c in reversed(cf):
            y = x * y + c
        return y

    @staticmethod
    def _calc_water_density_Chen(Ta, Patm):
        """Calculate the density of water using Chen et al 2008."""

        # Calculate density at 1 atm (kg/m^3):
        # Chen water density
        chen_po = np.array([
            0.99983952, 6.788260e-5, -9.08659e-6, 1.022130e-7, -1.35439e-9,
            1.471150e-11, -1.11663e-13, 5.044070e-16, -1.00659e-18,
        ])
        po_coef = chen_po
        po = WaterDensity._evaluate_horner_polynomial(Ta, po_coef)

        # Calculate bulk modulus at 1 atm (bar):
        chen_ko = np.array([19652.17, 148.1830, -2.29995, 0.01281, -4.91564e-5, 1.035530e-7])
        ko_coef = chen_ko
        ko = WaterDensity._evaluate_horner_polynomial(Ta, ko_coef)

        # Calculate temperature dependent coefficients:
        chen_ca = np.array([3.26138, 5.223e-4, 1.324e-4, -7.655e-7, 8.584e-10])
        ca_coef = chen_ca
        ca = WaterDensity._evaluate_horner_polynomial(Ta, ca_coef)

        chen_cb = np.array([7.2061e-5, -5.8948e-6, 8.69900e-8, -1.0100e-9, 4.3220e-12])
        cb_coef = chen_cb
        cb = WaterDensity._evaluate_horner_polynomial(Ta, cb_coef)

        # Convert atmospheric pressure to bar (1 bar = 100000 Pa)
        pbar = (1.0e-5) * Patm

        pw = ko + ca * pbar + cb * pbar**2.0
        pw /= ko + ca * pbar + cb * pbar**2.0 - pbar
        pw *= (1e3) * po
        return pw

    @staticmethod
    def _calc_water_density_Fisher(Ta, Patm):
        """Calculate water density."""

        # Calculate lambda, (bar cm^3)/g:
        # Fisher Dial
        fisher_dial_lambda = np.array([1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06])

        lambda_coef = fisher_dial_lambda
        lambda_val = WaterDensity._evaluate_horner_polynomial(Ta, lambda_coef)

        # Calculate po, bar
        fisher_dial_Po = np.array([5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05])
        po_coef = fisher_dial_Po
        po_val = WaterDensity._evaluate_horner_polynomial(Ta, po_coef)

        # Calculate vinf, cm^3/g

        fisher_dial_Vinf = np.array([
            0.6980547, -0.0007435626, 3.704258e-05, -6.315724e-07, 9.829576e-09,
            -1.197269e-10, 1.005461e-12, -5.437898e-15, 1.69946e-17, -2.295063e-20
        ])

        vinf_coef = fisher_dial_Vinf
        vinf_val = WaterDensity._evaluate_horner_polynomial(Ta, vinf_coef)

        # Convert pressure to bars (1 bar <- 100000 Pa)
        pbar = 1e-5 * Patm

        # Calculate the specific volume (cm^3 g^-1):
        spec_vol = vinf_val + lambda_val / (po_val + pbar)

        # Convert to density (g cm^-3) -> 1000 g/kg; 1000000 cm^3/m^3 -> kg/m^3:
        rho = 1e3 / spec_vol
        return rho

    @staticmethod
    def _calc_water_density_approx(Ta):
        """
        Calculate water density (kg/m³) at temperature Ta in Celsius
        using empirical formula (valid 0-100°C at ~1 atm).
        """
        rho = 1000 * (1 - ((Ta + 288.9414)*(T - 3.9863)**2)/(508929.2*(Ta + 68.12963)))
        return rho

# ========================================================================================================================
# Module Optimality
# ========================================================================================================================

class Optimality:
    """
    Optimality-based model for the carbon assimilation–transpiration trade-off.

    This class implements the core physiological relationships from the
    **P-model** (Stocker et al. 2020), which assumes that vegetation optimizes
    carbon gain per unit of water loss under given environmental conditions.

    It computes temperature- and pressure-dependent quantities required for
    photosynthesis models, including:
    - Ambient CO₂ partial pressure
    - Photorespiratory CO₂ compensation point (Γ*)
    - Relative water viscosity (η*)
    - Michaelis–Menten coefficient (Kₘₘ)

    References
    ----------
    - Stocker, B. D. et al. (2020). *P-model v1.0: An optimality-based light use efficiency model for simulating ecosystem gross primary production.*
      Geoscientific Model Development, 13(3), 1545–1581.
    - Prentice, I. C. et al. (2014). *Balancing the costs of carbon gain and water transport:
      testing a new theoretical framework for plant functional ecology.*
      Ecology Letters, 17(1), 82–91.
    - Wang, H. et al. (2017). *Towards a universal model for carbon dioxide uptake by plants.*
      Nature Plants, 3(9), 734–741.

    Parameters
    ----------
    env_params : dict
        Environmental parameters, with keys:
        - 'Ta' : float — Air temperature (°C)
        - 'Patm' : float — Atmospheric pressure (hPa)
        - 'VPD' : float — Vapour pressure deficit (hPa)
        - 'CO2' : float — Atmospheric CO₂ concentration (ppm)
        - Optional: 'FAPAR' (-), 'PPFD' (µmol m-2 s-1) (if used in GPP estimation)
    T_ref : float, optional
        Reference temperature for enzyme kinetics (°C). Default is 25°C.

    Attributes
    ----------
    env_params : dict
        Updated environmental parameters with additional computed quantities:
        - 'Ca' : float — Ambient CO₂ partial pressure (Pa)
        - 'gammastar' : float — Photorespiratory compensation point (Pa)
        - 'ns_star' : float — Relative water viscosity (-)
        - 'kmm' : float — Michaelis–Menten constant (Pa) for Rubisco-limited assimilation
    """
    def __init__(self, env_params: dict, T_ref = 25.) -> None:
        self.env_params = env_params
        # Ta, Patm, VPD, CO2
        self.env_params['VPD'] *= 100 # hPa -> Pa
        self.env_params['Patm'] *= 100 # hPa -> Pa
        self.env_params['Ca'] = self._calc_CO2_to_Ca(self.env_params['CO2'], self.env_params['Patm'])
        self.env_params['gammastar'] = self._calc_gammastar(self.env_params['Ta'], self.env_params['Patm'], T_ref = T_ref)
        self.env_params['ns_star'] = self._calc_ns_star(self.env_params['Ta'], self.env_params['Patm'], T_ref = T_ref)
        self.env_params['kmm'] = self._calc_kmm(self.env_params['Ta'], self.env_params['Patm'], T_ref = T_ref)

    @staticmethod
    def _calc_CO2_to_Ca(CO2, Patm):
        '''
        Convert CO2 ppm to Pa
        Args:
            CO2: atmospheric :CO2 concentration, ppm
            Patm (float): atmospheric pressure, Pa

        Returns:
            Ambient :CO2 in units of Pa
        '''
        Ca = 1.0e-6 * CO2 * Patm 
        return Ca

    @staticmethod
    def _calc_gammastar(Ta, Patm, T_ref = 25.):
        # Calculate the photorespiratory CO2 compensation point.
        # Bernacchi estimate of gs25_0
        bernacchi_gs25_0 = 4.332  # Reported as 42.75 µmol mol-1
        # Standard reference atmosphere (Allen, 1973) (:math:`P_o` , 101325.0, Pa)
        Patm_ref = 101325.0
        # Bernacchi estimate of activation energy for gammastar (J/mol)
        bernacchi_dha = 37830
        return (
            bernacchi_gs25_0 * Patm / Patm_ref
            * Optimality._calc_arrhenius((Ta + 273.15), ha = bernacchi_dha, T_ref = T_ref)
        )

    @staticmethod
    def _calc_arrhenius(Ta_K, ha, T_ref = 25.):
        """
        Calculate enzyme kinetics scaling factor.
        Ta_K: air temperature, K
        ha: activation energy for gammastar (J/mol)
        T_ref: reference temperature, 25°C
        """
        # Universal gas constant (8.3145, J/mol/K)
        R = 8.3145

        T_ref_K = T_ref + 273.15

        return np.exp(ha * (Ta_K - T_ref_K) / (T_ref_K * R * Ta_K))

    @staticmethod
    def _calc_ns_star(Ta, Patm, T_ref = 25.):
        """
        Calculate the relative viscosity of water.
        Ta: air temperature, °C
        Patm: atmospheric pressure, Pa
        T_ref: reference temperature, 25°C
        """

        visc_env = Optimality._calc_water_viscosity(Ta, Patm)
        # Standard reference atmosphere (Allen, 1973) (101325.0, Pa)
        Patm_ref = 101325.0
        visc_std = Optimality._calc_water_viscosity(T_ref, np.array(Patm_ref))

        return visc_env / visc_std

    @staticmethod
    def _calc_water_viscosity(Ta, Patm, simple_viscosity = False):
        """Calculate the viscosity of water."""

        if simple_viscosity:
            # The reference for this is unknown, but is used in some implementations
            # so is included here to allow intercomparison.
            return np.exp(-3.719 + 580 / ((Ta + 273) - 138))

        # Get the density of water, kg/m^3
        rho = WaterDensity(Ta, Patm).rho

        # Calculate dimensionless parameters:
        # Huber reference temperature (647.096, Kelvin)
        Huber_T_K = 647.096
        # Huber reference density (:math:`\rho_{ast}`, 322.0, kg/m^3)
        Huber_rho_ref = 322.0
        tbar = (Ta + 273.15) / Huber_T_K
        rbar = rho / Huber_rho_ref

        # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
        # Temperature dependent parameterisation of Hi in Huber.
        Huber_H_i = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
        mu0 = Huber_H_i[0] + Huber_H_i[1] / tbar
        mu0 += Huber_H_i[2] / (tbar * tbar)
        mu0 += Huber_H_i[3] / (tbar * tbar * tbar)
        mu0 = (1e2 * np.sqrt(tbar)) / mu0

        # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
        ctbar = (1.0 / tbar) - 1.0
        mu1 = 0.0

        # Iterate over the rows of the H_ij core_constants matrix
        # Temperature and mass density dependent parameterisation of Hij in Huber.
        Huber_H_ij = np.array([
                    [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
                    [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
                    [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
                    [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
                    [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
                    [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
                ])

        for row_idx in np.arange(Huber_H_ij.shape[1]):
            cf1 = ctbar**row_idx
            cf2 = 0.0
            for col_idx in np.arange(Huber_H_ij.shape[0]):
                cf2 += Huber_H_ij[col_idx, row_idx] * (rbar - 1.0) ** col_idx
            mu1 += cf1 * cf2

        mu1 = np.exp(rbar * mu1)

        # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
        mu_bar = mu0 * mu1

        # Calculate mu (Eq. 1, Huber et al., 2009)
        # Huber reference pressure (:math:`\mu_{ast}` 1.0e-6, Pa s)
        huber_mu_ast = 1e-06
        return mu_bar * huber_mu_ast  # Pa s

    @staticmethod
    def _calc_kmm(Ta, Patm, T_ref = 25):
        """Calculate the Michaelis Menten coefficient of Rubisco-limited assimilation."""

        # conversion to Kelvin
        T_K = Ta + 273.15

        # Bernacchi estimate of kc25
        bernacchi_kc25 = 39.97  # Reported as 404.9 µmol mol-1
        # Bernacchi estimate of activation energy Kc for CO2 (J/mol)
        bernacchi_dhac = 79430
        kc = bernacchi_kc25 * Optimality._calc_arrhenius(
            T_K, ha = bernacchi_dhac, T_ref = T_ref
        )
        # Bernacchi estimate of ko25
        bernacchi_ko25 = 27480  # Reported as 278.4 mmol mol-1
        # Bernacchi estimate of activation energy Ko for O2 (J/mol)
        bernacchi_dhao = 36380
        ko = bernacchi_ko25 * Optimality._calc_arrhenius(
            T_K, ha = bernacchi_dhao, T_ref = T_ref
        )

        # O2 partial pressure
        # O2 partial pressure, Standard Atmosphere (209476.0, ppm)
        PO2_ref = 209476.0
        po = PO2_ref * 1e-6 * Patm

        return kc * (1.0 + po / ko)


# ------------------------------------------------------------------------------------------------------------------------
# # Example

# opt_model = Optimality({
#     'Ta': 11,
#     'Patm': 1000,
#     'VPD': 1.5,
#     'CO2': 410
# })

# print(
#     'Photorespiratory CO2 compensation point (gammastar, Pa): ',
#     opt_model.env_params['gammastar'],
#     '\nRelative viscosity of water (ns_star, -): ',
#     opt_model.env_params['ns_star'],
#     '\nMichaelis Menten coefficient of Rubisco-limited assimilation (kmm, Pa): ',
#     opt_model.env_params['kmm'],
# )