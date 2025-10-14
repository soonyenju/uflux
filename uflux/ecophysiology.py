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
