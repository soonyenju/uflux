import numpy as np

# Constants (from l2sm_constants)
sif_scale = 1.0  # Scaling factor for SIF; can adjust based on units or calibration

class SIFModel:
    """
    Two-stream canopy solar-induced fluorescence (SIF) model
    for vertically inhomogeneous canopies.

    Leaf-level SIF models included:
      - Gu et al. (2019)
      - van der Tol et al. (2014)
      - Li et al. (2022)

    The model calculates:
      1. Leaf-level SIF per layer
      2. Canopy radiative transfer to compute escaping SIF
    """

    def __init__(self, sif_source='gu'):
        """
        Initialize the SIF model.

        Parameters
        ----------
        sif_source : str
            Leaf-level SIF model to use:
            'gu'  -> Gu et al. 2019
            'vdt' -> van der Tol 2014
            'li'  -> Li et al. 2022
        """
        self.sif_source = sif_source.lower()

    @staticmethod
    def layer_RT_diffuse(lai, leaf_r, leaf_t):
        """
        Two-stream radiative transfer for a single canopy layer.
        Calculates layer reflectance (R) and transmittance (T) for diffuse light.

        Parameters
        ----------
        lai : float or array
            Leaf area index of the layer
        leaf_r : float or array
            Leaf reflectance (fraction)
        leaf_t : float or array
            Leaf transmittance (fraction)

        Returns
        -------
        R : float or array
            Layer reflectance
        T : float or array
            Layer transmittance
        """
        # Single scattering albedo and asymmetry factors
        w = leaf_r + leaf_t        # total fraction of light scattered
        d = leaf_r - leaf_t        # difference between reflection and transmission
        G = 0.5                     # leaf projection factor (spherical leaf angle distribution)
        B = 0.5 * (w + d / 3.0) / w # backward scattering factor

        # Gamma coefficients from Meador & Weaver two-stream solution
        g1 = 2 * (1 - (1 - B) * w)
        g2 = 2 * w * B

        tau = lai * G              # optical depth of the layer
        k = np.sqrt(g1**2 - g2**2)
        D = k + g1 + (k - g1) * np.exp(-2 * k * tau)

        # Layer reflectance and transmittance
        R = g2 * (1 - np.exp(-2 * k * tau)) / D
        T = 2 * k * np.exp(-k * tau) / D

        return R, T

    @staticmethod
    def sif_layer_gu(J):
        """
        Compute leaf SIF per layer using Gu et al. (2019) model.

        Parameters
        ----------
        J : array (nsp, nlayers)
            Electron transport rate per layer (µmol/m2/s)

        Returns
        -------
        sif : array (nsp, nlayers)
            Leaf-level SIF for each layer
        """
        psi_PSIImax = 0.83  # Maximum photochemical quantum yield of PSII
        q_L = 0.8           # Fraction of open PSII reaction centers
        k_DF = 19.0         # Ratio of non-radiative to radiative decay

        # Leaf SIF (mol/m2/s)
        return J * sif_scale * (1 - psi_PSIImax) / (q_L * psi_PSIImax * (1 + k_DF))

    @staticmethod
    def get_k_N_vdT14(x, use_li_kn):
        """
        Compute rate coefficient for non-photochemical quenching (NPQ)
        based on van der Tol et al. (2014).

        Parameters
        ----------
        x : float
            NPQ parameter (0 <= x <= 1)
        use_li_kn : bool
            If True, use sustained NPQ from Li et al. (2022)

        Returns
        -------
        k_N : float
            Rate coefficient for NPQ
        """
        alpha = 2.83
        if use_li_kn:
            k_N0 = 2.48
            beta = 0.114
        else:
            k_N0 = 5.01
            beta = 10.0
        nu = ((1 + beta) * x**alpha) / (beta + x**alpha)
        return nu * k_N0

    @staticmethod
    def get_phi_fdashm(x, tleaf, use_li_kd, use_li_kn):
        """
        Compute maximum fluorescence yield per leaf (Phi_F'm)
        based on van der Tol et al. (2014).

        Parameters
        ----------
        x : float
            NPQ parameter
        tleaf : float
            Leaf temperature (K)
        use_li_kd : bool
            Use temperature-dependent k_D from Li et al. (2022)
        use_li_kn : bool
            Use sustained NPQ from Li et al. (2022)

        Returns
        -------
        phi_fdashm : float
            Maximum leaf fluorescence yield
        """
        k_F = 0.05  # First-order rate constant for fluorescence
        # Non-radiative decay rate (temperature-dependent if selected)
        if use_li_kd:
            k_D = max(0.8738, (tleaf - 273.15) * 0.0301 + 0.0773)
        else:
            k_D = 0.95
        k_N = SIFModel.get_k_N_vdT14(x, use_li_kn)
        return k_F / (k_F + k_D + k_N)

    @staticmethod
    def sif_layer_vdT14(J, APAR, x, tleaf, use_li_kd=False, use_li_kn=False):
        """
        Compute leaf SIF per layer using van der Tol 2014 / Li 2022.

        Parameters
        ----------
        J : array (nsp, nlayers)
            Actual electron transport rate per layer
        APAR : array (nsp, nlayers)
            Absorbed PAR per layer
        x : array (nsp, nlayers)
            NPQ parameter (0 <= x <= 1)
        tleaf : array (nsp, nlayers)
            Leaf temperature per layer (K)
        use_li_kd : bool
            Use Li 2022 temperature-dependent k_D
        use_li_kn : bool
            Use Li 2022 sustained NPQ

        Returns
        -------
        sif : array (nsp, nlayers)
            Leaf-level SIF per layer
        """
        nsp, nlayers = J.shape
        sif = np.zeros_like(J)
        for l in range(nlayers):
            # Compute maximum leaf fluorescence for each sample
            phi_fdashm = np.array([SIFModel.get_phi_fdashm(x[i, l], tleaf[i, l], use_li_kd, use_li_kn)
                                   for i in range(nsp)])
            # Eqn 12 in van der Tol et al. 2014
            sif[:, l] = sif_scale * APAR[:, l] * (1 - J[:, l] / APAR[:, l]) * phi_fdashm
        return sif

    @staticmethod
    def sif_canopy(nlayers, lai_canopy, leaf_r, leaf_t, soil_albedo, sif_l):
        """
        Compute escaping SIF from the canopy using two-stream RT.

        Parameters
        ----------
        nlayers : int
            Number of canopy layers
        lai_canopy : array (nsp,)
            Total canopy LAI
        leaf_r : array (nsp,)
            Leaf reflectance
        leaf_t : array (nsp,)
            Leaf transmittance
        soil_albedo : array (nsp,)
            Soil reflectance
        sif_l : array (nsp, nlayers)
            Leaf-level SIF per layer

        Returns
        -------
        sif_c : array (nsp,)
            Canopy SIF escaping to the top
        """
        nsp = len(lai_canopy)
        sif_c = np.zeros(nsp)
        for l in range(nlayers):
            # LAI above and below current layer
            lai_above = ((l + 0.5) * lai_canopy) / nlayers
            lai_below = ((nlayers - l - 0.5) * lai_canopy) / nlayers

            # Compute reflectance and transmittance of upper and lower layers
            R_above, T_above = SIFModel.layer_RT_diffuse(lai_above, leaf_r, leaf_t)
            R_below, T_below = SIFModel.layer_RT_diffuse(lai_below, leaf_r, leaf_t)

            # Include soil albedo in lower layer reflectance
            Z = R_below * soil_albedo
            R_below = R_below + T_below**2 * soil_albedo * (1 + Z / (1 - Z))

            # Multiple scattering factor
            m_scat = 1 + R_above * R_below / (1 - R_above * R_below)

            # SIF emitted upward from this layer
            sif_c += 0.5 * sif_l[:, l] * T_above.ravel() * (1 + R_below.ravel()) * m_scat.ravel()
        return sif_c

    def fluorescence(self, nlayers, lai, temp_c, leaf_r, leaf_t, soil_albedo, J, Jpot=None, APAR=None):
        """
        Compute total canopy SIF.

        Parameters
        ----------
        nlayers : int
            Number of canopy layers
        lai : array (nsp,)
            Leaf area index per sample
        temp_c : array (nsp,)
            Leaf temperature in Celsius
        leaf_r : array (nsp,)
            Leaf reflectance
        leaf_t : array (nsp,)
            Leaf transmittance
        soil_albedo : array (nsp,)
            Soil albedo
        J : array (nsp, nlayers)
            Electron transport rate per layer
        Jpot : array (nsp, nlayers), optional
            Potential electron transport for NPQ computation
        APAR : array (nsp, nlayers), optional
            Absorbed PAR per layer for van der Tol / Li models

        Returns
        -------
        sif_c : array (nsp,)
            Escaping SIF from the canopy mol m⁻² s⁻¹
        """
        if isinstance(lai, float):
            nsp = 1
        else:
            nsp = len(lai)
        # Convert leaf temperature to Kelvin for SIF calculations
        tleaf = temp_c[:, np.newaxis] + 273.15

        # NPQ parameter (x) initialization
        x = np.ones_like(J)
        tiny = 1e-10
        if Jpot is not None:
            mask = Jpot > tiny
            x[mask] = 1 - (J[mask] / Jpot[mask])

        # Initialize layer SIF
        sif_l = np.zeros_like(J)

        # Compute leaf SIF using selected model
        if self.sif_source == 'gu':
            sif_l = self.sif_layer_gu(J)
        elif APAR is not None:
            if self.sif_source == 'li':
                sif_l = self.sif_layer_vdT14(J, APAR, x, tleaf, use_li_kd=False, use_li_kn=True)
            elif self.sif_source == 'vdt':
                sif_l = self.sif_layer_vdT14(J, APAR, x, tleaf, use_li_kd=False, use_li_kn=False)
            else:
                raise ValueError("Unknown SIF source model")
        else:
            # No APAR provided and model not Gu: return zero
            return np.zeros(nsp)

        # Compute escaping SIF from the canopy
        return self.sif_canopy(nlayers, lai, leaf_r, leaf_t, soil_albedo, sif_l)


# # --------------------------
# # Example usage
# # --------------------------
# nsp = 1
# nlayers = 10
# lai_ts = np.array([3, 4, 5, 2, 3])           # Leaf area index per sample
# leaf_r = np.array([0.1]*nsp)              # Leaf reflectance
# leaf_t = np.array([0.05]*nsp)             # Leaf transmittance
# soil_albedo = np.array([0.2]*nsp)         # Soil reflectance
# J = np.full((nsp, nlayers), 120.0)        # Electron transport rate
# Jpot = np.full_like(J, 150.0)             # Potential electron transport
# APAR = np.full_like(J, 200.0)             # Absorbed PAR
# temp_c = np.array([25]*nsp)               # Leaf temperature in Celsius

# # Initialize SIF model
# sif_model = SIFModel(sif_source='gu')
# sif_canopy = sif_model.fluorescence(nlayers, lai_ts, temp_c, leaf_r, leaf_t, soil_albedo, J, Jpot, APAR)

# print("Escaping canopy SIF:", sif_canopy)