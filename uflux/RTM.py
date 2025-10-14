import warnings
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.stats import poisson
from copy import deepcopy

# ========================================================================================================================
# Module A: PROSPECT-5D (Leaf Optics Module)
# ========================================================================================================================

"""
PROSPECT 5D or PROSPECT-PRO model.

Feret et al. - PROSPECT-D: Towards modeling leaf optical properties
    through a complete lifecycle

Féret et al. (2021) - PROSPECT-PRO for estimating content of nitrogen-containing leaf proteins
and other carbon-based constituents
"""

class PROSPECT_5D:
    """
    PROSPECT_5D model. PROT and CBC are for PROSPECT-PRO (Féret et al., 2021)

    Parameters
    ----------
    leafbio : LeafBiology
        Object holding user specified leaf biology model parameters.
    optical_params : dict
        Optical parameter constants. Loaded externally and passed in.

    Returns
    -------
    leafopt : LeafOptics
        Contains attributes relf, tran, kChlrel for reflectance, transmittance
        and contribution of chlorophyll over the 400 nm to 2400 nm spectrum
    """
    def __init__(self, leafbio: dict, optical_params: dict):
        self.leafbio = {
            "Cab":         0.0,  # Chlorophyll concentration, (micro g/cm^2)
            "Cdm":         0.0,  # Leaf mass per unit area, (g/cm^2)
            "Cw":          0.0,  # Equivalent water thickness, (cm)
            "Cs":          0.0,  # Brown pigments (unitless)
            "Cca":         0.0,  # Carotenoid concentration, (micro g/cm^2)
            "Cant":        0.0,  # Anthocyanin content, (micro g/cm^2)
            "N":           0.0,  # Leaf structure parameter (unitless)
            "PROT":        0.0,  # leaf protein content, (g/cm^2). Default: 0.0 [0 - 0.003]  (Féret et al., 2021)
            "CBC":         0.0,  # non-protein carbon-based constituent content, (g/cm^2). Default: 0.0 [0 - 0.01] (Féret et al., 2021)
            "rho_thermal": 0.01, # Reflectance in the thermal range, assumption: 0.01
            "tau_thermal": 0.01  # Transmittance in the thermal range, assumption: 0.01
        }
        self.leafbio.update(leafbio)

        # Run PROSPECT-PRO if PROT and/or CBC are non-zero (Cdm = PROT + CBC).
        if (self.leafbio['PROT'] > 0 or self.leafbio['CBC'] > 0) and self.leafbio['Cdm'] > 0:
            self.leafbio['Cdm'] = 0.0

        self.optical_params = optical_params

        # calc leaf optics, _set_leaf_refl_trans_assumptions makes sure thermal spectral is included
        self.leafopt = self._calc_leafopt
        self.leafoptS = self._set_leaf_refl_trans_assumptions(
                    {
                        'refl': self.leafopt[['refl']],
                        'tran': self.leafopt[['tran']],
                        'kChlrel': self.leafopt[['kChlrel']]
                    }, self.leafbio, self.spectral_info()
        )
        self.leafoptS = pd.DataFrame({
            'refl': self.leafoptS['refl'].flatten(),
            'tran': self.leafoptS['tran'].flatten(),
        }, index = self.spectral_info()['wlS'])

    @property
    def _calc_leafopt(self):
        # Compact leaf layer
        Kall = (
            self.leafbio['Cab'] * self.optical_params['Kab'] +
            self.leafbio['Cca'] * self.optical_params['Kca'] +
            self.leafbio['Cdm'] * self.optical_params['Kdm'] +
            self.leafbio['Cw'] * self.optical_params['Kw'] +
            self.leafbio['Cs'] * self.optical_params['Ks'] +
            self.leafbio['Cant'] * self.optical_params['Kant'] +
            self.leafbio['CBC'] * self.optical_params['cbc'] +
            self.leafbio['PROT'] * self.optical_params['prot']
        ) / self.leafbio['N']

        # Non-conservative scattering (normal case)
        j = np.where(Kall > 0)[0]
        t1 = (1 - Kall) * np.exp(-Kall)

        t2 = Kall ** 2 * np.vectorize(self._expint)(Kall)[0]
        tau = np.ones((len(t1), 1))
        tau[j] = t1[j] + t2[j]
        kChlrel = np.zeros((len(t1), 1))
        kChlrel[j] = self.leafbio['Cab'] * self.optical_params['Kab'][j] / (Kall[j] * self.leafbio['N'])

        t_alph = self._calculate_tav(40, self.optical_params["nr"])
        r_alph = 1 - t_alph
        t12 = self._calculate_tav(90, self.optical_params["nr"])
        r12 = 1 - t12
        t21 = t12 / (self.optical_params["nr"] ** 2)
        r21 = 1 - t21

        # top surface side
        denom = 1 - r21 * r21 * tau ** 2
        Ta = t_alph * tau * t21 / denom
        Ra = r_alph + r21 * tau * Ta

        # bottom surface side
        t = t12 * tau * t21 / denom
        r = r12 + r21 * tau * t

        # Stokes equations to compute properties of next N-1 layers (N real)

        # Normal case
        D = np.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
        rq = r ** 2
        tq = t ** 2
        a = (1 + rq - tq + D) / (2 * r)
        b = (1 - rq + tq + D) / (2 * t)

        bNm1 = b ** (self.leafbio['N'] - 1)
        bN2 = bNm1 ** 2
        a2 = a ** 2
        denom = a2 * bN2 - 1
        Rsub = a * (bN2 - 1) / denom
        Tsub = bNm1 * (a2 - 1) / denom

        # Case of zero absorption
        j = np.where(r + t >= 1)[0]
        Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (self.leafbio['N'] - 1))
        Rsub[j] = 1 - Tsub[j]

        # Reflectance and transmittance of the leaf:
        #   combine top llayer with next N-1 layers
        denom = 1 - Rsub * r
        tran = Ta * Tsub / denom
        refl = Ra + Ta * Rsub * t / denom

        return pd.DataFrame({
            'refl': refl.flatten(),      # Spectral reflectance of the leaf, 400 to 2400 nm
            'tran': tran.flatten(),      # Spectral transmittance of the leaf, 400 to 2400 nm
            'kChlrel': kChlrel.flatten() # Relative portion of chlorophyll contribution to reflecntace / transmittance in the spectral range, 400 to 2400 nm
        }, index = self.optical_params['wl'].flatten())


    @staticmethod
    def _calculate_tav(alpha, nr):
        """
        Calculate average transmissitivity of a dieletrie plane surface.

        Parameters
        ----------
        alpha : float
            Maximum incidence angle defining the solid angle.
        nr : float
            Refractive index

        Returns
        -------
        float
            Transmissivity of a dielectric plane surface averages over all
            directions of incidence and all polarizations.

        NOTE
        ----
        Lifted directly from original run_spart matlab calculations.
        Papers cited in original PROSPECT model:
            Willstatter-Stoll Theory of Leaf Reflectance Evaluated
            by Ray Tracinga - Allen et al.
            Transmission of isotropic radiation across an interface
            between two dielectrics - Stern
        """
        rd = np.pi / 180
        n2 = nr ** 2
        n_p = n2 + 1
        nm = n2 - 1
        a = (nr + 1) * (nr + 1) / 2
        k = -(n2 - 1) * (n2 - 1) / 4
        sa = np.sin(alpha * rd)

        b1 = 0
        if alpha != 90:
            b1 = np.sqrt((sa ** 2 - n_p / 2) * (sa ** 2 - n_p / 2) + k)
        b2 = sa ** 2 - n_p / 2
        b = b1 - b2
        b3 = b ** 3
        a3 = a ** 3
        ts = (k ** 2 / (6 * b3) + k / b - b / 2) - (k ** 2 / (6 * a3) + k / a - a / 2)

        tp1 = -2 * n2 * (b - a) / (n_p ** 2)
        tp2 = -2 * n2 * n_p * np.log(b / a) / (nm ** 2)
        tp3 = n2 * (1 / b - 1 / a) / 2
        tp4 = (
            16
            * n2 ** 2
            * (n2 ** 2 + 1)
            * np.log((2 * n_p * b - nm ** 2) / (2 * n_p * a - nm ** 2))
            / (n_p ** 3 * nm ** 2)
        )
        tp5 = (
            16
            * n2 ** 3
            * (1 / (2 * n_p * b - nm ** 2) - 1 / (2 * n_p * a - nm ** 2))
            / n_p ** 3
        )
        tp = tp1 + tp2 + tp3 + tp4 + tp5
        tav = (ts + tp) / (2 * sa ** 2)

        return tav

    @staticmethod
    def _expint(x):
        return integrate.quad(lambda t: np.exp(-t) / t, x, np.inf)

    @staticmethod
    def spectral_info():
        SpectralBands = {
            'wlP': np.arange(400, 2401, 1),     # Range of wavelengths over which the PROSPECT model operates.
            'wlE': np.arange(400, 751, 1),      # Range of wavelengths in E-F excitation matrix
            'WlF': np.arange(640, 851, 1),      # Range of wavelengths for chlorophyll fluorescence in E-F matrix
            'wlO': np.arange(400, 2401, 1),     # Range of wavelengths in the optical part of the spectrum
            'wlT': np.concatenate(
                [np.arange(2500, 15001, 100), np.arange(16000, 50001, 1000)]
            ),                                  # Range of wavelengths in the thermal part of the spectrum
            'wlPAR': np.arange(400, 701, 1)     # Range of wavelengths for photosynthetically active radiation
        }

        SpectralBands.update({
            'wlS': np.concatenate(
                [SpectralBands['wlO'], SpectralBands['wlT']]
            ),                                  # Range of wavelengths for the solar spectrum. wlO and wlT combined.
            'nwlP': len(SpectralBands['wlP']),  # Number of optical bands
            'nwlT': len(SpectralBands['wlT'])   # Number of thermal bands
        })

        SpectralBands.update({
            'IwlP': np.arange(
                0, SpectralBands['nwlP'], 1
            ),                                   # Index of optical bands
            'IwlT': np.arange(
                SpectralBands['nwlP'], SpectralBands['nwlP'] + SpectralBands['nwlT'], 1
            )                                    # Index of thermal bands
        })

        return SpectralBands

    @staticmethod
    def _set_leaf_refl_trans_assumptions(leafopt, leafbio, spectral):
        """Sets the model assumptions about soil and leaf reflectance and
        transmittance in the thermal range.

        These are that leaf relctance and
        transmittance are 0.01 in the thermal range (this is
        a model assumption that is set in the LeafBiology class in the prospect_5d
        script)

        Returns
        -------
        LeafOptics
        """
        # Leaf reflectance array, spans 400 to 2400 nm in 1 nm increments
        # then 2500 to 15000 in 100 nm increments
        # then 16000 to 50000 in 1000 nm increments
        leafopt = deepcopy(leafopt)
        rho = np.zeros((spectral['nwlP'] + spectral['nwlT'], 1))
        tau = np.zeros((spectral['nwlP'] + spectral['nwlT'], 1))
        rho[spectral['IwlT']] = leafbio['rho_thermal']
        tau[spectral['IwlT']] = leafbio['tau_thermal']
        rho[spectral['IwlP']] = leafopt['refl']
        tau[spectral['IwlP']] = leafopt['tran']
        leafopt['refl'] = rho
        leafopt['tran'] = tau

        return leafopt


# # Leaf model
# cab = 40   # Chlorophyll concentration, micro g / cm ^ 2
# cca = 10   # Carotenoid concentration, micro g / cm ^ 2
# cw = 0.02  # Equivalent water thickness, cm
# cdm = 0.01 # Leaf mass per unit area, g / cm ^ 2
# cs = 0     # Brown pigments (from SPART paper, unitless)
# cant = 10  # Anthocyanin content, micro g / cm ^ 2
# n = 1.5    # Leaf structure parameter. Unitless.
# leafbio = {
#     'Cab': cab, 'Cca': cca, 'Cw': cw, 'Cdm': cdm, 'Cs': cs, 'Cant': cant, 'N': n
# }
# leaf_model = PROSPECT_5D(leafbio, optical_params)
# leaf_model.leafopt['refl'].plot()
# # leaf_model.leafoptS.loc[leaf_model.leafopt['refl'].index, 'refl'].plot()

# ========================================================================================================================
# Module B: BSM (Soil Optics Module)
# ========================================================================================================================

"""
Brightness-Shape-Moisture soil model.

Ported from the original matlat run_spart code.

Model as outlined in:
    The run_spart model: A soil-plant-atmosphere radiative transfer model
    for satellite measurements in the solar spectrum - Yang et al.
"""

class BSM:
    """
    Run the BSM soil model, but you can also get soil reflectance spectrum from
    the JPL soil reflectance data available at https://speclib.jpl.nasa.gov/

    Parameters
    ----------
    soilpar : SoilParameters
        Object with attributes [B, lat, lon] / dry soil spectra, and SMp, SMC,
        film
    optical_params : dict
        Contains keys ['GSV', 'kw', 'nw'] which key the Global Soil Vectors,
        water absorption constants for the spectrum and water refraction index
        for the spectrum. Loaded in in the main run_spart script and passed to this
        function.

    Returns
    -------
    SoilOptics
    """
    def __init__(self, soil_params: dict, optical_params: dict):
        self.soil_params = {
            'B': 0.5,         # Soil brightness.
            'lat': 0,         # Soil spectral coordinate, latitiude, realistic range 80 - 120 deg for soil behavior (see paper, phi)
            'lon': 100,       # Soil spectral coordinate, longitude, realistic range -30 - 30 deg for soil behaviour (see paper, lambda)
            'SMp': 15,        # Soil moisture percentage [%]
            'SMC': None,      # Soil moisture carrying capacity of the soil
            'film': None,     # Single water film optical thickness, cm
            'rdry_set': False # Declares that the object doesnt' contain a dry soil reflectance
        }
        self.soil_params.update(soil_params)

        if isinstance(self.soil_params['SMC'], type(None)):
            warnings.warn("BSM soil model: SMC not supplied," " set to default of 25 %")
            self.soil_params['SMC'] = 25
        if isinstance(self.soil_params['film'], type(None)):
            warnings.warn(
                "BSM soil model: water film optical thickness"
                " not supplied, set to default of 0.0150 cm"
            )
            self.soil_params['film'] = 0.0150
        self.optical_params = optical_params
        self._calculate_tav = PROSPECT_5D._calculate_tav
        self.spectral_info = PROSPECT_5D.spectral_info

        # calc soil optics, _set_soil_refl_trans_assumptions makes sure thermal spectral is included
        self.soilopt = self._calc_soilopt
        self.soiloptS = self._set_soil_refl_trans_assumptions({
            'refl': self.soilopt[['refl']],
        }, self.spectral_info())
        self.soiloptS = pd.DataFrame({
            'refl': self.soiloptS['refl'].flatten(),
        }, index = self.spectral_info()['wlS'])

    @property
    def _calc_soilopt(self):
        # Spectral parameters of the soil
        if not self.soil_params['rdry_set']:
            f1 = self.soil_params['B'] * np.sin(self.soil_params['lat'] * np.pi / 180)
            f2 = self.soil_params['B'] * np.cos(self.soil_params['lat'] * np.pi / 180) * np.sin(self.soil_params['lon'] * np.pi / 180)
            f3 = self.soil_params['B'] * np.cos(self.soil_params['lat'] * np.pi / 180) * np.cos(self.soil_params['lon'] * np.pi / 180)
            # Global Soil Vectors spectra
            self.soil_params['rdry'] = f1 * self.optical_params["GSV"][:, [0]] + f2 * self.optical_params["GSV"][:, [1]] + f3 * self.optical_params["GSV"][:, [2]]

        # kw: water absoprtion specturm; nw: water refraction index spectrum
        return self._soilwat(self.soil_params['rdry'], self.optical_params["nw"], self.optical_params["Kw"], self.soil_params['SMp'], self.soil_params['SMC'], self.soil_params['film'])


    def _soilwat(self, rdry, nw, kw, SMp, SMC, deleff):
        """
        Model soil water effects on soil reflectance and return wet reflectance.

        From original matlab code:
            In this model it is assumed that the water film area is built up
            according to a Poisson process.

        See the description in the original model paper in the top of script
        docstring.

        Parameters
        ----------
        rdry : np.array
            Dry soil reflectance
        nw : np.array
            Refraction index of water
        kw : np.array
            Absorption coefficient of water
        SMp : float
            Soil moisture volume [%]
        SMC : float
            Soil moisture carrying capacity
        deleff : float
            Effective optical thickness of single water film, cm

        Returns
        -------
        np.array
            Wet soil reflectance spectra across 400 nm to 2400 nm

        NOTE
        ----
        The original matlab script accepts SMp row vectors for different SM
        percentages. This is not implemented here but may need to be in future
        if there is a significant speed bonus to doing so.
        """
        k = [0, 1, 2, 3, 4, 5, 6]
        nk = len(k)
        mu = (SMp - 5) / SMC
        if mu <= 0:  # below 5 % SM -> model assumes no effect
            rwet = rdry
        else:
            # From original matlab: Lekner & Dorf (1988)
            #
            # Uses t_av calculation from PROSPECT-5D model. If you want BSM
            # script to operate independently you will need to paste that
            # function in here.
            rbac = 1 - (1 - rdry) * (
                rdry * self._calculate_tav(90, 2 / nw) / self._calculate_tav(90, 2) + 1 - rdry
            )

            # total reflectance at bottom of water film surface
            p = 1 - self._calculate_tav(90, nw) / nw ** 2

            # reflectance of water film top surface, use 40 degrees incidence angle
            # like in PROSPECT (note from original matlab script)
            Rw = 1 - self._calculate_tav(40, nw)

            fmul = poisson.pmf(k, mu)
            tw = np.exp(-2 * kw * deleff * k)
            Rwet_k = Rw + (1 - Rw) * (1 - p) * tw * rbac / (1 - p * tw * rbac)
            rwet = (rdry * fmul[0]) + Rwet_k[:, 1:nk].dot(fmul[1:nk])[:, np.newaxis]

        return pd.DataFrame({
            'refl': rwet.flatten(),      # Soil reflectance spectra (with SM taken into account)
            'refl_dry': rdry.flatten(),  # Dry soil reflectance spectra
        }, index = self.optical_params['wl'].flatten())


    @staticmethod
    def _set_soil_refl_trans_assumptions(soilopt, spectral):
        """Sets the model assumptions about soil and leaf reflectance and
        transmittance in the thermal range.

        These are that soil reflectance is the value for 2400 nm
        in the entire thermal range

        Returns
        -------
        SoilOptics
        """
        soilopt = deepcopy(soilopt)
        rsoil = np.zeros((spectral['nwlP'] + spectral['nwlT'], 1))
        rsoil[spectral['IwlP']] = soilopt['refl']
        rsoil[spectral['IwlT']] = 1 * rsoil[spectral['nwlP'] - 1]
        soilopt['refl'] = rsoil
        return soilopt


# soil_params = {
#     'B': 0.5,         # Soil brightness.
#     'lat': 0,         # Soil spectral coordinate, latitiude, realistic range 80 - 120 deg for soil behavior (see paper, phi)
#     'lon': 100,       # Soil spectral coordinate, longitude, realistic range -30 - 30 deg for soil behaviour (see paper, lambda)
#     'SMp': 15,        # Soil moisture percentage [%]
#     'SMC': None,      # Soil moisture carrying capacity of the soil
#     'film': None,     # Single water film optical thickness, cm
#     'rdry_set': False # Declares that the object doesnt' contain a dry soil reflectance
# }
# soil_model = BSM(soil_params, optical_params)
# soil_model.soilopt['refl'].plot()
# soil_model.soiloptS.loc[soil_model.soilopt['refl'].index, 'refl'].plot()

# ========================================================================================================================
# Module C: SSAILH (Canopy Optics Module)
# ========================================================================================================================

"""
SAILH Canopy model.
    Theory of radiative transfer models applied in optical remote sensing
        - W Verhoef 1998
"""

class SAILH:
    """
    Run the SAILH model.

    Parameters
    ----------
    soil : bsm.SoilOptics
        Contains soil reflectance spectra for 400 nm to 2400 nm
    leafopt : prospect_5d.LeafOptics
        Contains leaf reflectance and transmittance spectra, 400 nm to 2400 nm,
        2500 to 15000 nm, and 16000 to 50000 nm.
    canopy : CanopyStructure
        Contains canopy information and SAIL model assumptions
    angles : Angles
        Holds solar zenith, observer zenith, and relative azimuth angles

    Returns
    -------
    CanopyReflectances
        Contains the four canopy reflectances arrays as attributes rso, rdo,
        rsd, rdd.
    """

    def __init__(self, soilopt, leafopt, canopy, angles: dict = {}):
        if len(leafopt.refl) != 2162:
            raise RuntimeError(
                "Parameter leafopt.refl must be of len 2162"
                " i.e. include thermal specturm. \n This error"
                " usually occurs if you are feeding the prospect_5d"
                " output directly into the SAILH model with adding"
                "\n the neccessary thermal wavelengths."
            )
        self.soilopt = soilopt; self.leafopt = leafopt

        # canopy properties
        self.canopy = {
            'LAI': 3,       # Leaf area index, 0 to 8
            'LIDFa': -0.35, # Leaf inclination distribution function parameter a, range -1 to 1
            'LIDFb': -0.15, # Leaf inclination distribution function parameter b, range -1 to 1
            'q': 0.05,      # Canopy hotspot parameter: leaf width / canopy height, range 0 to 0.2
            'nlayers': 60,  # Number of layers in canopy, 60 (SAIL assumption)
            'nlincl': 13,   # Number of different leaf inclination angles, 13 (SAIL assumption)
            'nlazi': 36,    # Number of different leaf azimuth angles, 36 (SAIL assumption)
        }

        self.canopy['lidf'] = self._calculate_leafangles(
            self.canopy['LIDFa'], self.canopy['LIDFb']
        )                   # Leaf inclination distribution function, calculated from LIDF params

        self.angles = {
            'sol_angle': 40, # Solar zenith angle, degrees, range: 0–75, default: 40
            'obs_angle': 0, # Observer zenith angle, degrees, range: 0–75, default: 0
            'rel_angle': 0  # Relative azimuth angle, degrees, range: 0–180, default 0
        }
        # sza, dza = get_sza(date, lon, lat)

        if angles: self.angles.update(angles)

        self.canopopt = self._calc_canopopt

    @property
    def _calc_canopopt(self):
        deg2rad = np.pi / 180

        nl = self.canopy['nlayers']
        litab = np.array([*range(5, 80, 10), *range(81, 91, 2)])[:, np.newaxis]
        LAI = self.canopy['LAI']
        lidf = self.canopy['lidf']
        xl = np.arange(0, -1 - (1 / nl), -1 / nl)[:, np.newaxis]
        dx = 1 / nl
        iLAI = LAI * dx

        rho = self.leafopt[['refl']].values
        tau = self.leafopt[['tran']].values
        rs = self.soilopt[['refl']].values
        tts = self.angles['sol_angle']
        tto = self.angles['obs_angle']

        # Set geometric quantities

        # ensures symmetry at 90 and 270 deg
        psi = abs(self.angles['rel_angle'] - 360 * round(self.angles['rel_angle'] / 360))
        psi_rad = psi * deg2rad
        sin_tts = np.sin(tts * deg2rad)
        cos_tts = np.cos(tts * deg2rad)
        tan_tts = np.tan(tts * deg2rad)

        sin_tto = np.sin(tto * deg2rad)
        cos_tto = np.cos(tto * deg2rad)
        tan_tto = np.tan(tto * deg2rad)

        sin_ttli = np.sin(litab * deg2rad)
        cos_ttli = np.cos(litab * deg2rad)

        dso = np.sqrt(tan_tts ** 2 + tan_tto ** 2 - 2 * tan_tts * tan_tto * np.cos(psi_rad))

        # geometric factors associated with extinction and scattering
        chi_s, chi_o, frho, ftau = self._volscatt(
            sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli
        )
        # extinction coefficient in direction of sun per
        ksli = chi_s / cos_tts  # leaf angle
        koli = chi_o / cos_tto  # observer angle
        k_ext_sun = ksli.T.dot(self.canopy['lidf'])
        k_ext_obs = koli.T.dot(self.canopy['lidf'])
        # area scattering coefficient fractions
        sobli = frho * np.pi / (cos_tts * cos_tto)
        sofli = ftau * np.pi / (cos_tts * cos_tto)
        bfli = cos_ttli ** 2

        # integration over angles using dot product
        k = ksli.T.dot(lidf)
        K = koli.T.dot(lidf)
        bf = bfli.T.dot(lidf)
        sob = sobli.T.dot(lidf)
        sof = sofli.T.dot(lidf)

        # geometric factors for use with rho and tau
        sdb = 0.5 * (k + bf)  # specular to diffuse backward scattering
        sdf = 0.5 * (k - bf)  # specular to diffuse forward scattering
        ddb = 0.5 * (1 + bf)  # diffuse to diffuse backward scattering
        ddf = 0.5 * (1 - bf)  # diffuse to diffuse forward scattering
        dob = 0.5 * (K + bf)  # diffuse to directional backward scattering
        dof = 0.5 * (K - bf)  # diffuse to directional forward scattering

        # Probabilites
        Ps = np.exp(k * xl * LAI)  # of viewing a leaf in solar direction
        Po = np.exp(K * xl * LAI)  # of viewing a leaf in observation direction

        if LAI > 0:
            Ps[0:nl] = Ps[0:nl] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx)
            Po[0:nl] = Po[0:nl] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx)

        q = self.canopy['q']
        Pso = np.zeros(Po.shape)

        for j in range(len(xl)):
            Pso[j, :] = (
                integrate.quad(self._Psofunction, xl[j] - dx, xl[j], args=(K, k, LAI, q, dso))[0]
                / dx
            )

        # NOTE: there are two lines in the original script here that deal with
        # rounding errors. I have excluded them. If this becomes a problem see
        # lines 115 / 116 in SAILH.m

        # scattering coefficients for
        sigb = ddb * rho + ddf * tau  # diffuse backscatter incidence
        sigf = ddf * rho + ddb * tau  # forward incidence
        sb = sdb * rho + sdf * tau  # specular backscatter incidence
        sf = sdf * rho + sdb * tau  # specular forward incidence
        vb = dob * rho + dof * tau  # directional backscatter diffuse
        vf = dof * rho + dob * tau  # directional forward scatter diffuse
        w = sob * rho + sof * tau  # bidirectional scattering
        a = 1 - sigf  # attenuation
        m = np.sqrt(a ** 2 - sigb ** 2)
        rinf = (a - m) / sigb
        rinf2 = rinf * rinf

        # direct solar radiation
        J1k = self._calcJ1(-1, m, k, LAI)
        J2k = self._calcJ2(0, m, k, LAI)
        J1K = self._calcJ1(-1, m, K, LAI)
        J2K = self._calcJ2(0, m, K, LAI)

        e1 = np.exp(-m * LAI)
        e2 = e1 ** 2
        re = rinf * e1

        denom = 1 - rinf2 ** 2

        s1 = sf + rinf * sb
        s2 = sf * rinf + sb
        v1 = vf + rinf * vb
        v2 = vf * rinf + vb
        Pss = s1 * J1k
        Qss = s2 * J2k
        Poo = v1 * J1K
        Qoo = v2 * J2K

        tau_ss = np.exp(-k * LAI)
        tau_oo = np.exp(-K * LAI)

        Z = (1 - tau_ss * tau_oo) / (K + k)

        tau_dd = (1 - rinf2) * e1 / denom
        rho_dd = rinf * (1 - e2) / denom
        tau_sd = (Pss - re * Qss) / denom
        tau_do = (Poo - re * Qoo) / denom
        rho_sd = (Qss - re * Pss) / denom
        rho_do = (Qoo - re * Poo) / denom

        T1 = v2 * s1 * (Z - J1k * tau_oo) / (K + m) + v1 * s2 * (Z - J1K * tau_ss) / (k + m)
        T2 = -(Qoo * rho_sd + Poo * tau_sd) * rinf
        rho_sod = (T1 + T2) / (1 - rinf2)

        rho_sos = w * np.sum(Pso[0:nl]) * iLAI
        rho_so = rho_sod + rho_sos

        Pso2w = Pso[nl]

        # Sail analytical reflectances
        denom = 1 - rs * rho_dd

        rso = (
            rho_so
            + rs * Pso2w
            + ((tau_sd + tau_ss * rs * rho_dd) * tau_oo + (tau_sd + tau_ss) * tau_do)
            * rs
            / denom
        )
        rdo = rho_do + (tau_oo + tau_do) * rs * tau_dd / denom
        rsd = rho_sd + (tau_ss + tau_sd) * rs * tau_dd / denom
        rdd = rho_dd + tau_dd * rs * tau_dd / denom

        rad = pd.DataFrame({
            'rso': rso.flatten(), # Bidirectional reflectance of the canopy
            'rdo': rdo.flatten(), # Directional reflectance for diffuse incidence of the canopy
            'rsd': rsd.flatten(), # Diffuse reflectance for specular incidence of the canopy
            'rdd': rdd.flatten()  # Diffuse reflectance for diffuse incidence of the canopy
        }, index = self.leafopt.index)
        ext = {
            'k_ext_sun': k_ext_sun.flatten(), # Extinction coefficient in the solar direction
            'k_ext_obs': k_ext_obs.flatten(), # Extinction coefficient in the observer direction
        }
        return {
            'rad': rad,
            'ext': ext
        }


    @staticmethod
    def _Psofunction(x, K, k, LAI, q, dso):
        # From APPENDIX IV of original matlab code
        if dso != 0:
            alpha = (dso / q) * 2 / (k + K)
            pso = np.exp(
                (K + k) * LAI * x
                + np.sqrt(K * k) * LAI / alpha * (1 - np.exp(x * alpha))
            )
        else:
            pso = np.exp((K + k) * LAI * x - np.sqrt(K * k) * LAI * x)

        return pso

    @staticmethod
    def _calcJ1(x, m, k, LAI):
        # For getting numerically stable solutions
        J1 = np.zeros((len(m), 1))
        sing = np.abs((m - k) * LAI) < 1e-6

        CS = np.where(sing)
        CN = np.where(~sing)

        J1[CN] = (np.exp(m[CN] * LAI * x) - np.exp(k * LAI * x)) / (k - m[CN])
        J1[CS] = (
            -0.5
            * (np.exp(m[CS] * LAI * x) + np.exp(k * LAI * x))
            * LAI
            * x
            * (1 - 1 / 12 * (k - m[CS]) ** 2 * LAI ** 2)
        )
        return J1

    @staticmethod
    def _calcJ2(x, m, k, LAI):
        # For getting numerically stable solutions
        J2 = (np.exp(k * LAI * x) - np.exp(-k * LAI) * np.exp(-m * LAI * (1 + x))) / (
            k + m
        )
        return J2

    @staticmethod
    def _volscatt(sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli):
        # Calculate geometric factors. See SAILH.m code.
        # See original matlab code. Adapted here to save recalculating trigs.
        nli = len(cos_ttli)

        psi_rad = psi_rad * np.ones((nli, 1))
        cos_psi = np.cos(psi_rad)

        Cs = cos_ttli * cos_tts
        Ss = sin_ttli * sin_tts

        Co = cos_ttli * cos_tto
        So = sin_ttli * sin_tto

        As = np.maximum(Ss, Cs)
        Ao = np.maximum(So, Co)

        bts = np.arccos(-Cs / As)
        bto = np.arccos(-Co / Ao)

        chi_o = 2 / np.pi * ((bto - np.pi / 2) * Co + np.sin(bto) * So)
        chi_s = 2 / np.pi * ((bts - np.pi / 2) * Cs + np.sin(bts) * Ss)

        delta1 = np.abs(bts - bto)
        delta2 = np.pi - np.abs(bts + bto - np.pi)

        Tot = psi_rad + delta1 + delta2

        bt1 = np.minimum(psi_rad, delta1)
        bt3 = np.maximum(psi_rad, delta2)
        bt2 = Tot - bt1 - bt3

        T1 = 2 * Cs * Co + Ss * So * cos_psi
        T2 = np.sin(bt2) * (2 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

        Jmin = bt2 * T1 - T2
        Jplus = (np.pi - bt2) * T1 + T2

        frho = Jplus / (2 * np.pi ** 2)
        ftau = -Jmin / (2 * np.pi ** 2)

        zeros = np.zeros((nli, 1))
        frho = np.maximum(zeros, frho)
        ftau = np.maximum(zeros, ftau)

        return chi_s, chi_o, frho, ftau

    @staticmethod
    def _calculate_leafangles(LIDFa, LIDFb):
        """Calculate the Leaf Inclination Distribution Function as outlined
        by Verhoef in paper cited at the top of this script.

        Parameters
        ----------
        LIDFa : float
            Leaf inclination distribution function parameter a, range -1 to 1
        LIDFb : float
            Leaf inclination distribution function parameter b, range -1 to 1

        Returns
        -------
        np.array
            Leaf inclination distribution function, calculated from LIDF
        """

        def dcum(a, b, theta):
            # Calculate cumulative distribution
            rd = np.pi / 180
            if LIDFa > 1:
                f = 1 - np.cos(theta * rd)
            else:
                eps = 1e-8
                delx = 1
                x = 2 * rd * theta
                theta2 = x
                while delx > eps:
                    y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
                    dx = 0.5 * (y - x + theta2)
                    x = x + dx
                    delx = abs(dx)
                f = (2 * y + theta2) / np.pi
            return f

        # F sized to 14 entries so diff for actual LIDF becomes 13 entries
        F = np.zeros((14, 1))
        for i in range(1, 9):
            theta = i * 10
            F[i] = dcum(LIDFa, LIDFb, theta)
        for i in range(9, 13):
            theta = 80 + (i - 8) * 2
            F[i] = dcum(LIDFa, LIDFb, theta)
        F[13] = 1

        lidf = np.diff(F, axis=0)

        return lidf


# # -------------------------------------------------
# # Canopy model
# canopy = {
#     'lai': 3,
#     'lidfa': -0.35, # Leaf inclination distribution function parameter a, range -1 to 1
#     'lidfb':  -0.15, # Leaf inclination distribution function parameter b, range -1 to 1
#     'q': 0.05, # Canopy hotspot parameter: leaf width / canopy height, range 0 to 0.2
# }

# canopy_model = SAILH(
#     soil_model.soiloptS,
#     leaf_model.leafoptS,
#     canopy
# )
# canopy_model.canopopt['rad'].loc[soil_model.soilopt['refl'].index, 'rso'].plot()
# # canopy_model.canopopt['ext']['k_ext_sun']


# ========================================================================================================================
# Helpers: Load model parameters
# ========================================================================================================================

from pathlib import Path

def load_optical_params():
    params_file = Path(__file__).parent / "model_parameters/optical_params.csv"
    optical_params = pd.read_csv(f, index_col = 0)
    optical_params = {
        k: optical_params[[k]].values for k in optical_params.columns if not k in ['GSV_dim0', 'GSV_dim1', 'GSV_dim2']
    } | {
        'GSV': optical_params[['GSV_dim0', 'GSV_dim1', 'GSV_dim2']].values
    }
    return optical_params