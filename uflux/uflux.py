import uflux
from uflux.RTM import BSM, PROSPECT_5D, SAILH
from uflux.ecophysiology import FvCB, BallBerry, WaterDensity, Optimality, PhotosynthesisKinetics, PhotosynLimiters
from uflux.energy_balance import EnergyBalance
from uflux.micrometeorology import calc_aerodynamic_resistance, calc_RH_from_VPD
from uflux.profiles import allocate_canopy_temperature, allocate_lai_layers
from uflux.solar_induced_fluorescence import SIFModel
from uflux.utilities import shortwave_down_to_APAR, calc_GPP_LUE
from uflux.remotesensing import LAI_to_FAPAR

optical_params = uflux.load_optical_params()

lai = 1
Ta = 11
shortwave_down = 1651
longwave_down = 3963
VPD = 3.53
RH = calc_RH_from_VPD(Ta, VPD)
ws = 0.73
co2 = 410
patm = 101383 # Pa
n_layers = 10

# # PROSPECT and SAILH parameters
# scope_params = {"lai":10.0, "v_cmax25":70, "j_max25":140, "n_layers":10,
#                 "cab":40, "cca":10, "cdm":0.01, "cw":0.02, "N_leaf": 1.5, "PROT": 0.001, "CBC": 0.005,
#                 "LIDFa": 0.0, "LIDFb": 0.0, "q": 0.1, "sol_angle": 30.0, "obs_angle": 0.0, "rel_angle": 180.0}


# def par_from_sw(sw_down):
#     """将短波辐射 (SWdown) 转换为 PAR。"""
#     return sw_down * 4.6

# Leaf model
cab = 40 # Chlorophyll concentration, micro g / cm ^ 2
cca = 10 # Carotenoid concentration, micro g / cm ^ 2
cw = 0.02 # Equivalent water thickness, cm
cdm = 0.01 # Leaf mass per unit area, g / cm ^ 2
cs = 0 # Brown pigments (from SPART paper, unitless)
cant = 10 # Anthocyanin content, micro g / cm ^ 2
n = 1.5 # Leaf structure parameter. Unitless.
# Leaf model
cab = 40   # Chlorophyll concentration, micro g / cm ^ 2
cca = 10   # Carotenoid concentration, micro g / cm ^ 2
cw = 0.02  # Equivalent water thickness, cm
cdm = 0.01 # Leaf mass per unit area, g / cm ^ 2
cs = 0     # Brown pigments (from SPART paper, unitless)
cant = 10  # Anthocyanin content, micro g / cm ^ 2
n = 1.5    # Leaf structure parameter. Unitless.
leafbio = {
    'Cab': cab, 'Cca': cca, 'Cw': cw, 'Cdm': cdm, 'Cs': cs, 'Cant': cant, 'N': n
}
leaf_model = PROSPECT_5D(leafbio, optical_params)
leaf_absorption = (1 - leaf_model.leafopt['refl'] - leaf_model.leafopt['tran']).values

soil_params = {
    'B': 0.5,         # Soil brightness.
    'lat': 0,         # Soil spectral coordinate, latitiude, realistic range 80 - 120 deg for soil behavior (see paper, phi)
    'lon': 100,       # Soil spectral coordinate, longitude, realistic range -30 - 30 deg for soil behaviour (see paper, lambda)
    'SMp': 15,        # Soil moisture percentage [%]
    'SMC': None,      # Soil moisture carrying capacity of the soil
    'film': None,     # Single water film optical thickness, cm
    'rdry_set': False # Declares that the object doesnt' contain a dry soil reflectance
}
soil_model = BSM(soil_params, optical_params)

# Canopy model
canopy = {
    'lai': lai,
    'lidfa': -0.35, # Leaf inclination distribution function parameter a, range -1 to 1
    'lidfb':  -0.15, # Leaf inclination distribution function parameter b, range -1 to 1
    'q': 0.05, # Canopy hotspot parameter: leaf width / canopy height, range 0 to 0.2
}

canopy_model = SAILH(
    soil_model.soiloptS,
    leaf_model.leafoptS,
    canopy
)

opt_model = Optimality({
    'Ta': Ta,
    'Patm': patm / 100,
    'VPD': VPD,
    'CO2': co2
})

rad = canopy_model.canopopt['rad'].mean().to_dict()

# PAR = par_from_sw(shortwave_down)
fAPAR = LAI_to_FAPAR(lai, k=None, lad=None, sza_deg=None, clumping_index=1.0)
PAR, PAR_mol = shortwave_down_to_APAR(shortwave_down, fAPAR)

lai_layers, cumulative = allocate_lai_layers(lai, n_layers, distribution='exponential', k_ext=canopy_model.canopopt['ext']['k_ext_sun'])

APAR_layers = np.array([PAR * (1 - np.exp(-canopy_model.canopopt['ext']['k_ext_sun'] * lai_layer)) for lai_layer in lai_layers])

APAR_leafs = APAR_layers * leaf_absorption.flatten()

T_layers, z_mid = allocate_canopy_temperature(Ta, n_layers, 5, delta_T_max=2.0, a=3.0)

Vcmax, Jmax = PhotosynthesisKinetics.calc_Vcmax_Jmax(Ta, Vcmax25 = 60.0, Jmax25 = 100.0)
J = PhotosynthesisKinetics.calc_J(APAR_layers.sum(), Jmax)
Vpmax, Kp, gbs = PhotosynthesisKinetics.calc_C4_kinetics(Ta)


# from top to bottom

GPP_canopy = 0
H_canopy = 0
LE_canopy = 0
SIF_canopy = 0
for l in range(n_layers):
    T_leaf = T_layers[l]
    lai_layer = lai_layers[l]
    APAR_layer = APAR_layers[l]

    # Ci in opt_model is Pa but FvCB accepts umol mol-1, conversion: Ci_umolmol = (Ci_pa / Patm) * 1e6 # Patm in Pa
    # Kc [i.e., kmm] in opt_model is Pa but FvCB accepts µmol mol⁻¹, conversion: Kc_umolmol = (Kc_Pa / Patm) * 1e6 # Patm in Pa
    A_net = FvCB(
        mode='C3', Vcmax=Vcmax, Jmax=Jmax, Rd=1,
        Kc=(opt_model.env_params['kmm'] / 101325) * 1e6,
        Ko=248, O=210, Vpmax = Vpmax, Kp = Kp, gbs = gbs,
        Gamma_star=opt_model.env_params['gammastar']
    ).A_net(Ci=(opt_model.env_params['Ci'] / 101325) * 1e6, PAR=APAR_layer)
    gs = BallBerry(g0=0.01, g1=9.0).gs(A_net, RH, co2)
    ra = calc_aerodynamic_resistance(u=ws, z=2.0, method='log')
    # ra = calc_aerodynamic_resistance(u=ws, method='empirical')
    eb = EnergyBalance(T_leaf, Ta, VPD, shortwave_down, longwave_down, rad, gs, ra)
    rs = eb.stomatal_conductance_to_resistance(gs, T_leaf)
    T_leaf_final = eb._solve_equilibrium(max(T_leaf, Ta+0.1))

    GPP_layer = A_net * lai_layer
    LE_layer = eb.latent_heat_flux(T_leaf_final, VPD, ra, rs)['LE']
    H_layer = eb.sensible_heat_flux(T_leaf_final, Ta, ra)['H']
     
    # --------------------------------------------------------------------------
    nsp = 1
    Jpot = np.full((nsp, n_layers), 150.0)

    # # Initialize SIF model
    sif_model = SIFModel(sif_source='gu')
    sif_layer = sif_model.fluorescence(
        n_layers, np.array([lai_layer]), np.array([T_leaf_final]),
        np.array([leaf_model.leafopt['refl'].mean()]),
        np.array([leaf_model.leafopt['tran'].mean()]),
        np.array([soil_model.soilopt['refl'].mean()]), np.full_like(Jpot, J), Jpot, np.array([APAR_layer])
    )
    # --------------------------------------------------------------------------

    GPP_canopy += A_net * lai_layer
    H_canopy += H_layer * lai_layer
    LE_canopy += LE_layer * lai_layer

    SIF_canopy += sif_layer * lai_layer
    # SIF_canopy += sif_layer * np.exp(-canopy_model.canopopt['ext']['k_ext_sun'] * lai_layer * (n_layers-l-1))


GPP_canopy, H_canopy, LE_canopy, SIF_canopy