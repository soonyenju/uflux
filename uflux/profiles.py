import numpy as np

# ========================================================================================================================
# Function A: allocate_lai_layers
# ========================================================================================================================

def allocate_lai_layers(LAI_total, n_layers=10, distribution="exponential", k_ext=0.5):
    """
    Allocate total canopy LAI into multiple layers.
    
    Parameters
    ----------
    LAI_total : float
        Total leaf area index (m2 leaf area per m2 ground area).
    n_layers : int, optional
        Number of canopy layers.
    distribution : str, optional
        Type of distribution ('uniform' or 'exponential').
    k_ext : float, optional
        Exponential decay coefficient (only used if distribution='exponential').
    
    Returns
    -------
    lai_layers : ndarray of shape (n_layers,)
        LAI allocated to each canopy layer from top to bottom.
    cumulative_lai : ndarray of shape (n_layers,)
        Cumulative LAI from canopy top to each layer.
    """

    # Vertical positions of layers (0 = top, 1 = bottom)
    z = np.linspace(0, 1, n_layers)

    if distribution.lower() == "uniform":
        lai_layers = np.full(n_layers, LAI_total / n_layers)

    elif distribution.lower() == "exponential":
        # Create an exponential profile: more LAI near the top
        weights = np.exp(-k_ext * z)
        weights /= np.sum(weights)  # normalize weights to sum to 1
        lai_layers = LAI_total * weights

    else:
        raise ValueError("distribution must be 'uniform' or 'exponential'")

    cumulative_lai = np.cumsum(lai_layers)
    return lai_layers, cumulative_lai


# # ============================
# # Example usage
# # ============================
# import matplotlib.pyplot as plt
# LAI_total = 3.0
# n_layers = 10

# # Try both distributions
# for dist in ["uniform", "exponential"]:
#     lai_layers, cumulative = allocate_lai_layers(LAI_total, n_layers, distribution=dist, k_ext=1.0)
#     print(f"\nDistribution: {dist}")
#     print(f"Sum of lai_layers = {np.sum(lai_layers):.4f} (should be {LAI_total})")
#     print("Layer LAI:", np.round(lai_layers, 4))

#     # Plot
#     plt.plot(cumulative, np.arange(n_layers), label=dist.capitalize())

# plt.gca().invert_yaxis()
# plt.xlabel("Cumulative LAI")
# plt.ylabel("Canopy Layer (top to bottom)")
# plt.title("Canopy LAI Allocation")
# plt.legend()
# plt.show()

# ========================================================================================================================
# Function B: allocate_canopy_temperature
# ========================================================================================================================

def allocate_canopy_temperature(T_air_2m, n_layers, canopy_height, delta_T_max=2.0, a=3.0):
    """
    Allocate air temperature across canopy layers using 2m temperature and canopy height.

    Parameters
    ----------
    T_air_2m : float
        Air temperature measured at 2 m above ground (°C).
    n_layers : int, optional
        Number of canopy layers.
    canopy_height : float
        Canopy height (m).
    delta_T_max : float, optional
        Maximum expected temperature increase from canopy top to ground (°C).
    a : float, optional
        Exponential shape factor (higher = stronger stratification).

    Returns
    -------
    T_layers : ndarray
        Air temperature (°C) for each canopy layer, top to bottom.
    z_mid : ndarray
        Height (m) of each layer midpoint.
    """

    # Normalized layer midpoints (0=top, 1=bottom)
    z_norm = np.linspace(0, 1, n_layers)
    z_mid = canopy_height * (1 - z_norm)  # height from ground upward

    # Exponential temperature increase toward ground
    T_layers = T_air_2m + delta_T_max * (1 - np.exp(-a * z_norm))
    
    return T_layers, z_mid


# # ============================
# # Example usage
# # ============================
# import matplotlib.pyplot as plt
# T_air_2m = 22.0       # °C (measured at 2 m)
# canopy_height = 10.0  # m
# n_layers = 10

# T_layers, z_mid = allocate_canopy_temperature(T_air_2m, n_layers, canopy_height, 
#                                                 delta_T_max=2.5, a=3.0)

# print("Layer temperatures (top→bottom):", np.round(T_layers, 2))

# plt.plot(T_layers, z_mid, "o-", label="Within-canopy T profile")
# plt.axvline(T_air_2m, color="gray", linestyle="--", label="2m temperature")
# plt.gca().invert_yaxis()
# plt.xlabel("Air Temperature (°C)")
# plt.ylabel("Height (m)")
# plt.title("Canopy Air Temperature Profile from 2m Measurement")
# plt.legend()
# plt.show()


# # ========================================================================================================================
# # Module A: CanopyTemperatureProfile [DEPRECATED, see profiles.py]
# # ========================================================================================================================

# class CanopyTemperatureProfile:
#     """
#     Compute canopy temperature profile using absorbed radiation and simple energy balance.
#     """
#     def __init__(self, nlayers, LAI, S_top, L_down=400, T_air=298.15, emissivity=0.97):
#         """
#         Parameters
#         ----------
#         nlayers : int
#             Number of canopy layers
#         LAI : float
#             Leaf Area Index
#         S_top : float
#             Incoming solar radiation at canopy top (W/m²)
#         L_down : float
#             Incoming longwave radiation (W/m²)
#         T_air : float
#             Air temperature (K)
#         emissivity : float
#             Leaf emissivity
#         """
#         self.nlayers = nlayers
#         self.LAI = LAI
#         self.S_top = S_top
#         self.L_down = L_down
#         self.T_air = T_air
#         self.emissivity = emissivity
#         self.sigma = 5.67e-8  # Stefan-Boltzmann constant

#         self.layer_LAI = LAI / nlayers
#         self.T_leaf = np.zeros(nlayers)

#     def compute_absorbed_radiation(self):
#         """
#         Simple exponential decay of light through canopy using Beer-Lambert.
#         """
#         k = 0.5  # extinction coefficient
#         self.S_abs = np.zeros(self.nlayers)
#         for i in range(self.nlayers):
#             cum_LAI = i * self.layer_LAI
#             self.S_abs[i] = self.S_top * np.exp(-k * cum_LAI) * (1 - np.exp(-k * self.layer_LAI))
#         # Longwave absorbed: assume all layers receive L_down
#         self.L_abs = np.ones(self.nlayers) * self.L_down

#     def compute_temperature(self):
#         """
#         Compute leaf temperature per layer using energy balance
#         """
#         self.compute_absorbed_radiation()
#         for i in range(self.nlayers):
#             Q_abs = self.S_abs[i] + self.L_abs[i]
#             # assume convection and latent heat flux is proportional to temperature difference with air
#             H = 10 * (self.T_air - 300)  # simple placeholder
#             self.T_leaf[i] = ( (Q_abs + H) / (self.emissivity * self.sigma) )**0.25

#         return self.T_leaf

#     def layer_temperatures_from_profile(self, T_profile, T_air, nlayers, deltaT_top=None):
#         """
#         Estimate individual layer temperatures from canopy profile.

#         Parameters
#         ----------
#         T_profile : float
#             Average canopy temperature (K) or array of layer temperatures
#         T_air : float
#             Air temperature (K)
#         nlayers : int
#             Number of canopy layers
#         deltaT_top : float
#             Temperature difference at top layer relative to air
#             If None, use max(T_profile - T_air)
#         """
#         if isinstance(T_profile, np.ndarray) and len(T_profile) == nlayers:
#             # Already has layer info
#             return T_profile
#         else:
#             # Compute exponential profile from top to bottom
#             if deltaT_top is None:
#                 deltaT_top = T_profile - T_air
#             layer_LAI = np.linspace(0, 1, nlayers)
#             T_layers = T_air + deltaT_top * np.exp(-2 * layer_LAI)  # example decay
#             return T_layers

# # # -------------------------------
# # # Example usage
# # # -------------------------------
# # canopy = CanopyTemperatureProfile(
# #     nlayers=10,
# #     LAI=3,
# #     S_top=600,  # W/m²
# #     L_down=400, # W/m²
# #     T_air=298.15 # K (25°C)
# # )

# # T_profile = canopy.compute_temperature()
# # print("Canopy temperature profile (K):", T_profile)

# # import matplotlib.pyplot as plt
# # plt.plot(T_profile, np.arange(1,11))
# # plt.gca().invert_yaxis()
# # plt.xlabel("Leaf temperature (K)")
# # plt.ylabel("Canopy layer (top to bottom)")
# # plt.title("Canopy Temperature Profile")
# # plt.grid(True)
# # plt.show()
