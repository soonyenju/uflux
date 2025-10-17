import numpy as np

class CarbonDynamics:
    """
    CarbonDynamics: Daily ecosystem carbon balance model based on DALEC (Data Assimilation
    Linked Ecosystem Carbon).

    This model simulates the daily carbon fluxes and pool dynamics among the main
    ecosystem carbon reservoirs: leaf, wood, root, litter, soil organic matter (SOM),
    and labile carbon. It represents carbon inputs from GPP, respiration losses,
    allocation, and turnover processes.

    References
    ----------
    - Williams et al. (2005). "An improved analysis of forest carbon dynamics using data assimilation."
      Biogeosciences, 2, 1–20.
    - Bloom & Williams (2015). "Constraining ecosystem carbon dynamics in a data-limited world."
      Global Change Biology, 21(4), 1424–1436.
    - Smallman et al. (2017). "Assimilation of repeated woody biomass observations constrains
      decadal ecosystem carbon cycle uncertainty in aggrading forests."

    --------------------------------------------------------------------------
    Model Pools (state variables)
    --------------------------------------------------------------------------
    C_leaf   : Leaf carbon pool [g C m⁻²]
    C_wood   : Woody biomass carbon pool [g C m⁻²]
    C_root   : Root carbon pool [g C m⁻²]
    C_litter : Litter carbon pool [g C m⁻²]
    C_soil   : Soil organic matter (SOM) carbon pool [g C m⁻²]
    C_labile : Labile carbon (short-term storage for new growth) [g C m⁻²]

    --------------------------------------------------------------------------
    Model Parameters
    --------------------------------------------------------------------------
    f_a : Fraction of GPP lost as autotrophic respiration (Ra / GPP)
    f_f : Fraction of NPP allocated to foliage
    f_r : Fraction of remaining NPP allocated to roots
    f_l : Fraction of remaining NPP allocated to labile pool
    LMA : Leaf Mass per Area [g C m⁻² leaf area⁻¹]
    tau_x : Carbon residence time in pool x [days]

    --------------------------------------------------------------------------
    Input / Output Units
    --------------------------------------------------------------------------
    GPP, Reco : g C m⁻² d⁻¹
    Carbon fluxes : g C m⁻² d⁻¹
    Carbon pools : g C m⁻²
    LAI : m² m⁻² (dimensionless)

    --------------------------------------------------------------------------
    Example
    --------------------------------------------------------------------------
    carbon_dynamics = CarbonDynamics()
    GPP = 5.0  # g C m⁻² d⁻¹
    out = carbon_dynamics.update(GPP)
    print(out)
    """

    def __init__(self,
                 C_leaf=200.0, C_wood=5000.0, C_root=500.0,
                 C_litter=2000.0, C_soil=10000.0,
                 tau_leaf=365.0, tau_wood=1.0/0.0001,
                 tau_root=1.0/0.002, tau_litter=1.0/0.008,
                 tau_soil=1.0/0.00005,
                 f_a=0.3, f_f=0.4, f_r=0.2, f_l=0.1, LMA=50.0):
        """Initialize all carbon pools and turnover parameters."""
        self.P_RATES = {
            'f_a': f_a, 'f_f': f_f, 'f_r': f_r, 'f_l': f_l,
            't_litter_rate': 1.0 / tau_litter,
            't_som_rate': 1.0 / tau_soil,
            't_wood_rate': 1.0 / tau_wood,
            't_root_rate': 1.0 / tau_root,
            'LMA': LMA
        }

        self.C = {
            'leaf': C_leaf, 'wood': C_wood, 'root': C_root,
            'litter': C_litter, 'soil': C_soil, 'labile': 50.0
        }

        self.tau = {
            'leaf': tau_leaf, 'wood': tau_wood, 'root': tau_root,
            'litter': tau_litter, 'soil': tau_soil, 'labile': 10.0
        }

    # ----------------------------------------------------------------------
    def update(self, GPP, dt=1.0):
        """
        Perform one daily update of carbon pools and fluxes.

        Parameters
        ----------
        GPP : float
            Gross Primary Production [g C m⁻² d⁻¹]
        dt : float
            Time step [days]

        Returns
        -------
        results : dict
            A dictionary containing:
              - 'GPP', 'Reco' : g C m⁻² d⁻¹
              - 'NPP', 'Ra' : g C m⁻² d⁻¹
              - 'P_leaf', 'P_root', 'P_wood', 'P_labile' : g C m⁻² d⁻¹
              - 'T_leaf', 'T_root', 'T_wood', 'T_litter', 'T_som', 'T_labile' : g C m⁻² d⁻¹
              - 'C_leaf', 'C_root', 'C_wood', 'C_litter', 'C_soil', 'C_labile' : g C m⁻²
              - 'LAI' : dimensionless
        """
        # 1. Compute autotrophic respiration and NPP
        Ra = self.P_RATES['f_a'] * GPP       # Autotrophic respiration [g C m⁻² d⁻¹]
        NPP = GPP - Ra                      # Net Primary Production [g C m⁻² d⁻¹]

        # 2. Carbon allocation (fractions of NPP)
        P_leaf = NPP * self.P_RATES['f_f']
        P_root = (NPP - P_leaf) * self.P_RATES['f_r']
        P_labile = (NPP - P_leaf - P_root) * self.P_RATES['f_l']
        P_wood = NPP - P_leaf - P_root - P_labile

        # 3. Daily turnover (decay) rates
        k_labile = 1.0 / self.tau['labile']
        k_leaf = 1.0 / self.tau['leaf']
        k_wood = self.P_RATES['t_wood_rate']
        k_root = self.P_RATES['t_root_rate']
        k_litter = self.P_RATES['t_litter_rate']
        k_som = self.P_RATES['t_som_rate']

        # 4. Compute turnover fluxes (g C m⁻² d⁻¹)
        T_labile = self.C['labile'] * k_labile * dt
        T_leaf = self.C['leaf'] * k_leaf * dt
        T_wood = self.C['wood'] * k_wood * dt
        T_root = self.C['root'] * k_root * dt
        T_litter = self.C['litter'] * k_litter * dt
        T_som = self.C['soil'] * k_som * dt

        # 5. Update pool changes (ΔC)
        dC_labile = P_labile - T_labile
        dC_leaf = P_leaf + T_labile - T_leaf
        dC_root = P_root - T_root
        dC_wood = P_wood - T_wood
        dC_litter = T_leaf + T_root - T_litter
        dC_soil = T_litter + T_wood - T_som

        # 6. Apply updates
        self.C['labile'] += dC_labile
        self.C['leaf'] += dC_leaf
        self.C['root'] += dC_root
        self.C['wood'] += dC_wood
        self.C['litter'] += dC_litter
        self.C['soil'] += dC_soil

        # Ensure no negative pools
        for k in self.C:
            self.C[k] = max(0.0, self.C[k])

        # 7. Compute total ecosystem respiration
        Reco = Ra + T_litter + T_som          # g C m⁻² d⁻¹

        # 8. Compute LAI
        LAI = max(0.1, self.C['leaf'] / self.P_RATES['LMA'])

        # 9. Return all diagnostics
        return {
            # Primary fluxes
            'GPP': GPP,
            'Ra': Ra,
            'NPP': NPP,
            'Reco': Reco,

            # Allocation fluxes
            'P_leaf': P_leaf,
            'P_root': P_root,
            'P_wood': P_wood,
            'P_labile': P_labile,

            # Turnover fluxes
            'T_leaf': T_leaf,
            'T_root': T_root,
            'T_wood': T_wood,
            'T_litter': T_litter,
            'T_som': T_som,
            'T_labile': T_labile,

            # Carbon pools
            'C_leaf': self.C['leaf'],
            'C_root': self.C['root'],
            'C_wood': self.C['wood'],
            'C_litter': self.C['litter'],
            'C_soil': self.C['soil'],
            'C_labile': self.C['labile'],

            # Derived variable
            'LAI': LAI
        }
