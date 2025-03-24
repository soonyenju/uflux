import numpy as np
import emcee  # MCMC sampling library

class DALECModel:
    def __init__(self, params):
        """Initialize DALEC with given parameters"""
        self.params = params
        self.reset()  # Reset the state variables to their initial values

    def reset(self):
        """Reset state variables to initial values"""
        p = self.params
        self.leaf = p["leaf_init"]
        self.wood = p["wood_init"]
        self.root = p["root_init"]
        self.soil_fast = p["soil_fast_init"]
        self.soil_slow = p["soil_slow_init"]

    def step(self, gpp):
        """Advance the model by one timestep based on GPP"""
        p = self.params

        # Compute Carbon Fluxes
        Ra = p["Ra_frac"] * gpp
        NPP = gpp - Ra  # Net Primary Productivity
        leaf_growth = p["alloc_leaf"] * NPP
        wood_growth = p["alloc_wood"] * NPP
        root_growth = p["alloc_root"] * NPP

        # Update Carbon Pools
        self.leaf += leaf_growth - (self.leaf * p["leaf_fall"])
        self.wood += wood_growth
        self.root += root_growth - (self.root * p["root_turnover"])

        # Litter and Soil Decomposition
        litter_input = (self.leaf * p["leaf_fall"]) + (self.root * p["root_turnover"])
        Rh_fast = p["decomp_fast"] * self.soil_fast
        Rh_slow = p["decomp_slow"] * self.soil_slow

        # Update Soil Pools
        self.soil_fast += litter_input - Rh_fast - (p["transfer_fast_to_slow"] * self.soil_fast)
        self.soil_slow += (p["transfer_fast_to_slow"] * self.soil_fast) - Rh_slow

        # Compute Net Ecosystem Exchange (NEE)
        Rh = Rh_fast + Rh_slow
        NEE = Ra + Rh - gpp
        return NEE

def log_likelihood(theta, gpp_obs, nee_obs, nee_uncertainty):
    """Computes log-likelihood for MCMC given model parameters"""
    Ra_frac, alloc_leaf, decomp_fast = theta

    # Set up parameters dictionary
    params = {
        "leaf_init": 0.2,
        "wood_init": 5.0,
        "root_init": 0.5,
        "soil_fast_init": 1.0,
        "soil_slow_init": 10.0,
        "Ra_frac": Ra_frac,
        "alloc_leaf": alloc_leaf,
        "alloc_wood": 0.4,
        "alloc_root": 0.3,
        "leaf_fall": 0.1,
        "root_turnover": 0.05,
        "decomp_fast": decomp_fast,
        "decomp_slow": 0.01,
        "transfer_fast_to_slow": 0.02
    }

    # Create DALEC model and run it for each GPP observation
    model = DALECModel(params)
    model_nee = np.array([model.step(gpp) for gpp in gpp_obs])

    # Compute Likelihood (assuming Gaussian errors)
    log_likelihood_val = -0.5 * np.sum(((nee_obs - model_nee) / nee_uncertainty) ** 2)
    return log_likelihood_val

def log_probability(theta, gpp_obs, nee_obs, nee_uncertainty):
    """Calculates the log-probability of the parameters based on MCMC"""
    Ra_frac, alloc_leaf, decomp_fast = theta

    # Apply bounds to parameters to restrict the search space
    if 0.1 < Ra_frac < 0.9 and 0.1 < alloc_leaf < 0.5 and 0.01 < decomp_fast < 0.5:
        return log_likelihood(theta, gpp_obs, nee_obs, nee_uncertainty)
    else:
        return -np.inf  # Return a very low likelihood for invalid parameters

def run_mcmc(gpp_obs, nee_obs, nee_uncertainty, n_walkers=10, n_steps=1000):
    """Run MCMC to estimate parameters"""
    initial_params = [0.5, 0.3, 0.2]  # Initial guesses for [Ra_frac, alloc_leaf, decomp_fast]
    ndim = len(initial_params)

    # Create initial positions for walkers by adding noise to the initial guesses
    initial_positions = [initial_params + 0.01 * np.random.randn(ndim) for _ in range(n_walkers)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(gpp_obs, nee_obs, nee_uncertainty))
    sampler.run_mcmc(initial_positions, n_steps, progress=True)

    return sampler

# Example Usage
if __name__ == "__main__":
    # Example GPP Observations (kgC/m²/day)
    gpp_obs = np.array([4, 5, 6, 5, 4, 3, 2, 3, 4, 5])

    # Generate synthetic NEE with some uncertainty
    true_params = [0.45, 0.28, 0.15]
    true_model = DALECModel({
        "leaf_init": 0.2,
        "wood_init": 5.0,
        "root_init": 0.5,
        "soil_fast_init": 1.0,
        "soil_slow_init": 10.0,
        "Ra_frac": true_params[0],
        "alloc_leaf": true_params[1],
        "alloc_wood": 0.4,
        "alloc_root": 0.3,
        "leaf_fall": 0.1,
        "root_turnover": 0.05,
        "decomp_fast": true_params[2],
        "decomp_slow": 0.01,
        "transfer_fast_to_slow": 0.02
    })
    nee_obs = np.array([true_model.step(gpp) for gpp in gpp_obs]) + np.random.normal(0, 0.1, len(gpp_obs))
    nee_uncertainty = np.full(len(nee_obs), 0.1)  # Assume constant uncertainty of 0.1

    # Run MCMC
    print("\nRunning MCMC Parameter Estimation...\n")
    sampler = run_mcmc(gpp_obs, nee_obs, nee_uncertainty, n_walkers=20, n_steps=2000)

    # Extract the best-fit parameters from the MCMC chain
    samples = sampler.get_chain(discard=500, thin=10, flat=True)
    best_fit_params = np.median(samples, axis=0)

    print("\nEstimated Parameters:")
    print(f"Ra_frac = {best_fit_params[0]:.3f}")
    print(f"alloc_leaf = {best_fit_params[1]:.3f}")
    print(f"decomp_fast = {best_fit_params[2]:.3f}")
