import numpy as np
from scipy.optimize import minimize

def assimilate_4dvar(forward_model,
                     observations,
                     param_names,
                     x0=None,
                     B_inv=None,
                     R_inv=None,
                     obs_operator=None,
                     bounds=None,
                     model_args=None):
    """
    Generalized 4D-Var data assimilation framework.

    Parameters
    ----------
    forward_model : callable
        The user-supplied forward model function that predicts system states or outputs.
        It should have the signature:
            forward_model(params: dict, **model_args) -> np.ndarray
        where params is a dictionary of parameter names → values.

    observations : np.ndarray
        Array of observed values (shape: n_obs,).
        Can be NDVI, LAI, GPP, or any observable.

    param_names : list of str
        Names of model parameters to be optimized (must match those used inside forward_model).

    x0 : np.ndarray, optional
        Initial parameter vector (len = len(param_names)).
        If None, must be provided inside model_args or defaults in the model.

    B_inv : np.ndarray, optional
        Inverse of background error covariance matrix (len = len(param_names)).
        Default is identity.

    R_inv : np.ndarray, optional
        Inverse of observation error covariance matrix (len = len(observations)).
        Default is identity.

    obs_operator : callable, optional
        Function mapping model outputs to observable space.
        For example:
            obs_operator(LAI) → NDVI
        If None, the forward_model output is assumed to be directly comparable to observations.

    bounds : list of tuple, optional
        Parameter bounds for optimization, e.g. [(0, 100), (10, 200)].

    model_args : dict, optional
        Additional arguments passed to the forward_model (e.g., meteorology, constants, etc.).

    Returns
    -------
    result : OptimizeResult
        The optimization result from scipy.optimize.minimize.
        Access with result.x, result.fun, etc.
    """

    # Set defaults
    n_params = len(param_names)
    if x0 is None:
        raise ValueError("Initial parameter vector x0 must be provided.")
    if B_inv is None:
        B_inv = np.eye(n_params)
    if R_inv is None:
        R_inv = np.eye(len(observations))
    if model_args is None:
        model_args = {}

    # -------------------------
    # Define cost function J(x)
    # -------------------------
    def cost_function(param_vector):
        # Build parameter dict
        params = {name: val for name, val in zip(param_names, param_vector)}

        # Forward simulation
        model_output = forward_model(params, **model_args)

        # Transform to observation space
        if obs_operator is not None:
            model_obs = obs_operator(model_output)
        else:
            model_obs = model_output

        # Observation mismatch (innovation)
        resid = model_obs - observations

        # Cost terms
        J_obs = 0.5 * resid.T @ R_inv @ resid
        delta = param_vector - x0
        J_b = 0.5 * delta.T @ B_inv @ delta

        return J_obs + J_b

    # -------------------------
    # Run optimization (4D-Var)
    # -------------------------
    res = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')

    # Return result with final parameter mapping
    res.params = {name: val for name, val in zip(param_names, res.x)}
    return res

# # ==============================================================================
# # Running example:
# # Run the 4D-Var assimilation to recover the parameters (cab, Vcmax25, Jmax25) 
# # against synthetic LAI observations with random noise.
# # ==============================================================================

# # ===============================================================
# # Step 1. Define a simple dummy forward model
# # ===============================================================

# def simple_forward_model(params, met):
#     """
#     Simple 'ecosystem' model:
#     Predicts LAI from climate (Tair, SWdown) and parameters.
    
#     LAI = a * log(SWdown + 1) + b * exp(0.05*Tair) + c
#     where a,b,c are derived from (cab, v_cmax25, j_max25)
#     """
#     cab = params['cab']
#     v_cmax25 = params['v_cmax25']
#     j_max25 = params['j_max25']

#     # Model equation
#     LAI = (0.002 * cab) * np.log(met['SWdown'] + 1) \
#         + (0.0005 * v_cmax25) * np.exp(0.05 * met['Tair']) \
#         + 0.001 * (j_max25 / 2)
#     return LAI


# # ===============================================================
# # Step 2. Create synthetic (dummy) data
# # ===============================================================
# np.random.seed(42)
# n_days = 30
# Tair = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, n_days))
# SWdown = 200 + 300 * np.random.rand(n_days)

# true_params = {'cab': 50, 'v_cmax25': 90, 'j_max25': 180}
# true_LAI = simple_forward_model(true_params, {'Tair': Tair, 'SWdown': SWdown})
# LAI_obs = true_LAI + np.random.normal(0, 0.2, size=n_days)  # add Gaussian noise

# # ===============================================================
# # Step 3. Run 4D-Var assimilation
# # ===============================================================
# param_names = ['cab', 'v_cmax25', 'j_max25']
# x0 = np.array([30, 60, 100])  # initial guess
# B_inv = np.eye(3) * 0.01
# R_inv = np.eye(n_days) * 5.0
# bounds = [(0, 80), (20, 150), (50, 300)]

# res = assimilate_4dvar(
#     forward_model=simple_forward_model,
#     observations=LAI_obs,
#     param_names=param_names,
#     x0=x0,
#     B_inv=B_inv,
#     R_inv=R_inv,
#     obs_operator=None,  # forward model already predicts LAI
#     bounds=bounds,
#     model_args={'met': {'Tair': Tair, 'SWdown': SWdown}}
# )

# # ===============================================================
# # Step 4. Display results
# # ===============================================================
# print("True parameters:")
# for k, v in true_params.items():
#     print(f"  {k:<10} = {v:.2f}")

# print("\nRecovered parameters:")
# for k, v in res.params.items():
#     print(f"  {k:<10} = {v:.2f}")

# # Plot observed vs modeled LAI
# LAI_model = simple_forward_model(res.params, {'Tair': Tair, 'SWdown': SWdown})

# plt.figure(figsize=(8, 4))
# plt.plot(LAI_obs, 'ro', label='Observed LAI')
# plt.plot(true_LAI, 'k--', label='True LAI')
# plt.plot(LAI_model, 'b-', label='Assimilated model LAI')
# plt.xlabel('Day')
# plt.ylabel('LAI')
# plt.legend()
# plt.title('4D-Var Assimilation Against LAI')
# plt.tight_layout()
# plt.show()