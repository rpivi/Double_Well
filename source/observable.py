import jax 
import jax.numpy as jnp
import numpy as np  
from typing import Callable

def make_potential(a: float = 1.0, b: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def V(x: jax.Array) -> jax.Array:
        double_well = (a/b**4) * (x[0] ** 2 - b) ** 2
        harmonic    = 0.5 * jnp.dot(x[1:], x[1:])
        return double_well + harmonic
 
    return V

def find_plateau(R, window=3, tolerance=0.2, obs="_"):
    window = int(window)
    R = np.array(R, dtype=float)

    # we use log because R can grow esponenzialmente, and we want to detect plateaus in a more stable way
    logR = np.log(R)

    for k in range(window, len(R)):

        # 1. Window of previous values in log scale
        segment = logR[k-window:k]

        # 2. local mean in log scale
        mean = np.mean(segment)

        # 3. STD of log values in the window
        std = np.std(segment)

        # 4. Check of plateau:
        #    Last values should be close to the local mean, within tolerance * mean (in log scale)
        if abs(logR[k] - mean) < tolerance * abs(mean) and std < tolerance * abs(mean):
            # we want the linear value of R, not the log
            R_plateau = R[k]
            tau = 0.5 * (R_plateau - 1)
            return R_plateau, tau

    # --- Fall-back: plateau not found ---
    print(f"WARNING: plateau not found for {obs}, using last value.")
    R_plateau = R[-1]
    tau = 0.5 * (R_plateau - 1)
    return R_plateau, tau

def blocking_analysis(data, window, threshold, obs="_"):
    x = jnp.array(data, dtype=jnp.float32)

    # varianza completa 
    varX = jnp.var(x, ddof=1)

    R_list = []
    cur = x.copy()
    m = 1

    # blocling diminuendo di 1 la dimensione ad ogni iterazione, fino a quando non rimangono almeno 2 blocchi
    while cur.shape[0] >= 2:

        if cur.shape[0] % 2 == 1:
            cur = cur[:-1]

        var_block_mean = jnp.var(cur, ddof=1)

        # R(m) = m * Var(block) / Var(full)
        R = m * var_block_mean / varX
        R_list.append(R)

        # blocking → media di coppie
        cur = 0.5 * (cur[0::2] + cur[1::2])
        m *= 2
    R_plateau, tau = find_plateau(R_list, threshold, window, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau

def append_observables(results,D: int,T,trajectory,acceptance_rate,V: Callable,tolerance: float = 0.01, window: int = 5, kb: float = 8.617333262145e-5):
    energies = np.array(jax.vmap(lambda x: V(x))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _ = blocking_analysis(energies, tolerance, window, obs="E_mean")

    # blocking for Cv
    E2_mean_err, _ = blocking_analysis(energies2, tolerance, window, obs="Cv",)
    Cv = (np.mean(energies2) - E_mean**2) / (kb * T**2)
    # error propagation for Cv = (E2_mean - E_mean^2) / (k_b * T^2)
    Cv_err = (E2_mean_err + 2 * E_mean * E_mean_err) / (kb * T**2)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["trajectory_x"].append(trajectory[:, 0])

    
