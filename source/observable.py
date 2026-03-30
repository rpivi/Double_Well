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

def find_plateau(R_values, threshold=0.05, obs="_"):
    R = list(R_values)

    for k in range(2, len(R)):
        rel1 = abs(R[k]   - R[k-1]) / abs(R[k-1])
        rel2 = abs(R[k-1] - R[k-2]) / abs(R[k-2])

        if rel1 < threshold and rel2 < threshold:
            R_plateau = R[k]
            tau = 0.5 * (R_plateau - 1.0)
            return R_plateau, tau

    # fallback
    print(f"_Warning: no plateau found for {obs}, using last R")
    R_plateau = R[-1]
    tau = 0.5 * (R_plateau - 1.0)
    return R_plateau, tau


def blocking_analysis(data, threshold=0.01, obs="_"):
    x = jnp.array(data, dtype=jnp.float32)

    # varianza completa 
    varX = jnp.var(x, ddof=1)

    R_list = []
    cur = x.copy()
    m = 1

    # blocling decreasing the size by 1 at each iteration, until at least 2 blocks remain
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
    R_plateau, tau = find_plateau(R_list, threshold, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau
   

def autocorrelation_time(data):
    # no use of bloccking analysis, beacuse no plateau was found for tau_x
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    data_centered = data - np.mean(data)
    # FFT-based autocorrelation
    fft = np.fft.rfft(data_centered, n=2*N)
    power = fft * np.conj(fft)
    # the correlation funztion is: chi (k) = <x_i x_{i+k}> - <x_i> <x_{i+k}>
    chi = np.fft.irfft(power)[:N].real / (N - np.arange(N))
    if chi[0] == 0:
        return 0.0
    chi /= chi[0]
    # tau = sum (1 - k/M) chi(k) / chi(0)
    tau = np.sum((1 - np.arange(N) / N) * chi)
    return tau

def append_observables(results,D: int,T,trajectory,acceptance_rate,V: Callable,threshold: float = 0.01, kb: float = 8.617333262145e-5):
    energies = np.array(jax.vmap(lambda x: V(x))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _ = blocking_analysis(energies, threshold, obs="E_mean")

    # blocking for Cv
    E2_mean_err, _ = blocking_analysis(energies2, threshold, obs="Cv")
    Cv = (np.mean(energies2) - E_mean**2) / (kb * T**2)
    # error propagation for Cv = (E2_mean - E_mean^2) / (k_b * T^2)
    Cv_err = (E2_mean_err + 2 * E_mean * E_mean_err) / (kb * T**2)

    #tau_x
    x = np.array(trajectory[:, 0])
    tau_x = autocorrelation_time(x)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)
    results[D]["trajectory_x"].append(trajectory[:, 0])
    
