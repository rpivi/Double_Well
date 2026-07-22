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

def find_plateau(R, window=3, tollerance=0.2, obs="_"): #there is a abs tol to avoid problems with very small values of R
    window = int(window)
    R = np.array(R, dtype=float)
    logR = np.log(R)

    for k in range(window, len(R)):
        segment = logR[k-window:k]
        mean = np.mean(segment)
        std = np.std(segment)
        thresh = max(0.001, tollerance * abs(mean)) 

        if abs(logR[k] - mean) < thresh and std < thresh:
            R_plateau = R[k]
            tau = 0.5 * (R_plateau - 1)
            return R_plateau, tau

    print(f"\n WARNING: plateau not found for {obs}, using last value.")
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

    # blocling diminuendo di 1 la dimensione ad ogni iterazione, fino a quando non rimangono almeno 16 blocchi
    while len(cur) >= 16:

        if cur.shape[0] % 2 == 1:
            cur = cur[:-1]

        var_block_mean = jnp.var(cur, ddof=1)

        # R(m) = m * Var(block) / Var(full)
        R = m * var_block_mean / varX
        R_list.append(R)

        # blocking → media di coppie
        cur = 0.5 * (cur[0::2] + cur[1::2])
        m *= 2
    R_plateau, tau = find_plateau(R_list,window, threshold, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau

def autocorr_fft(x):
    x = x - np.mean(x)
    N = len(x)
    f = jnp.fft.fft(x, n=2*N)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[:N].real
    acf /= acf[0]
    return acf

#  tau with Sokal method
def tau_int(x, c=5):
    acf = autocorr_fft(x)
    N = len(acf)

    cum = jnp.cumsum(acf[1:])          # cum[i] = sum(acf[1:i+2])
    tau_t = 0.5 + cum                  # tau(t) per t = 1..N-1, tau_t[i] <-> t=i+1
    t_vals = jnp.arange(1, N)

    condition = t_vals > c * tau_t
    found = jnp.any(condition)
    idx = jnp.argmax(condition)        # primo True (0 se non trovato mai)

    M = jnp.where(found, t_vals[idx], N - 1)
    tau = jnp.where(found, tau_t[idx], tau_t[-1])

    if not bool(found):
        print("\n WARNING: windowing non convergente per tau_int, uso M=N-1.")

    delta_tau = tau * jnp.sqrt(2 * (2*M + 1) / len(x))
    return float(tau), float(delta_tau)

def append_observables(results,D: int,T,trajectory,acceptance_rate,V: Callable,tolerance: float = 0.01, window: int = 5,c: int = 5, kb: float = 8.617333262145e-5):
    energies = np.array(jax.vmap(lambda x: V(x))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _= blocking_analysis(energies, window, tolerance, obs="E_mean")

    # blocking for Cv
    E2_mean_err, _ = blocking_analysis(energies2, window, tolerance, obs="Cv",)
    Cv = (np.mean(energies2) - E_mean**2) / (kb * T**2)
    # error propagation for Cv = (E2_mean - E_mean^2) / (k_b * T^2)
    Cv_err = (E2_mean_err + 2 * E_mean * E_mean_err) / (kb * T**2)

    # tau for x[0]
    tau_x, delta_tau = tau_int(trajectory[:, 0],c)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)
    results[D]["delta_tau"].append(delta_tau)

    
