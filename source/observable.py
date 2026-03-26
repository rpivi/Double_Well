import jax 
import jax.numpy as jnp
import numpy as np  

@jax.jit
def V(x, a=1.0, b=1.0):
    return a * (x[0]**2 - b)**2 + 0.5 * jnp.sum(x[1:]**2)

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
    print(f"Warning: no plateau found for {obs}, using last R")
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
    R_plateau, tau = find_plateau(R_list, threshold, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau

def autocorr(x):
    x = np.asarray(x)
    N = len(x)
    x = x - np.mean(x)
    fft = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:N].real
    acf /= acf[0]
    return acf

def integrated_autocorrelation_time(x, c=6.0):
    """
    Metodo di Madras–Sokal (Gamma-method).
    c = fattore della finestra (tipicamente 4–8)
    """
    ac = autocorr(x)
    tau = 0.5
    for t in range(1, len(ac)):
        if ac[t] < 0:   # autocorrelazione rumorosa → fermiamo
            break
        tau += ac[t]
        # Finestra automatica (criterio di Sokal)
        if t > c * tau:
            break
    return tau

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate, a=1.0, b=1.0, threshold=0.01):
    energies = np.array(jax.vmap(lambda x: V(x, a, b))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _ = blocking_analysis(energies, threshold, obs="E_mean")

    # blocking for Cv
    E2_mean_err, _ = blocking_analysis(energies2, threshold, obs="Cv")
    Cv = (np.mean(energies2) - E_mean**2) / T**2
    # error propagation for Cv
    Cv_err = (E2_mean_err + 2 * E_mean * E_mean_err) / T**2

    #tau_x
    x = np.array(trajectory[:, 0])
    tau_x = integrated_autocorrelation_time(x)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)

    trajectories[D].append(trajectory[:, 0])
    
