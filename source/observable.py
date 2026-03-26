import jax 
import jax.numpy as jnp
import numpy as np  

@jax.jit
def V(x, a=1.0, b=1.0):
    return a * (x[0]**2 - b)**2 + 0.5 * jnp.sum(x[1:]**2)

def find_plateau(R_values, threshold=0.01):

    for k in range(1, len(R_values)):
        rel = jnp.abs(R_values[k] - R_values[k-1]) / jnp.abs(R_values[k-1]) # relative change
        if rel < threshold:
            R_plateau = R_values[k]
            tau = 0.5 * (R_plateau - 1.0)
            return R_plateau, tau

    # fallback → usa l'ultimo livello di blocking
    R_plateau = R_values[-1]
    tau = 0.5 * (R_plateau - 1.0)
    print("Warning: no plateau found, using last R value")
    return R_plateau, tau

def blocking_analysis(data):
    x = jnp.array(data, dtype=jnp.float32)

    # varianza completa 
    varX = jnp.var(x, ddof=1)

    R_list = []
    cur = x.copy()
    m = 1

    # fino a quando ci sono almeno 2 blocchi
    while cur.shape[0] >= 2:

        if cur.shape[0] % 2 == 1:
            cur = cur[:-1]
        # Media per blocco (cur è già il dataset "bloccato")
        var_block_mean = jnp.var(cur, ddof=1)

        # R(m) = m * Var(block) / Var(full)
        R = m * var_block_mean / varX
        R_list.append(R)

        # blocking → media di coppie
        cur = 0.5 * (cur[0::2] + cur[1::2])
        m *= 2
    R_plateau, tau = find_plateau(R_list, threshold=0.01)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate, a=1.0, b=1.0):
    energies = np.array(jax.vmap(lambda x: V(x, a, b))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _ = blocking_analysis(energies)
    

    # blocking for Cv
    E2_mean_err, _ = blocking_analysis(energies2)
    Cv = (np.mean(energies2) - E_mean**2) / T**2
    # error propagation for Cv
    Cv_err = (E2_mean_err + 2 * E_mean * E_mean_err) / T**2

    #tau_x
    x_values = np.array(trajectory[:, 0])
    _, tau_x = blocking_analysis(x_values)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)

    trajectories[D].append(trajectory[:, 0])
    
