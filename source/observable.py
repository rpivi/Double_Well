import jax 
import jax.numpy as jnp
import numpy as np

@jax.jit
def V(x, a=1.0, b=1.0):
    return a * (x[0]**2 - b)**2 + 0.5 * jnp.sum(x[1:]**2)

def blocking_analysis(data):
    data = np.array(data)
    N = len(data)

    var_naive = np.var(data, ddof=1)

    block_sizes = []
    R_values = []
    errors = []

    for B in range(1, N // 5):
        n_blocks = N // B
        if n_blocks < 2:
            break

        trimmed = data[:n_blocks * B]
        block_means = trimmed.reshape(n_blocks, B).mean(axis=1)

        var_B = np.var(block_means, ddof=1)

        R = B * var_B / var_naive

        err = np.sqrt(R * var_naive / N)

        block_sizes.append(B)
        R_values.append(R)
        errors.append(err)

    return np.array(R_values), np.array(errors)

def find_plateau(R_values, errors, window=50):
    R_values = np.array(R_values)
    errors = np.array(errors)

    if len(R_values) <= window:
        R_plateau = R_values[-1]
        err_plateau = errors[-1]
        tau = max((R_plateau - 1) / 2, 0.0)
        return err_plateau, tau

    grad = np.abs(np.diff(R_values))
    threshold = 0.01 * np.percentile(R_values, 95)
    candidates = np.where(grad < threshold)[0]

    # Use the last sustained flat region, not the first transient dip
    idx = candidates[-1] if len(candidates) > 0 else len(R_values) // 2
    idx = max(idx, window)

    end = min(idx + window, len(R_values))
    R_plateau = np.mean(R_values[idx:end])
    err_plateau = np.mean(errors[idx:end])

    tau = max((R_plateau - 1) / 2, 0.0)
    return err_plateau, tau

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate, a=1.0, b=1.0):
    energies = np.array(jax.vmap(lambda x: V(x, a, b))(trajectory))
    energies2 = energies**2

    # blocking su E
    R_E, err_E = blocking_analysis(energies)
    E_mean_err, _ = find_plateau(R_E, err_E)
    E_mean = np.mean(energies)

    # blocking su E^2 per Cv
    R_E2, err_E2 = blocking_analysis(energies2)
    E2_err, _ = find_plateau(R_E2, err_E2)
    Cv = (np.mean(energies2) - E_mean**2) / T**2
    Cv_err = np.sqrt((E2_err**2 + 4*E_mean**2 * E_mean_err**2)) / T**2

    #tau_x
    x_values = np.array(trajectory[:, 0])
    R_x, err_x = blocking_analysis(x_values)
    _, tau_x = find_plateau(R_x, err_x)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)

    trajectories[D].append(trajectory[:, 0])
    
