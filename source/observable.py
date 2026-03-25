import jax 
import jax.numpy as jnp

def blocking_error(data, block_size):
    n_blocks = len(data) // block_size
    data = data[:n_blocks * block_size]

    blocks = data.reshape(n_blocks, block_size)
    means = jnp.mean(blocks, axis=1)

    err = jnp.sqrt(jnp.var(means)/(n_blocks-1))

    return err

@jax.jit
def V(x, a=1.0, b=1.0):
    if x.shape[0] == 1:
        return a * (x[0]**2 - b)**2
    else:
        return a * (x[0]**2 - b)**2 + 0.5 * jnp.sum(x[1:]**2)

def mean_energy(trajectory, block_size):
    energies = jax.vmap(V)(trajectory)
    mean_energy = jnp.mean(energies)
    # error on mean energy using blocking
    E_mean_err = blocking_error(energies, block_size)
    return mean_energy, E_mean_err

def heat_capacity(trajectory, T, block_size):
    energies = jax.vmap(V)(trajectory)
    mean_E = jnp.mean(energies)
    mean_E2 = jnp.mean(energies**2)
    Cv = (mean_E2 - mean_E**2) / (T**2)
    # error on Cv using blocking
    E_mean_err = blocking_error(energies, block_size)
    E_2_mean_err = blocking_error(energies**2, block_size)
    Cv_err = jnp.sqrt((E_2_mean_err**2 + 4*mean_E**2*E_mean_err**2) / (T**4))
    return Cv, Cv_err

def autocorrelation(x):
    x = x - jnp.mean(x)
    n = len(x)

    f = jnp.fft.fft(x, n=2*n)
    corr = jnp.fft.ifft(f * jnp.conj(f)).real[:n]

    corr = corr / jnp.arange(n, 0, -1)
    corr = corr / corr[0]

    return corr

def integrated_autocorrelation_time(x, c=5.0):
    corr = autocorrelation(x)
    tau = 0.5
    for t in range(1, len(corr)):
        tau += corr[t]
    return tau

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate):
    tau_x= integrated_autocorrelation_time(trajectory[:,0])
    block_size = int(jnp.maximum(10, 5*tau_x))
    E_mean, E_mean_err = mean_energy(trajectory, block_size)
    Cv, Cv_err = heat_capacity(trajectory, T, block_size)

    results[D]["T"].append(T)
    
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)

    trajectories[D].append(trajectory[:,0])
    
