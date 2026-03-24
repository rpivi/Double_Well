import jax 
import jax.numpy as jnp

@jax.jit
def V(x, a=1.0, b=1.0):
    return a * (x[0]**2 - b)**2 + 0.5 * jnp.sum(x[1:]**2)

def mean_energy(trajectory):
    energies = jax.vmap(V)(trajectory)
    return jnp.mean(energies)

def heat_capacity(trajectory, T):
    energies = jax.vmap(V)(trajectory)
    mean_E = jnp.mean(energies)
    mean_E2 = jnp.mean(energies**2)
    Cv = (mean_E2 - mean_E**2) / (T**2)
    return Cv

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
        if t >= c * tau:  # finestra self-consistent
            break
    return tau

def barrier_crossings_rate(trajectory, eps=1e-3):
    x = trajectory[:,0]

    valid = jnp.abs(x) > eps
    x = x[valid]

    signs = jnp.sign(x)

    return jnp.sum(signs[:-1] * signs[1:] < 0)/len(x)

def blocking_error(data, block_size):
    n_blocks = len(data) // block_size
    data = data[:n_blocks * block_size]

    blocks = data.reshape(n_blocks, block_size)
    means = jnp.mean(blocks, axis=1)

    err = jnp.sqrt(jnp.var(means)/(n_blocks-1))

    return err

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate):
    tau_x= integrated_autocorrelation_time(trajectory[:,0])
    block_size = max(10, 2*int(tau_x))
    E_mean = mean_energy(trajectory)
    E_mean_err = blocking_error(jax.vmap(V)(trajectory), block_size)
    Cv = heat_capacity(trajectory, T)
    Cv_err = blocking_error(jax.vmap(lambda x: (V(x) - E_mean)**2)(trajectory), block_size)
    crossings_r = barrier_crossings_rate(trajectory)
    crossings_r_err = jnp.sqrt(crossings_r)/len(trajectory[:,0])  # standard error assuming Poisson distribution

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["E_mean_err"].append(E_mean_err)
    results[D]["Cv"].append(Cv)
    results[D]["Cv_err"].append(Cv_err)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)
    results[D]["barrier_crossings_rate"].append(crossings_r)
    results[D]["barrier_crossings_rate_err"].append(crossings_r_err)
    
    trajectories[D].append(trajectory[:,0])
    
