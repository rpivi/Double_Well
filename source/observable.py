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
    return corr / corr[0]

def integrated_autocorrelation_time(x, c=5.0):
    corr = autocorrelation(x)
    tau = 0.5
    for t in range(1, len(corr)):
        tau += float(corr[t])
        if t >= c * tau:  # finestra self-consistent
            break
    return tau

def append_observables(results, trajectories, D, T, trajectory, acceptance_rate):
    E_mean = mean_energy(trajectory)
    Cv = heat_capacity(trajectory, T)
    tau_x= integrated_autocorrelation_time(trajectory[:,0])
    tau_x2 = integrated_autocorrelation_time(trajectory[:,0]**2)

    results[D]["T"].append(T)
    results[D]["E_mean"].append(E_mean)
    results[D]["Cv"].append(Cv)
    results[D]["acceptance"].append(acceptance_rate)
    results[D]["tau_x"].append(tau_x)
    results[D]["tau_x^2"].append(tau_x2)

    trajectories[D].append(trajectory[:,0])
    
