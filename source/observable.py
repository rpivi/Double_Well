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
