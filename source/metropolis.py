import jax 
import jax.numpy as jnp
import functools
import observable as obs

@functools.partial(jax.jit, static_argnums=(1,2))
def generate_config(key, D, type="uniform"):
    key, subkey = jax.random.split(key)
    if type == "uniform":
        x = jax.random.uniform(subkey, shape=(D,))
    elif type == "normal":
        x = jax.random.normal(subkey, shape=(D,))
    elif type == "zeros":
        x = jnp.zeros(D)
    elif type == "ones":
        x = jnp.ones(D)
    else:
        raise ValueError(f"Tipo di configurazione sconosciuto: {type}")
    return x, key

@functools.partial(jax.jit, static_argnums=(1,))
def generate_normal(key,D):
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(D,))
    return x, key
    
@jax.jit
def generate_uniform(key):
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey)
    return x, key

@jax.jit
def metropolis_acceptance(delta_E, T):
    A = jnp.where(delta_E <= 0, 1.0, jnp.exp(-delta_E / T))
    return A

@jax.jit
def metropolis_step(x, key, T, step_size=0.1):
    D = x.shape[0]
    # nuova configurazione proposta
    eta, key = generate_normal(key,D)
    x_proposed = x + step_size * eta
    # calcola la differenza di energia
    delta_E = obs.V(x_proposed) - obs.V(x)
    # accetta o rifiuta la nuova configurazione
    accept_prob = metropolis_acceptance(delta_E, T)
    u, key = generate_uniform(key)
    accept = u < accept_prob
    x_new = jnp.where(accept, x_proposed, x)
    return x_new, key, accept

@functools.partial(jax.jit, static_argnums=(2,))
def run_simulation(key, T, n_steps, step_size=0.1, initial_x=None):
    if initial_x is None:
        raise ValueError("initial_x must be provided")
    
    x = initial_x

    def body(carry, _):
        x, key, acc = carry
        x, key, accepted = metropolis_step(x, key, T, step_size)
        return (x, key, acc + accepted), x

    (x_final, keyfinal, acceptances), trajectory = jax.lax.scan(
        body, (x, key, 0), None, length=n_steps
    )
    return trajectory, acceptances / n_steps, keyfinal, x_final