import jax 
import jax.numpy as jnp
import functools
import observable as obs

def generate_config(key, D, type="uniform"):
    if type == "uniform":
        return generate_uniform_vec(key, D)
    elif type == "normal":
        return generate_normal_vec(key, D)
    elif type == "zeros":
        return jnp.zeros(D), key
    elif type == "ones":
        return jnp.ones(D), key
    else:
        raise ValueError("Tipo non valido")

@functools.partial(jax.jit, static_argnums=(1,))
def generate_uniform_vec(key, D):
    key, subkey = jax.random.split(key)
    return jax.random.uniform(subkey, shape=(D,)), key

@functools.partial(jax.jit, static_argnums=(1,))
def generate_normal_vec(key, D):
    key, subkey = jax.random.split(key)
    return jax.random.normal(subkey, shape=(D,)), key
    
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
def metropolis_step(x, key, T, step_size=0.1, a=1.0, b=1.0):
    D = x.shape[0]
    # new proposed configuration
    eta, key = generate_normal_vec(key,D)
    x_proposed = x + step_size * eta
    # delta energy between new and old configuration
    delta_E = obs.V(x_proposed,a,b) - obs.V(x,a,b)
    # acceptance of the new configuration
    accept_prob = metropolis_acceptance(delta_E, T)
    u, key = generate_uniform(key)
    accept = u < accept_prob
    x_new = jnp.where(accept, x_proposed, x)
    return x_new, key, accept

@functools.partial(jax.jit, static_argnums=(2,))
def run_simulation(key, T, n_steps, step_size=0.1, initial_x=None, a=1.0, b=1.0):
    if initial_x is None:
        raise ValueError("initial_x must be provided")
    
    x = initial_x

    def body(carry, _):
        x, key, acc = carry
        x, key, accepted = metropolis_step(x, key, T, step_size, a, b)
        return (x, key, acc + accepted), x

    (x, key, acceptances), trajectory = jax.lax.scan(
        body, (x, key, 0), None, length=n_steps
    )
    return trajectory, acceptances / n_steps, key, trajectory[-1]