import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp
from tqdm import tqdm


def main():
    dimensions   = [1, 2, 3, 4]
    # temperatures (K) from 5 to 1015 (one T every 10K) K is Kelvin
    temperatures = jnp.arange(5, 1015, 10)
    # k_b in eV/K
    kb = 8.617333262145e-5
    #MCMC parameters
    n_thermalization = 1000
    n_steps = 300000
    step_size = 0.1
    # potential parameters
    a = 0.01
    b = 1.0
    V = obs.make_potential(a, b)
    #blocking parameters
    tollerance = 0.10 # 10% of the mean value of the observable, in log scale
    window = 4 # number of consecutive values to consider for the plateau detection, in log scale
    c = 5 # Sokal method parameter for tau estimation, tau_int(x, c)

    #thermalization example for 3 configurations
    D_demo, T_demo = dimensions[-1], temperatures[20]
    key = jax.random.PRNGKey(0)
 
    # factory per generatori e simulazione (compilati una volta per D_demo)
    run_therm_demo = metro.make_simulation(D_demo, n_thermalization, V, kb)
 
    x1, key = metro.make_config_generator(D_demo, "ones")(key)
    trajectory1, _, _, _ = run_therm_demo(key, T_demo, step_size, x1)
 
    x3, key = metro.make_config_generator(D_demo, "normal")(key)
    trajectoryR, _, _, _ = run_therm_demo(key, T_demo, step_size, x3)
 
    plot.plot_thermalization_energies(
        trajectory1, trajectoryR,
        T_demo, D_demo, n_thermalization, V )

    #simulations for different dimensions and temperatures
    results = {}
    for D in dimensions:
        results[D] = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "Cv": [],
            "Cv_err": [],
            "acceptance": [],
            "trajectory_x": [],
            "tau_x": []
        }

        gen_config = metro.make_config_generator(D, "normal")
        run_first_therm = metro.make_simulation(D, n_thermalization*10, V, kb)
        run_therm  = metro.make_simulation(D, n_thermalization, V, kb)
        run_prod   = metro.make_simulation(D, n_steps, V, kb)

        x, key = gen_config(key)
        for T in tqdm(temperatures, desc=f"D={D}", unit="T"):
            # thermalization 
            if T == temperatures[0]:
                _, _, key , x = run_first_therm(key, T, step_size, x) # thermalization for the first temperature, starting from a fixed configuration (normal distribution)
            _, _, key , x = run_therm(key, T, step_size, x) # thermalization for the other temperatures, starting from the last configuration of the previous thermalization

            # x is the last configuration of the thermalization, used as initial configuration for the production
            # production
            trajectory, acceptance_rate, key , x = run_prod(key, T, step_size, x)
            # x is the last configuration of the trajectory, used as initial configuration for the next temperature

            # append observables to results
            obs.append_observables(results, D, T, trajectory, acceptance_rate, V, tollerance,window, c, kb)
    
    # plot results
    plot.plot_obs_D_T(results, dimensions, "E_mean", error=True)
    plot.plot_obs_D_T(results, dimensions, "Cv", error=True)
    plot.plot_obs_D_T(results, dimensions, "acceptance")
    Ds = [1, 4]
    plot.plot_tau(results, Ds)

    #plotting Ts
    Ts =[temperatures[0], temperatures[20], temperatures[80], temperatures[-1]]
    
    plot.plot_trajectory_x_relative_freq(results, dimensions, Ts)

    #plotting Ts
    Ts =[temperatures[0], temperatures[20], temperatures[-1]]
    plot.plot_trajectory_x_t(results, Ds, Ts)
    
if __name__ == "__main__":
    main()
