import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [1, 2, 5]
    temperatures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,1.0, 1.5, 2.0]
    n_thermalization = 10000
    n_steps = 200000
    step_size = 0.1
    # potential parameters
    a = 1.0
    b = 1.0

    #thermalization example for 3 configurations
    D = 2
    T = 0.5

    key = jax.random.PRNGKey(0)
    x1, key = metro.generate_config(key, D, "ones")
    trajectory1, _, _, _ = metro.run_simulation(key, T, n_thermalization, step_size, x1, a, b)

    x2, key = metro.generate_config(key, D, "-ones")
    trajectory_1, _, _, _ = metro.run_simulation(key, T, n_thermalization, step_size, x2, a, b)

    x3, key = metro.generate_config(key, D, "normal")
    trajectoryR, _, _, _ = metro.run_simulation(key, T, n_thermalization, step_size, x3, a, b)
    
    plot.plot_thermalization_energies(trajectory1, trajectory_1, trajectoryR, T, D, n_thermalization)

    #simulations for different dimensions and temperatures
    trajectories = {}
    results = {}
    for D in dimensions:

        trajectories[D] = []

        results[D] = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "Cv": [],
            "Cv_err": [],
            "acceptance": [],
            "tau_x": []
        }

        x, key = metro.generate_config(key, D, "ones")
        for T in temperatures:
            
            # thermalization 
            _, _, key , x = metro.run_simulation(key, T, n_thermalization, step_size, initial_x=x, a=a, b=b)
            # x is the last configuration of the thermalization, used as initial configuration for the production
            # production
            trajectory, acceptance_rate, key , x = metro.run_simulation(
                key, T, n_steps, step_size, initial_x=x, a=a, b=b)
            # x is the last configuration of the trajectory, used as initial configuration for the next temperature
            # append observables to results
            obs.append_observables(results,trajectories, D, T, trajectory, acceptance_rate, a, b)
            print(f"D={D}, T={T:.3f}.")

    # plot results
    plot.plot_obs_D_T(results, dimensions, "E_mean", error=True)
    plot.plot_obs_D_T(results, dimensions, "Cv", error=True)
    plot.plot_obs_D_T(results, dimensions, "tau_x")
    plot.plot_obs_D_T(results, dimensions, "acceptance")
    plot.plot_trajectory(trajectories, temperatures, dimensions, bins=30)

if __name__ == "__main__":
    main()
