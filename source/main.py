import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [1, 2, 5]
    temperatures = jnp.linspace(0.5, 3.0, 6)
    n_thermalization = 100000
    n_steps = 10000
    step_size = 0.2
    # potential parameters
    a = 1.0
    b = 3.0

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

        key = jax.random.PRNGKey(0)
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
    plot.plot_obs__D_T(results, dimensions, "E_mean", error=True)
    plot.plot_obs__D_T(results, dimensions, "Cv", error=True)
    plot.plot_obs__D_T(results, dimensions, "tau_x")
    plot.plot_obs__D_T(results, dimensions, "acceptance")
    plot.plot_trajectory(trajectories, temperatures, dimensions, bins=30)

if __name__ == "__main__":
    main()
