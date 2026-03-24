import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [1, 2, 5, 7]
    temperatures = [0.05 ,0.1, 0.5, 1.0, 2.0, 5.0]
    n_thermalization = 200000
    n_steps = 100000
    step_size = 0.1

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
            "tau_x": [],
            "barrier_crossings_rate": [],
            "barrier_crossings_rate_err": []
        }

        key = jax.random.PRNGKey(0)
        x, key = metro.generate_config(key, D, "ones")

        for T in temperatures:
            # thermalization
            _, _, key , x = metro.run_simulation(key, T, n_thermalization, step_size, initial_x=x)
            # production
            trajectory, acceptance_rate, key , x = metro.run_simulation(
                key, T, n_steps, step_size, initial_x=x)
            # append observables to results
            obs.append_observables(results,trajectories, D, T, trajectory, acceptance_rate)
            print(f"D={D}, T={T:.3f}.")
    # plot results
    plot.plot_obs__D_T(results, dimensions, "E_mean", error=True)
    plot.plot_obs__D_T(results, dimensions, "Cv")
    plot.plot_obs__D_T(results, dimensions, "tau_x")
    plot.plot_obs__D_T(results, dimensions, "acceptance")
    plot.plot_obs__D_T(results, dimensions, "barrier_crossings_rate", error=True)
    plot.plot_trajectory(trajectories, temperatures, dimensions, bins=30)

if __name__ == "__main__":
    main()
