import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [2, 5, 10]
    temperatures = jnp.linspace(0.5, 3.0, 6)
    n_thermalization = 20000
    n_steps = 80000
    step_size = 0.1

    results = {}
    for D in dimensions:

        results[D] = {
            "T": [],
            "E_mean": [],
            "Cv": [],
            "acceptance": [],
            "tau_x": [],
            "tau_x^2": []
        }
        key = jax.random.PRNGKey(0)
        x, key = metro.generate_config(key, D, "zeros")

        for T in temperatures:
            _, _, key , x = metro.run_simulation(key, T, n_thermalization, step_size, initial_x=x)

            trajectory, acceptance_rate, key , x = metro.run_simulation(
                key, T, n_steps, step_size, initial_x=x)

            obs.append_observables(results, D, T, trajectory, acceptance_rate)
            print(f"D={D}, T={T:.3f}.")

    plot.plot_obs__D_T(results, dimensions, "E_mean")
    plot.plot_obs__D_T(results, dimensions, "Cv")
    plot.plot_obs__D_T(results, dimensions, "tau_x")
    plot.plot_obs__D_T(results, dimensions, "tau_x^2")
    plot.plot_obs__D_T(results, dimensions, "acceptance")


if __name__ == "__main__":
    main()
