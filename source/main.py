import source.metropolis as metro
import source.plot as plot
import source.observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [2, 5, 10]
    temperatures = jnp.linspace(0.001, 2, 30)
    n_steps = 70000
    step_size = 0.05

    results = {}
    for D in dimensions:
        results[D] = {
            "T": [],
            "E_mean": [],
            "Cv": [],
            "acceptance": []
        }

        for T in temperatures:
            key = jax.random.PRNGKey(0)

            trajectory, acceptance_rate = metro.run_simulation(
                key, D, T, n_steps, step_size, "random")

            E_mean = obs.mean_energy(trajectory)
            Cv = obs.heat_capacity(trajectory, T)

            results[D]["T"].append(T)
            results[D]["E_mean"].append(E_mean)
            results[D]["Cv"].append(Cv)
            results[D]["acceptance"].append(acceptance_rate)

    plot.plot_obs__D_T(results, dimensions, "E_mean")
    plot.plot_obs__D_T(results, dimensions, "Cv")


if __name__ == "__main__":
    main()
