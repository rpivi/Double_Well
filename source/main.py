import metropolis as metro
import plot as plot
import observable as obs
import jax
import jax.numpy as jnp


def main():
    dimensions = [2, 5, 10]
    temperatures = jnp.linspace(0.5, 3.0, 15)
    n_thermalization = 20000
    n_steps = 80000
    step_size = 0.1

    results = {}
    for D in dimensions:

        results[D] = {
            "T": [],
            "E_mean": [],
            "Cv_D": [],
            "acceptance": [],
            "tau": []
        }
        key = jax.random.PRNGKey(0)
        x, key = metro.generate_config(key, D)

        for T in temperatures:
            _, _, key , x = metro.run_simulation(key, T, n_thermalization, step_size, initial_x=x)

            trajectory, acceptance_rate, key , x = metro.run_simulation(
                key, T, n_steps, step_size, initial_x=x)

            E_mean = obs.mean_energy(trajectory)
            Cv = obs.heat_capacity(trajectory, T)
            Cv_per_dim = Cv / trajectory.shape[1]
            tau = obs.integrated_autocorrelation_time(trajectory[:,0])

            results[D]["T"].append(T)
            results[D]["E_mean"].append(E_mean)
            results[D]["Cv_D"].append(Cv_per_dim)
            results[D]["acceptance"].append(acceptance_rate)
            results[D]["tau"].append(tau)
            print(f"D={D}, T={T:.3f}.")

    plot.plot_obs__D_T(results, dimensions, "E_mean")
    plot.plot_obs__D_T(results, dimensions, "Cv_D")
    plot.plot_obs__D_T(results, dimensions, "tau")


if __name__ == "__main__":
    main()
