import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import observable as obs

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def plot_thermalization_energies(trajectory1, trajectory_1, trajectoryR, T, D, n_steps):
    _ensure_report_dir()
    plt.figure(figsize=(8, 6))
    plt.plot(jnp.arange(n_steps), jax.vmap(obs.V)(trajectory1), label="Inizializzazione a 1")
    plt.plot(jnp.arange(n_steps), jax.vmap(obs.V)(trajectory_1), label="Inizializzazione a -1")
    plt.plot(jnp.arange(n_steps), jax.vmap(obs.V)(trajectoryR), label="Inizializzazione random")
    plt.xlabel("Passi di Metropolis")
    plt.ylabel("Energia")
    plt.title("Evoluzione dell'energia media durante la termalizzazione per D={} e T={}".format(D, T))
    plt.legend()
    plt.grid()
    filename = f"thermalization_energies_D{D}_T{T:.2f}.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()

def plot_obs_D_T(results, dimensions, observable, error=None):
    _ensure_report_dir()
    if error is None:
        plt.figure(figsize=(8,6))

        for D in dimensions:
            plt.plot(
                results[D]["T"],
                results[D][observable],
                marker='o',
                label=f"D={D}"
            )

        plt.xlabel("Temperatura")
        plt.ylabel(observable.replace('_', ' ').capitalize())
        plt.title(f"{observable.replace('_', ' ').capitalize()} vs Temperatura per diverse dimensioni")
        plt.legend()
        plt.grid()

        filename = f"{observable}_vs_T.png"
        filepath = os.path.join(REPORT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {filepath}")
        plt.close()
    else: # plot with error bars
        plt.figure(figsize=(8,6))

        for D in dimensions:
            plt.errorbar(
                    results[D]["T"],
                    results[D][observable],
                    yerr=results[D][f"{observable}_err"],
                    marker='o',
                    label=f"D={D}"
                )
        plt.xlabel("Temperatura")
        plt.ylabel(observable.replace('_', ' ').capitalize())
        plt.title(f"{observable.replace('_', ' ').capitalize()} vs Temperatura per diverse dimensioni")
        plt.legend()
        plt.grid()

        filename = f"{observable}_vs_T.png"
        filepath = os.path.join(REPORT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {filepath}")
        plt.close()

def plot_trajectory(trajectories, temperatures, dimensions, bins=30):
    _ensure_report_dir()

    fig, axes = plt.subplots(len(dimensions), len(temperatures), figsize=(15, 10))

    for i, D in enumerate(dimensions):
        for j, T in enumerate(temperatures):
            ax = axes[i, j]

            weights = jnp.ones_like(trajectories[D][j]) / len(trajectories[D][j])

            ax.hist(trajectories[D][j], bins=bins, weights=weights, alpha=0.7)
            ax.set_title(f"D={D}, T={T:.2f}")
            ax.set_xlabel("x₀")
            ax.set_ylabel("Relative frequency")

    plt.tight_layout()
    filepath = os.path.join(REPORT_DIR, "trajectories_frequency.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")

    print(f"Grafico salvato in: {filepath}")
    plt.close(fig)