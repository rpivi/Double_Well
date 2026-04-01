import matplotlib.pyplot as plt
import numpy as np
import os
import jax
import jax.numpy as jnp
from typing import Callable

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")

def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def plot_thermalization_energies(trajectory1, trajectory_1, trajectoryR, T, D, n_steps, V: Callable):
    _ensure_report_dir()
    plt.figure(figsize=(8, 6))
    plt.plot(jnp.arange(n_steps), jax.vmap(V)(trajectory1), label="Inizializzazione a 1")
    plt.plot(jnp.arange(n_steps), jax.vmap(V)(trajectory_1), label="Inizializzazione a -1")
    plt.plot(jnp.arange(n_steps), jax.vmap(V)(trajectoryR), label="Inizializzazione random")
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

def plot_trajectory_x_relative_freq(results, D, Ts):
    #subplot of the histogram of the relative frequency of x[0] for different temperatures, for a specific D
    _ensure_report_dir()
    plt.figure(figsize=(12, 8))
    for i, T in enumerate(Ts):
        plt.subplot(2, 3, i+1)
        trajectory_x = results[D]["trajectory_x"][results[D]["T"].index(T)]
        plt.hist(trajectory_x, bins=50, 
         weights=np.ones(len(trajectory_x)) / len(trajectory_x))
        plt.xlabel("x[0]")
        plt.ylabel("Frequenza relativa")
        plt.title(f"D={D}, T={T}")
        plt.grid()
    plt.tight_layout()
    filename = f"trajectory_x_relative_freq_D{D}.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()