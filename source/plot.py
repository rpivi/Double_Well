import matplotlib.pyplot as plt
import numpy as np
import os
import jax
import jax.numpy as jnp
from typing import Callable

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")

def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def plot_thermalization_energies(trajectory1,trajectoryR, T, D, n_steps, V: Callable):
    _ensure_report_dir()
    plt.figure(figsize=(8, 6))
    plt.plot(jnp.arange(n_steps), jax.vmap(V)(trajectory1), label="Inizializzazione a 1")
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

def plot_trajectory_x_relative_freq(results, Ds, Ts):
    _ensure_report_dir()
    n_rows = len(Ds)
    n_cols = len(Ts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
                             sharex=False, sharey=False)  # scale indipendenti
    # gestione caso n_rows=1 o n_cols=1
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, D in enumerate(Ds):
        for j, T in enumerate(Ts):
            trajectory_x = results[D]["trajectory_x"][results[D]["T"].index(T)]
            axes[i, j].hist(trajectory_x, bins=30, density=True)
            axes[i, j].set_title(f"D={D}, T={T}")
            axes[i, j].set_xlabel("x[0]")
            axes[i, j].set_ylabel("Freq. relativa")
            axes[i, j].grid()

    fig.suptitle("Distribuzione di x[0] per diverse dimensioni e temperature", fontsize=13)
    plt.tight_layout()
    filename = f"trajectory_x_relative_freq.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()

def plot_trajectory_x_t(results, Ds, Ts):
    _ensure_report_dir()
    n_rows = len(Ds)
    n_cols = len(Ts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
                             sharex=True, sharey=False)  # stesso asse x (passi), y libero
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, D in enumerate(Ds):
        for j, T in enumerate(Ts):
            trajectory_x = results[D]["trajectory_x"][results[D]["T"].index(T)]
            axes[i, j].plot(trajectory_x)
            axes[i, j].set_title(f"D={D}, T={T}")  # titolo completo su ogni subplot
            axes[i, j].set_xlabel("Passi di Metropolis")
            axes[i, j].set_ylabel("x[0]")
            axes[i, j].grid()

    fig.suptitle("Traiettoria di x[0] per diverse dimensioni e temperature", fontsize=13)
    plt.tight_layout()
    filename = "trajectory_x_t.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()

def plot_tau(results, dimensions):
    _ensure_report_dir()
    plt.figure(figsize=(8,6))

    #plot skipping the first 3 temperatures to avoid the low T regime where tau is very large and dominates the plot
    for D in dimensions:
        plt.plot(
                results[D]["T"][3:],
                results[D]["tau_x"][3:],
                marker='o',
                label=f"D={D}"
            )

    plt.xlabel("Temperatura")
    plt.ylabel("Tau")
    plt.title("Tau vs Temperatura per diverse dimensioni")
    plt.legend()
    plt.grid()

    filename = "Tau_vs_T.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()