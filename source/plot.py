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
    plt.title("Evoluzione dell'energia durante la termalizzazione per D={} e T={}".format(D, T))
    plt.legend()
    plt.grid()
    filename = f"thermalization_energies_D{D}_T{T:.2f}.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()

def plot_obs_D_T(results, dimensions, observable, error=None, a=None, b=None):
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
        plt.legend(loc="upper right")
        plt.grid()

        if a is not None and b is not None:
            param_text = (rf"$a = {a:.3f}$" + "\n" + rf"$b = {b:.3f}$")

            plt.text(0.05, 0.95,param_text, transform=plt.gca().transAxes,fontsize=11,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9)           )

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

        if a is not None and b is not None:
            param_text = (rf"$a = {a:.3f}$" + "\n" + rf"$b = {b:.3f}$")

            plt.text(0.05, 0.95,param_text, transform=plt.gca().transAxes,fontsize=11,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9)           )

        filename = f"{observable}_vs_T.png"
        filepath = os.path.join(REPORT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {filepath}")
        plt.close()

def plot_tau(results, dimensions, a=None, b=None):
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
        
    if a is not None and b is not None:
            param_text = (rf"$a = {a:.3f}$" + "\n" + rf"$b = {b:.3f}$")

            plt.text(0.05, 0.95,param_text, transform=plt.gca().transAxes,fontsize=11,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9)           )

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