import matplotlib.pyplot as plt
import os

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def plot_obs__D_T(results, dimensions, observable):
    _ensure_report_dir()
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

def plot_trajectory(trajectories, temperatures, dimensions):
    _ensure_report_dir()

    #one image with subplots for each dimension and temperature
    fig, axes = plt.subplots(len(dimensions), len(temperatures), figsize=(15, 10))
    for i, D in enumerate(dimensions):
        for j, T in enumerate(temperatures):
            ax = axes[i, j]
            ax.hist(trajectories[D][j], bins=30, density=True)
            ax.set_title(f"D={D}, T={T:.2f}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("Density")
    plt.tight_layout()
    plt.legend()
    filename = f"trajectories_density.png"
    filepath = os.path.join(REPORT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {filepath}")
    plt.close()