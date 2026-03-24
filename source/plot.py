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