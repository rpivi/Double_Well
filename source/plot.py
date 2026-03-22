import matplotlib.pyplot as plt

def plot_obs__D_T(results, dimensions, observable="E_mean"):
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
    plt.show()