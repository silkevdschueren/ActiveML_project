import numpy as np
import matplotlib.pyplot as plt
import sys


def main():

    classifier, actives, cycles, batch_size, n_hypertuning = sys.argv[1:]
    actives = actives.strip('[]').split(',')

    fig, ax = plt.subplots()

    for active in actives:
        x, y = np.loadtxt(f"AllBatch{batch_size}/{classifier}_{active}_{cycles}_{batch_size}_{n_hypertuning}.txt", unpack=True)
        ax.plot(x, y, label=active )
        ax.set(xlabel="Labeled samples", ylabel="Accuracy", 
                title=f"{classifier}: {cycles} cycles, batch size {batch_size}, hypertuning every {n_hypertuning} cycles")

    plt.legend()
    plt.savefig(f"{classifier}_{cycles}_{batch_size}_{n_hypertuning}")


if __name__ == "__main__":
    main()
