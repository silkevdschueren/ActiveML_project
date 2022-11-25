import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import colors


def colorgradient(total):
    cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

    cmap = colors.LinearSegmentedColormap('custom', cdict)
    return cmap


def main():
    classifier, active, cycles, batch_size = sys.argv[1:]

    fig, ax = plt.subplots(figsize=(8,5))

    stepsize=2
    maxcycles=50
    totalcolors=maxcycles/stepsize
    cmap = colorgradient(totalcolors)

    for i, n_hypertuning in enumerate(range(maxcycles, 0, -stepsize)):
        x, y = np.loadtxt(f"Hypertuning/Hypertuning_DT_US_batch50/{classifier}_{active}_{cycles}_{batch_size}_{n_hypertuning:02d}.txt", 
                unpack=True)
        ax.plot(x, y, color=cmap(i/totalcolors))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=stepsize, vmax=maxcycles))
    plt.colorbar(sm, label='Number of cycles between tunings')
    ax.set(xlabel="Labeled samples", ylabel="Accuracy")
    plt.suptitle(f"Hypertuning {classifier}: {cycles} cycles, batch size {batch_size}")
#            \nwith additional initial hyperparameter tuning")
    plt.savefig(f"Figures/HypertuningFrequency_{classifier}_{cycles}_{batch_size}")


if __name__ == "__main__":
    main()
