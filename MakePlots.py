import numpy as np
import matplotlib.pyplot as plt
import sys


def main():

    n_hypertuning = 10

    classifiers = ['DecisionTree', 'RandomForest', 'SVM', 'LogisticRegression']
    actives = ['RandomSampling', 'UncertaintySampling', 'QueryByCommittee', 
            'CostEmbedding', 'GreedySampling']

    batch_sizes = [50, 75, 100, 125, 150]
    cycles = [280, 186, 140, 112, 93]

    for batch_size, cycle in zip(batch_sizes, cycles):
        for classifier in classifiers:
            fig, ax = plt.subplots()

            for active in actives:
                x, y = np.loadtxt(f"InitialTuning/AllBatch{batch_size}/{classifier}_{active}_{cycle}_{batch_size}_{n_hypertuning}.txt", unpack=True)
                ax.plot(x, y, label=active )
                ax.set(xlabel="Labeled samples", ylabel="Accuracy", 
                        title=f"{classifier}: {cycle} cycles, batch size {batch_size}, hyperparameter tuning\nevery {n_hypertuning} cycles and additional initial tuning")

            plt.legend()
            plt.savefig(f"Figures/InitialHypertuning_{classifier}_{cycle}_{batch_size}_{n_hypertuning}")


if __name__ == "__main__":
    main()
