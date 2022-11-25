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
    
    colors=['red', 'orange', 'yellow', 'green', 'blue']

    for classifier in classifiers:
        for active in actives: 
            fig, ax = plt.subplots()

            for index, (batch_size, cycle) in enumerate(zip(batch_sizes, cycles)):
                x, y = np.loadtxt(f"InitialTuning/AllBatch{batch_size}/{classifier}_{active}_{cycle}_{batch_size}_{n_hypertuning}.txt", unpack=True)
                ax.plot(x, y, label=f"batch size {batch_size}", color=colors[index])

            ax.set(xlabel="Labeled samples", ylabel="Accuracy", 
                   title=f"{classifier} {active}: Influence of batch size, with\nhyperparameter tuning every {n_hypertuning} cycles and additional initial tuning")

            plt.legend()
            plt.savefig(f"Figures/Batchsize_{classifier}_{active}_{n_hypertuning}")


if __name__ == "__main__":
    main()
