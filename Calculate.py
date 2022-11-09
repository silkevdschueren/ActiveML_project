import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import pipeline, preprocessing

from Methods import *


def CalcActiveML(X, y_true, classifier, active, n_samples=23, batch_size=100, folds=5):
    """
    Calculate the information needed to plot the learning curves for a given active learning 
    model on a given machine learning model, and write this information to a txt file.

    :param X: Features of the data.
    :param y_true: True labels of the data.
    :param classifier: Classifier to use. Options are:
    RandomForest, LogisticRegression, SVM, DecisionTree.
    :param active: Active learning method to use. Options are:
    RandomSampling, UncertaintySampling, QueryByCommittee, CostEmbedding, GreedySampling.
    :param n_samples: Optional parameter giving the number of times a sample should be taken. 
    Default is 23.
    :param batch_size: Optional parameter giving the batch size. Note that 
    n_samples * batch_size can be maximal 4/5*(total size dataset) - batch_size 
    if there is 5 fold cross validation. Default value 100.
    :param folds: Optional parameter giving the number of partitions for cross validation.
    Default is 5.
    """

    if classifier == "LogisticRegression":
        clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))

    if classifier == "RandomForest":
        clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))

    if classifier == "SVM":
        clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), classes=np.unique(y_true))

    if classifier == "DecisionTree":
        clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))


    if active == "RandomSampling":
        cycles, accuracies = ActiveLearning_CrossValidation(X, y_true, folds, 
                ActiveLearning_RandomSampling, n_samples, batch_size, clf)
    
    if active == "UncertaintySampling":
        cycles, accuracies = ActiveLearning_CrossValidation(X, y_true, folds, 
                ActiveLearning_UncertaintySampling, n_samples, batch_size, clf)
        
    if active == "QueryByCommittee":
        cycles, accuracies = ActiveLearning_CrossValidation(X, y_true, folds, 
                ActiveLearning_QueryByCommittee, n_samples, batch_size, clf)
        
    if active == "CostEmbedding":
        cycles, accuracies = ActiveLearning_CrossValidation(X, y_true, folds, 
                ActiveLearning_CostEmbeddingAL, n_samples, batch_size, clf)
        
    if active == "GreedySampling":
        cycles, accuracies = ActiveLearning_CrossValidation(X, y_true, folds, 
                ActiveLearning_GreedySamplingX, n_samples, batch_size, clf)


    # Write information to file.
    f = open(f'{classifier}_{active}_{n_samples}_{batch_size}.txt', 'w')
    for i, cycle in enumerate(cycles):
        f.write(str(cycle) + ' ' + str(accuracies[i]) + '\n')
    f.close()

    return
