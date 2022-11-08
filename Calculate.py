import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import pipeline, preprocessing

from Methods import *


def CalcAllDecisionTree(X, y_true, n_samples=23, batch_size=100, folds=5):
    """
    Calculate the information needed to plot the learning curves for all active learning 
    models on a decision tree, and write this information to a txt file.

    :param X: Features of the data.
    :param y_true: True labels of the data.
    :param n_samples: Optional parameter giving the number of times a sample should be taken. 
    Default is 23.
    :param batch_size: Optional parameter giving the batch size. Note that 
    n_samples * batch_size can be maximal 4/5*(total size dataset) - batch_size 
    if there is 5 fold cross validation. Default value 100.
    :param folds: Optional parameter giving the number of partitions for cross validation.
    Default is 5.
    """

    clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    cycle1, accuracies1 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_RandomSampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies2 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_UncertaintySampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies3 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_QueryByCommittee, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies4 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_CostEmbeddingAL, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(DecisionTreeClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies5 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_GreedySamplingX, n_samples, batch_size, clf)
    

    # Write information to file.
    f = open(f'dt_{n_samples*batch_size}_{batch_size}.txt', 'w')
    for i, cycle in enumerate(cycle1):
        f.write(str(cycle) + ' ' + str(accuracies1[i]) + ' ' + str(accuracies2[i]) + ' ' + 
                str(accuracies3[i]) + ' ' + str(accuracies4[i]) + ' ' + str(accuracies5[i]) + '\n')
    f.close()

    return


def CalcAllSVM(X, y_true, n_samples=23, batch_size=100, folds=5):
    """
    Calculate the information needed to plot the learning curves for all active learning 
    models on a support vector machine, and write this information to a txt file.

    :param X: Features of the data.
    :param y_true: True labels of the data.
    :param n_samples: Optional parameter giving the number of times a sample should be taken. 
    Default is 23.
    :param batch_size: Optional parameter giving the batch size. Note that 
    n_samples * batch_size can be maximal 4/5*(total size dataset) - batch_size 
    if there is 5 fold cross validation. Default value 100.
    :param folds: Optional parameter giving the number of partitions for cross validation.
    Default is 5.
    """

    clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), 
            classes=np.unique(y_true))
    cycle1, accuracies1 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_RandomSampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), 
            classes=np.unique(y_true))
    _, accuracies2 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_UncertaintySampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), 
            classes=np.unique(y_true))
    _, accuracies3 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_QueryByCommittee, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), 
            classes=np.unique(y_true))
    _, accuracies4 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_CostEmbeddingAL, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(SVC(kernel = 'rbf', C = 10, probability = True), 
            classes=np.unique(y_true))
    _, accuracies5 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_GreedySamplingX, n_samples, batch_size, clf)

    # Write information to file
    f = open(f'svm_{n_samples*batch_size}_{batch_size}.txt', 'w')
    for i, cycle in enumerate(cycle1):
        f.write(str(cycle) + ' ' + str(accuracies1[i]) + ' ' + str(accuracies2[i]) + ' ' + 
                str(accuracies3[i]) + ' ' + str(accuracies4[i]) + ' ' + str(accuracies5[i]) + '\n')
    f.close()
    
    return


def CalcAllLogisticRegression(X, y_true, n_samples=23, batch_size=100, folds=5):
    """
    Calculate the information needed to plot the learning curves for all active learning 
    models on logistic regression, and write this information to a txt file.

    :param X: Features of the data.
    :param y_true: True labels of the data.
    :param n_samples: Optional parameter giving the number of times a sample should be taken. 
    Default is 23.
    :param batch_size: Optional parameter giving the batch size. Note that 
    n_samples * batch_size can be maximal 4/5*(total size dataset) - batch_size 
    if there is 5 fold cross validation. Default value 100.
    :param folds: Optional parameter giving the number of partitions for cross validation.
    Default is 5.
    """

    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    cycle1, accuracies1 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_RandomSampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    _, accuracies2 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_UncertaintySampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    _, accuracies3 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_QueryByCommittee, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    _, accuracies4 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_CostEmbeddingAL, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    _, accuracies5 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_GreedySamplingX, n_samples, batch_size, clf)

    # Write information to file
    f = open(f'lr_{n_samples*batch_size}_{batch_size}.txt', 'w')
    for i, cycle in enumerate(cycle1):
        f.write(str(cycle) + ' ' + str(accuracies1[i]) + ' ' + str(accuracies2[i]) + ' ' + 
                str(accuracies3[i]) + ' ' + str(accuracies4[i]) + ' ' + str(accuracies5[i]) + '\n')
    f.close()

    return


def CalcAllRandomForest(X, y_true, n_samples=23, batch_size=100, folds=5):
    """
    Calculate the information needed to plot the learning curves for all active learning 
    models on a random forest classifier, and write this information to a txt file.

    :param X: Features of the data.
    :param y_true: True labels of the data.
    :param n_samples: Optional parameter giving the number of times a sample should be taken. 
    Default is 23.
    :param batch_size: Optional parameter giving the batch size. Note that 
    n_samples * batch_size can be maximal 4/5*(total size dataset) - batch_size 
    if there is 5 fold cross validation. Default value 100.
    :param folds: Optional parameter giving the number of partitions for cross validation.
    Default is 5.
    """

    clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    cycle1, accuracies1 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_RandomSampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies2 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_UncertaintySampling, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies3 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_QueryByCommittee, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies4 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_CostEmbeddingAL, n_samples, batch_size, clf)
    
    clf = SklearnClassifier(RandomForestClassifier(criterion = 'entropy'), classes=np.unique(y_true))
    _, accuracies5 = ActiveLearning_CrossValidation(X, y_true, folds, 
            ActiveLearning_GreedySamplingX, n_samples, batch_size, clf)

    # Write information to file
    f = open(f'rf_{n_samples*batch_size}_{batch_size}.txt', 'w')
    for i, cycle in enumerate(cycle1):
        f.write(str(cycle) + ' ' + str(accuracies1[i]) + ' ' + str(accuracies2[i]) + ' ' + 
                str(accuracies3[i]) + ' ' + str(accuracies4[i]) + ' ' + str(accuracies5[i]) + '\n')
    f.close()

    return
       
