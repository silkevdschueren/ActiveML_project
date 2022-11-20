import numpy as np

#Different classifiers
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from skactiveml.pool import UncertaintySampling, QueryByCommittee, GreedySamplingX
from skactiveml.classifier import SklearnClassifier
from skactiveml.utils import unlabeled_indices, labeled_indices, MISSING_LABEL
from skactiveml.pool import RandomSampling, UncertaintySampling, ProbabilisticAL, CostEmbeddingAL, EpistemicUncertaintySampling


def class_accs(y_pred, y_true):
    """
    Estimate the accuracy of the model on each class.

    :param y_pred: Output predicted by the model.
    :param y_test: True output.
    :return: List of accuracies of the model per class.
    """

    # Different classes in the data.
    classes = np.unique(y_true)
    acc = np.zeros(len(classes))
    
    # Derermine accuracy for each class.
    for i, y in enumerate(classes):
        acc[i] = ((y_pred == y_true) & (y_true == y)).sum() / (y_true == y).sum()

    return acc


def balanced_accuracy(y_pred, y_test):
    """
    Estimate balanced accuracy of the model, using accuracies for all classes.

    :param y_pred: Output predicted by the model.
    :param y_test: True output.
    :return: Accuracy of the model balanced over all classes.
    """

    # Balanced accuracy is the mean of the class accuracies.
    acc = class_accs(y_pred, y_test)
    return np.sum(acc) / len(acc)

def grid_search(X, y, model, model_name):
    """
    Tuning the hyperparameters by doing a grid search

    :param X: Features of the dataset
    :param y: Classes of the dataset
    :param model: classification model
    :return: The best parameters
    """
    if model_name == 'DecisionTree':
        p_grid = {"criterion": ['gini', 'entropy'], "max_depth": [3, 4, 5, None]}
    elif model_name == 'SVM':
        p_grid = {"C": [1, 10, 100], "kernel": ['linear', 'poly', 'rbf']}
    elif model_name == 'LogisticRegression':
        p_grid = {"penalty": ['l1', 'l2', 'none'], "C": [1, 10, 100]}
    else:
        p_grid = {"criterion": ['gini', 'entropy'], "max_depth": [3, 4, 5, None]}
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    clf = GridSearchCV(estimator=model, param_grid=p_grid, scoring="balanced_accuracy", cv=cv)
    clf.fit(X, y)
    return clf.best_params_


def get_clf_after_gridsearch(X, y, y_true, model, model_name):
    """
    Get the clf with the best parameters after doing a gridsearch

    :param X: Features of the dataset
    :param y: Classes of the dataset
    :param model: classification model
    :return: The best parameters
    """
    best_param = grid_search(X, y, model, model_name)
    if classifier == "LogisticRegression":
        clf = SklearnClassifier(LogisticRegression(penalty = best_param["penalty"], C = best_param["C"]), classes=np.unique(y_true))
    if classifier == "RandomForest":
        clf = SklearnClassifier(RandomForestClassifier(criterion = best_param["criterion"], max_depth = best_param["max_depth"]), classes=np.unique(y_true))
    if classifier == "SVM":
        clf = SklearnClassifier(SVC(C = best_param["C"], kernel = best_param["kernel"], probability=True), classes=np.unique(y_true))
    if classifier == "DecisionTree":
        clf = SklearnClassifier(DecisionTreeClassifier(criterion = best_param["criterion"], max_depth = best_param["max_depth"]), classes=np.unique(y_true))   
    return clf


def ActiveLearning_CrossValidation(X, y_true, folds, AL_method, n_cycles, batch_size, clf, classifier, n=10):
    """
    Use kfold cross validation to estimate generalisation performance of an active learning model.
    
    :param X: Features of the dataset.
    :param y_true: True classes of the dataset.
    :param folds: Number of partitions for kfold cross validation.
    :param AL_method: Active learning method to determine performance of.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param clf: Classifier method.
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Accuracy of a given method determined by kfold cross validation as a function of
    the number of labeled indices.
    """
    
    skf = StratifiedKFold(n_splits=folds)
    acc = []
    for i, (train_index, test_index) in enumerate(skf.split(X,y_true)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
    
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        cycle, accuracies, indep_accuracies = AL_method(X_train_scaled, X_test_scaled, y_train, 
                y_test, n_cycles, batch_size, clf, classifier, n=10)
        acc.append(indep_accuracies)
        
    return cycle, np.mean(acc, axis=0)


def ActiveLearning_UncertaintySampling(X_train, X_test, y_train, y_test, n_cycles, batch_size, model, model_name, method='entropy', n=10):
    """
    In uncertainty sampling, the unlabeled instance that the model is the least certain about 
    is chosen.
    
    :param X_train: Features of the dataset used for training of the model, already scaled.
    :param X_test: Features of the dataset, used as independent test set.
    :param y_train: True classes of the dataset, used for training.
    :param y_test: True classes of the dataset, used as independent testset.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param model: Classifier model.
    :param model_name: name of the Classifier
    :param method: Optimal parameter determining the measure for uncertainty. Default is entropy.
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Lists giving the number of labeled samples, the accuracy as determined from the 
    training set and the accuracy using an independent test set.
    """
        
    # Create initial set of labeled data and train initial model
    y = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)
    initials = np.random.randint(0, len(y_train), batch_size)
    y[initials] = y_train[initials]

    # Grid search on parameters
    X_grid = X_train[initials]
    y_grid = y_train[initials]
    clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
    clf.fit(X_train, y)
    
    # Active learning with uncertainty sampling.
    qs = UncertaintySampling(method=method)
    accuracies, indep_accuracies, cycle = [], [], []

    for c in range(n_cycles):
        # Create query to add to labeled data.
        query_idx = qs.query(X=X_train, y=y, clf=clf, batch_size=batch_size)
        y[query_idx] = y_train[query_idx]
        X_grid = np.append(X_grid, X_train[query_idx], axis = 0)
        y_grid = np.append(y_grid, y_train[query_idx], axis = 0)
        
        # Every nth cycle do another grid search
        if c%n == 0 and c != 0:
            clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
        
        # Train new model on labeled data.
        clf.fit(X_train, y)
        
        # Estimate performance of the model on the training set.
        accuracies.append(clf.score(X_train, y_train))

        # Estimate generalisation of the model using testset. Note that this data is not used 
        # for the model, only for the returned accuracies to follow active learning performance.
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy(y_pred, y_test)
        indep_accuracies.append(bal_acc)

        # Save number of labeled samples.
        cycle.append(c*batch_size)
    
    return cycle, accuracies, indep_accuracies


def ActiveLearning_RandomSampling(X_train, X_test, y_train, y_test, n_cycles, batch_size, model, model_name, n=10):
    """
    In random sampling, the query is chosen randomly from the unlabeled instances.
    
    :param X_train: Features of the dataset used for training of the model, already scaled.
    :param X_test: Features of the dataset, used as independent test set.
    :param y_train: True classes of the dataset, used for training.
    :param y_test: True classes of the dataset, used as independent testset.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param model: Classifier model.
    :param model_name: name of the Classifier
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Lists giving the number of labeled samples, the accuracy as determined from the 
    training set and the accuracy using an independent test set.
    """
        
    # Create initial set of labeled data and train initial model
    y = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)
    initials = np.random.randint(0, len(y_train), batch_size)
    y[initials] = y_train[initials]
    
    # Grid search on parameters
    X_grid = X_train[initials]
    y_grid = y_train[initials]
    clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
    clf.fit(X_train, y)
    
    # Active learning with random sampling.
    qs = RandomSampling()
    accuracies, indep_accuracies, cycle = [], [], []

    for c in range(n_cycles):
        # Create query to add to labeled data.
        query_idx = qs.query(X=X_train, y=y, batch_size=batch_size)
        y[query_idx] = y_train[query_idx]
        X_grid = np.append(X_grid, X_train[query_idx], axis = 0)
        y_grid = np.append(y_grid, y_train[query_idx], axis = 0)
        
        # Every nth cycle do another grid search
        if c%n == 0 and c != 0:
            clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
            
        # Train new model on labeled data.
        clf.fit(X_train, y)
        
        # Estimate performance of the model on the training set.
        accuracies.append(clf.score(X_train, y_train))

        # Estimate generalisation of the model using testset. Note that this data is not used 
        # for the model, only for the returned accuracies to follow active learning performance.
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy(y_pred, y_test)
        indep_accuracies.append(bal_acc)

        # Save number of labeled samples.
        cycle.append(c*batch_size)
    
    return cycle, accuracies, indep_accuracies


def ActiveLearning_QueryByCommittee(X_train, X_test, y_train, y_test, n_cycles, batch_size, model, model_name, committee_size=3, n=10):
    """
    The query-by-committee algorithm involves maintaining a committee of models which are all 
    trained on the current labeled set L, but representing competing hypotheses. Each committee 
    member is allowed to vote on the labeling of the query candidates. The most informative 
    query is considered to be the instance about which they most disagree.
    
    :param X_train: Features of the dataset used for training of the model, already scaled.
    :param X_test: Features of the dataset, used as independent test set.
    :param y_train: True classes of the dataset, used for training.
    :param y_test: True classes of the dataset, used as independent testset.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param model: Classifier model.
    :param model_name: name of the Classifier
    :param committee_size: Optional parameter giving number of methods in the committee. 
    Default value is three.
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Lists giving the number of labeled samples, the accuracy as determined from the 
    training set and the accuracy using an independent test set.
    """
    
    # Create initial set of labeled data and train initial model
    y = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)
    initials = np.random.randint(0, len(y_train), batch_size)
    y[initials] = y_train[initials]
    
    # Grid search on parameters
    X_grid = X_train[initials]
    y_grid = y_train[initials]
    clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)

    # Create ensemble of models as committee.
    ensemble_bagging = SklearnClassifier(estimator=BaggingClassifier(base_estimator=clf, 
        n_estimators=committee_size), classes=np.unique(y_train))
    ensemble_bagging.fit(X_train, y)
    
    # Active learning with query by committee.
    qs = QueryByCommittee()
    accuracies, indep_accuracies, cycle = [], [], []
    
    for c in range(n_cycles):
        # Create query to add to labeled data.
        query_idx = qs.query(X=X_train, y=y, ensemble=ensemble_bagging, batch_size=batch_size)
        y[query_idx] = y_train[query_idx]
        X_grid = np.append(X_grid, X_train[query_idx], axis = 0)
        y_grid = np.append(y_grid, y_train[query_idx], axis = 0)
        
        # Every nth cycle do another grid search
        if c%n == 0 and c != 0:
            clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
            ensemble_bagging = SklearnClassifier(estimator=BaggingClassifier(base_estimator=clf, 
                                                                             n_estimators=committee_size), classes=np.unique(y_train))
        # Train new model on labeled data.
        ensemble_bagging.fit(X_train, y)
        
        # Estimate performance of the model on the training set.
        accuracies.append(ensemble_bagging.score(X_train, y_train))

        # Estimate generalisation of the model using testset. Note that this data is not used 
        # for the model, only for the returned accuracies to follow active learning performance.
        y_pred = ensemble_bagging.predict(X_test)
        bal_acc = balanced_accuracy(y_pred, y_test)
        indep_accuracies.append(bal_acc)

        # Save number of labeled samples.
        cycle.append(c*batch_size)
    
    return cycle, accuracies, indep_accuracies


def ActiveLearning_CostEmbeddingAL(X_train, X_test, y_train, y_test, n_cycles, batch_size, model, model_name, n=10):
    """
    The Cost Embedding algorithm is a cost sensitive multi-class algorithm. It assumes that
    each class has at least one sample in the labeled pool. Cost-sensitive active learning 
    algorithms allows the user to pass in the cost matrix as a parameter and select the data 
    points that it thinks to perform the best on the given cost matrix. Assume we have a total 
    of K classes, cost matrix can be represented as a K*K matrix. The i-th row, j-th column 
    represents the cost of the ground truth being i-th class and prediction as j-th class. 
    The goal is to minimize the total cost.
    
    :param X_train: Features of the dataset used for training of the model, already scaled.
    :param X_test: Features of the dataset, used as independent test set.
    :param y_train: True classes of the dataset, used for training.
    :param y_test: True classes of the dataset, used as independent testset.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param model: Classifier model.
    :param model_name: name of the Classifier
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Lists giving the number of labeled samples, the accuracy as determined from the 
    training set and the accuracy using an independent test set.
    """
    
    # Create initial set of labeled data and train initial model
    y = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)
    initials = np.random.randint(0, len(y_train), batch_size)
    y[initials] = y_train[initials]
    
    # Grid search on parameters
    X_grid = X_train[initials]
    y_grid = y_train[initials]
    clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
    clf.fit(X_train, y)
    
    # Active learning with cost embedding.
    qs = CostEmbeddingAL(classes=np.unique(y_train))
    accuracies, indep_accuracies, cycle = [], [], []

    for c in range(n_cycles):
        # Create query to add to labeled data.
        query_idx = qs.query(X=X_train, y=y, batch_size=batch_size)
        y[query_idx] = y_train[query_idx]
        X_grid = np.append(X_grid, X_train[query_idx], axis = 0)
        y_grid = np.append(y_grid, y_train[query_idx], axis = 0)
        
        # Every nth cycle do another grid search
        if c%n == 0 and c != 0:
            clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
            
        # Train new model on labeled data.
        clf.fit(X_train, y)
        
        # Estimate performance of the model on the training set.
        accuracies.append(clf.score(X_train, y_train))

        # Estimate generalisation of the model using testset. Note that this data is not used 
        # for the model, only for the returned accuracies to follow active learning performance.
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy(y_pred, y_test)
        indep_accuracies.append(bal_acc)

        # Save number of labeled samples.
        cycle.append(c*batch_size)
    
    return cycle, accuracies, indep_accuracies


def ActiveLearning_GreedySamplingX(X_train, X_test, y_train, y_test, n_cycles, batch_size, model, model_name, n=10):
    """
    In greedy sampling, the query strategy tries to select those samples that increase the 
    diversity of the feature space the most.
    
    :param X_train: Features of the dataset used for training of the model, already scaled.
    :param X_test: Features of the dataset, used as independent test set.
    :param y_train: True classes of the dataset, used for training.
    :param y_test: True classes of the dataset, used as independent testset.
    :param n_cycles: Number of times that a query has to be taken.
    :param batch_size: Size of the batch of a single query.
    :param model: Classifier model.
    :param model_name: name of the Classifier
    :param n: Number of cycles at which there needs to be hyperparameter tuning
    Default is 10.
    :return: Lists giving the number of labeled samples, the accuracy as determined from the 
    training set and the accuracy using an independent test set.
    """
        
    # Create initial set of labeled data and train initial model
    y = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)
    initials = np.random.randint(0, len(y_train), batch_size)
    y[initials] = y_train[initials]
    
    # Grid search on parameters
    X_grid = X_train[initials]
    y_grid = y_train[initials]
    clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)
    clf.fit(X_train, y)
    
    # Active learning with greedy sampling.
    qs = GreedySamplingX()
    accuracies, indep_accuracies, cycle = [], [], []

    for c in range(n_cycles):
        # Create query to add to labeled data.
        query_idx = qs.query(X=X_train, y=y, batch_size=batch_size)
        y[query_idx] = y_train[query_idx]
        X_grid = np.append(X_grid, X_train[query_idx], axis = 0)
        y_grid = np.append(y_grid, y_train[query_idx], axis = 0)
        
        # Every nth cycle do another grid search
        if c%n == 0 and c != 0:
            clf = get_clf_after_gridsearch(X_grid, y_grid, y_true, model, model_name)

        # Train new model on labeled data.
        clf.fit(X_train, y)
        
        # Estimate performance of the model on the training set.
        accuracies.append(clf.score(X_train, y_train))

        # Estimate generalisation of the model using testset. Note that this data is not used 
        # for the model, only for the returned accuracies to follow active learning performance.
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy(y_pred, y_test)
        indep_accuracies.append(bal_acc)

        # Save number of labeled samples.
        cycle.append(c*batch_size)
    
    return cycle, accuracies, indep_accuracies