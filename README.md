# ActiveML_project
Project on active machine learning


Before doing calculations, the data should be preprocessed. For this, one can use the three notebooks for 
the three different datasets.

To perform a calculation, one should run the python file CalculateAll.py from the command line, giving 
- the desired classifier (options: DecisionTree, SVM, LogisticRegression, RandomForest), 
- active learning method (options: RandomSampling, UncertaintySampling, QueryByCommittee, CostEmbedding, GreedySampling)
- the number of learning cycles in the active learning method
- the number of samples in a batch
- the number of learning cycles in between hyperparameter tunings
- a parameter indicating if additional initial hyperparameter tuning is desired (options: 0 - no initial tuning, 1 - initial tuning)

The functions used for this calculation can be found in Calculate.py, which selects the desired classifier and active 
learning method, and writes all calculated information to a file formatted as classifier_active_ncycles_batchsize_cyclesbetweentunings.txt.
All other methods and implementations used for the model training can be found in the file Methods.py.

In order to plot the results, one should have a look at the files MakePlots.py, PlotBatches.py and PlotHypertuning.py. 



