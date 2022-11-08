from Calculate import *


def main():

    print("Reading data...")
    #read in dataframe
    data = pd.read_csv('dataset_1.csv')
    
    #define the target and categorize it in numbers
    target = 'PROFILE'
    profile_options = np.unique(data['PROFILE'])
    for index, profile in enumerate(profile_options):
        data.loc[data['PROFILE'] == profile,'PROFILE'] = index
    
    #then get your input features and labels
    X = np.array(data.drop(target, axis=1).values.tolist())
    y_true = np.array(data[target]).astype('int')
    
    
    print("Calculating Random Forest...")
    CalcAllRandomForest(X, y_true, n_samples=3, batch_size=100)
    print("Calculating Logistic Regression...")
    CalcAllLogisticRegression(X, y_true, n_samples=3, batch_size=100)
    print("Calculating SVM...")
    CalcAllSVM(X, y_true, n_samples=3, batch_size=100)
    print("Calculating Decision Tree...")
    CalcAllDecisionTree(X, y_true, n_samples=3, batch_size=100)

    return


if __name__ == "__main__":
    main()
