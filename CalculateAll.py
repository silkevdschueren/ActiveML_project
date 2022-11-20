from Calculate import *
import sys

def main():

    # Arguments given to the program in command line
    classifier, active, n_samples, batch_size, n = sys.argv[1:]
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    n = int(n)

    # Read in dataframe
    data = pd.read_csv('dataset_1.csv')
    
    # Define the target and categorize it in numbers
    target = 'PROFILE'
    profile_options = np.unique(data['PROFILE'])
    for index, profile in enumerate(profile_options):
        data.loc[data['PROFILE'] == profile,'PROFILE'] = index
    
    # Then get your input features and labels
    X = np.array(data.drop(target, axis=1).values.tolist())
    y_true = np.array(data[target]).astype('int')
    

    CalcActiveML(X, y_true, classifier, active, n_samples, batch_size, n=10)

    return


if __name__ == "__main__":
    main()
