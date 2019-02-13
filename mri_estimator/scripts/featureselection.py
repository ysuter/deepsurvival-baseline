#!/home/yannick/miniconda3/envs/tensorflow/bin/python

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

inputcsv = "/run/user/1000/gvfs/smb-share:server=istb-brain.unibe.ch,share=data/mia/BrainOncology/BraTS/Brats2018/Training/features_resection_categorical.csv"
outcsv = "/home/yannick/Dropbox/Doktorat/BraTS/featimp_newseg.csv"

data = pd.read_csv(inputcsv)
#data['ResectionStatus'] = data['ResectionStatus'].map({'NA': 0, 'GTR': 1 , 'STR': 2})
#data = load_boston()

features = list(data)[1:]
# print(features)
df = data[features]

bad_indices = np.where(np.isnan(df))

print(bad_indices[0])
print(bad_indices[1])

#imp = Imputer(strategy="mean", axis=0)
scaler = StandardScaler()
# Don't cheat - fit only on training data


X, y = df.drop('Survival',axis=1), df['Survival']
print(X.shape)
#X_train = (X - np.nanmean(X, axis=0)) / np.nanvar(X, axis=0)
#print(X_train.shape)
#X_new = scaler.fit_transform(imp.fit_transform(X))
#scaler.fit(X)
#X_train = scaler.transform(X_norm)
# apply same transformation to test data
#X_test = scaler.transform(X_test)
X_train = X
#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = data.target


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=2400,      #0.01
                       threshold_out = 2500,    #0.05
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X_train.columns)-set(included))
        #new_pval = pd.Series(index=excluded)
        new_aic = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_train[included+[new_column]])), missing='drop').fit()
            #print(model.aic)
            #exit()
            #new_pval[new_column] = model.pvalues[new_column]
            new_aic[new_column] = model.aic #[new_column] #    pvalues[new_column]
        best_aic = new_aic.min()
        if best_aic < threshold_in:
            best_feature = new_aic.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with AIC {:.6}'.format(best_feature, best_aic))

        # backward step
        #print("backward step")
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_train[included])), missing='drop').fit()
        # use all coefs except intercept
        aic = model.aic #[1:]
        worst_aic = aic.max() # null if pvalues is empty
        if worst_aic > threshold_out:
            changed=True
            worst_feature = aic.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with AIC {:.6}'.format(worst_feature, worst_aic))
        if not changed:
            break
    return included

result = stepwise_selection(X_train, y)

print('resulting features:')
print(result)