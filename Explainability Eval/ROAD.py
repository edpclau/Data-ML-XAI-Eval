#### Remove and Retrain ####
# This is a version of ROAD (as described in Rong et al 2022) 
# which works with any scikit-learn model and tabular data (pandas dataframes)

## Libraries ##
#General
import numpy as np
# import pandas as pd
import copy

#Imputation Methods
import miceforest as mf


#Model Explanation
import shap 

#Model Evalutation
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score


#### Code ####

## Explanation Function ##
# Trains a model, builds an explanation with KernelShap, and then
# outputs the ranking of each feature in descending order.
# Arguments:
# clf : A trained ML model with a .predict method
# X: Training data as pandas dataframe
# x:  Test data as pandas dataframe
# explainer: any explainer that follows the shap api format
def explain(clf, X, x, explainer = shap.explainers.Permutation):
    if explainer != shap.explainers.Tree and explainer != shap.explainers.Linear:
        try:
                    # #Build Explanation
            explanation = explainer(clf.predict_proba, X)
            shap_values = np.abs(explanation(x).values[...,1]).mean(0)
            return shap_values
        
        except:
        # #Build Explanation
            explanation = explainer(clf.predict, X)
            shap_values = np.abs(explanation(x).values).mean(0)
            return shap_values

    elif explainer == shap.explainers.Tree:
        # #Build Explanation
        explanation = explainer(clf, X)
        shap_values = explanation(x).values
        if len(shap_values.shape) == 2:
            shap_values = np.abs(shap_values).mean(0)
        else:
            shap_values = np.abs(shap_values[...,1]).mean(0)
        return shap_values
    else:
        # #Build Explanation
        explanation = explainer(clf, X)
        shap_values = np.abs(explanation(x).values).mean(0)
        return shap_values


## Ranking Function ##
# Outputs the ranking of each feature in descending order.
# Arguments:
# shap_values: the output of any shap explainer
def ranker(shap_values):
    values = copy.deepcopy(shap_values)     
    #Get ranks
    ranks = np.argsort(values)
    return ranks


## Metrics function ##
# Utility function which outputs a series of metrics to evaluate
# Currently gets accuracy, balanced_accuracy, and f-score

#Arguments:
#clf: A trained ML model with a predict method
#x: a pd.DataFrame of test data
#y: Target of the test data
def metrics(clf, x, y):   
    #Predict
    yhat = clf.predict(x)
    try:
        yscore = clf.predict_proba(x)[:, 1]
    except:
        yscore = clf.decision_function(x)
    #Get metrics
    accu = accuracy_score(y, yhat)
    accu_balanced = balanced_accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    auroc = roc_auc_score(y, yscore)
    auprc = average_precision_score(y, yscore)
    recall = recall_score(y, yhat)
        
    return np.array([[accu], [accu_balanced], [f1], [auroc], [auprc], [recall]])



## Imputation Function assistant ##
def impute(i,k, rankings, X_train, x_test):
       #Add NA values
        X_train = mf.ampute_data(X_train, variables= X_train.columns[rankings[i:k]].to_list(), perc = 0.95, random_state=42)
        x_test = mf.ampute_data(X_train, variables= x_test.columns[rankings[i:k]].to_list(), perc = 0.95, random_state=42)
        #impute
        variables = X_train.columns.to_list()
        index = rankings.copy()[i:k].pop()
        kds = mf.ImputationKernel(
        X_train,
        variable_schema={variables.pop(index) : variables},
        save_all_iterations=True,
        random_state=1991
        )
        # Run the MICE algorithm for 2 iterations
        kds.mice(2)
        # Return the completed dataset.
        X_train = kds.complete_data()

        variables = x_test.columns.to_list()
        index = rankings.copy()[i:k].pop()
        kds = mf.ImputationKernel(
        x_test,
        variable_schema={variables.pop(index) : variables},
        save_all_iterations=True,
        random_state=1991
        )
        # Run the MICE algorithm for 2 iterations
        kds.mice(2)
        # Return the completed dataset.
        x_test = kds.complete_data()

        return X_train, x_test


## Mask Features ## 
#These functions mask data using imputation and retrain the ML model
#They mask from the top %, bottom %, or random.

# Arguments:
# t: percentage or number of features to be masked in each iteration
# rankings: a list of rankings (descending rankings) to guide the removal
# X, x: dataframes of the data from which we will remove the features. Training set and testing set, respectively
# Y, y: Targets for the training and testing sets, respectively
#clf: Model to retrain
# base: metrics of the full model. Used to compare the performance of the masked model
# direction: direction of masking. Can be 'top', 'bottom', or 'random'
# seed: seed for random number generator

#Mask
def mask(t, rankings, X, Y, x, y, clf, base = np.empty((3,1)), direction = 'top'):
    #Make copies of our data to modify
    X_train = copy.deepcopy(X)
    x_test = copy.deepcopy(x)
    results = copy.deepcopy(base)

    ## Directional Masking ##
    #Top
    if direction == 'top':
        #Set masking schedule and iterator
        if type(t) != int:
            j = int(np.round(len(rankings)*t))
            i = len(rankings) - j
            k = len(rankings)
        else: 
            j = t
            i = len(rankings) - t
            k = len(rankings)

        #Impute and Predict
        while k >= j:
            #Impute
            X_train, x_test = impute(i,k, rankings, X_train, x_test)
            #Predict
            results =  np.hstack((results, metrics(clf, x_test, y)))
            
            #Move iterator forward
            i -= j
            k -= j

        return results 
        
    #Bottom
    elif direction == 'bottom':   
        #Set masking schedule
        if type(t) != int:
            j = int(np.round(len(rankings)*t))
            i = 0
            k = j
        else:
            j = t
            i = 0
            k = j

        #Impute and Predict
        while k <= len(rankings):
            #Impute
            X_train, x_test = impute(i,k, rankings, X_train, x_test)
            #Predict
            results =  np.hstack((results, metrics(clf, x_test, y)))
            
            #Move iterator forward
            i += j
            k += j
        return results
    
    #Random
    elif direction == 'random':
        np.random.seed(42)
        random_choices = np.random.permutation(rankings)
        #Set masking schedule
        if type(t) != int:
            j = int(np.round(len(rankings)*t))
            i = 0
            k = j
        else:
            j = t
            i = 0
            k = j

        #Impute and Predict
        while k <= len(rankings):
            #Impute
            X_train, x_test = impute(i,k, random_choices, X_train, x_test)
            #Predict
            results =  np.hstack((results, metrics(clf, x_test, y)))
            
            #Move iterator forward
            i += j
            k += j
        return results
   

    


    
## ROAD ##
# The main function of the library.
#Wraps all other functions in a nice pipeline which is easy to use.
#Accepts any scikit-learn model. It was built and tested using a binary target.

# Arguments:
# model : the model to be re-trained
# t: percentage of features to be removed in each iteration
# X: Training data as pandas dataframe
# Y: Target values for training (This was build using a binary target)
# x:  Test data as pandas dataframe
# y: Target values for testing
# explainer: any explainer which built with the shap api
# repeats: how many times to explain and do the whole retraining

#outputs accuracy, balanced_accuracy, f1_score, and ranks for each iteration.  
def road(X, Y, x, y, model, explainer = None, t = 0.10, repeats = 2, shap_values = None):
    
    #Initialize variables
    base = metrics(model, x, y)
    if explainer != None:
        values = explain(model, X, x, explainer)
        ranks = ranker(values)
       
    elif shap_values != None:
        values = shap_values
        ranks = ranker(values)
    else:
        print('Must supply either an explainer or shap_values')
        return
    
    top = mask(t, ranks, X, Y, x, y, model, base, direction='top')
   
    bottom = mask(t, ranks, X, Y, x, y, model, base, direction='bottom')
  
    random = mask(t, ranks, X, Y, x, y, model, base, direction='random')

    

    #Set progress bar
    #Repeat x times
    for i in range(repeats-1):
                #Initialize
        if explainer != None:
            iter_values = explain(model, X, x, explainer)
            ranks = ranker(iter_values)
            values = np.dstack((values, iter_values))
        elif shap_values != None:
            ranks = ranker(values)
        else:
            print('Must supply either an explainer or shap_values')
            return
        top = np.dstack((top, mask(t, ranks, X, Y, x, y, model, base, direction='top')))
        bottom = np.dstack((bottom, mask(t, ranks, X, Y, x, y, model, base, direction='bottom')))
        random = np.dstack((random, mask(t, ranks, X, Y, x, y, model, base, direction='random')))
    return [top, bottom, random, values]