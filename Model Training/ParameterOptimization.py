from skopt.space import Real, Categorical, Integer
import sklearn
import sklearn.naive_bayes as NB
from xgboost import XGBClassifier

#Models
models = {
    'PassiveAgressive': sklearn.linear_model.PassiveAggressiveClassifier(max_iter = 10000),
    'SGDClassifier': sklearn.linear_model.SGDClassifier(early_stopping = True),
    'RandomForest': sklearn.ensemble.RandomForestClassifier(),
    'Perceptron': sklearn.linear_model.Perceptron(),
    'RidgeClassifier': sklearn.linear_model.RidgeClassifier(max_iter = 10000),
    'LogisticRegression': sklearn.linear_model.LogisticRegression(max_iter = 10000),
    'DecisionTree': sklearn.tree.DecisionTreeClassifier(),
    'XGBoost': XGBClassifier( use_label_encoder = False, eval_metric = 'logloss'),
    'MultinomialNB': NB.MultinomialNB(),
    'GaussianNB': NB.GaussianNB(),
    'SVC': sklearn.svm.SVC(max_iter = 10000)
}


#Hyperparameters to Search
search_space = {
    'PassiveAgressive': {
        'C': Real(0.1,1),
        'tol': Real(0.1,1)
    },
    'SGDClassifier': {
        'loss': Categorical(['hinge', 'log_loss', 'modified_huber', 'squared_hinge',
        'perceptron']),
        'penalty': Categorical(['l2', 'l1', 'elasticnet']),
        'alpha': Real(0.1,1),
        'learning_rate': Categorical(['constant', 'optimal',
        'invscaling', 'adaptive']),
        'eta0': Real(0.1,1),
        'power_t': Real(0,0.5)
    },
    'RandomForest': {
        'bootstrap': Categorical([True, False]),
        'ccp_alpha': Real(0.1, 1),
        'class_weight': Categorical(['balanced', 'balanced_subsample']),
        'criterion': Categorical(['gini', 'entropy']),
        'max_depth': Integer(6, 20),
        'max_features': Categorical(['sqrt','log2']),
        'min_samples_leaf': Integer(2,10),
        'min_samples_split': Integer(2,10),
        'n_estimators': Integer(100,500)
    },
    'Perceptron' : {
        'penalty': Categorical(['l2', 'l1', 'elasticnet', None]),
        'alpha' : Real(0.1,1),
        'l1_ratio': Real(0,1),
        'fit_intercept': Categorical([True, False]),
        'max_iter': Integer(500, 10000),
        'tol': Real(0,1),
        'shuffle': Categorical([True, False]),
        'early_stopping': Categorical([True, False])
    },
    'RidgeClassifier' : {
        'alpha': Real(0.1,1),
        'fit_intercept': Categorical([True, False]),
        'tol': Real(0,1),
        'solver': Categorical(['svd', 'cholesky', 
        'lsqr', 'sparse_cg', 'sag', 'saga'])
    },
    'LogisticRegression': {
        'fit_intercept': Categorical([True, False]),
        'solver': Categorical(['newton-cg', 'liblinear', 'lbfgs', 'sag', 'saga'])
    },
    'DecisionTree': {
        'criterion' : Categorical(['gini', 'entropy']),
        'splitter': Categorical(['best', 'random']),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1,10),
        'min_weight_fraction_leaf': Real(0,0.5),
        'max_features': Categorical(['sqrt', 'log2', None]),
        'min_impurity_decrease': Real(0,1)
    },
    'XGBoost' : {
  
        'n_estimators': Integer(1, 100)
    },
    # 'XGBoost': {
    #     'n_estimators': Integer(100, 500),
    #     'learning_rate': Real(0,1),
    #     'booster': Categorical(['gbtree', 'gblinear', 'dart']),
    #     'gamma': Real(0,1),
    #     'min_child_weight': Real(0,1),
    #     'subsample': Real(0,1),
    #     'reg_alpha': Real(0,1),
    #     'reg_lambda': Real(0,1),
    #     'base_score': Real(0,1),
    #     'num_parallel_tree': Integer(1,10)

    # },
    'MultinomialNB': {
        'alpha': Real(0.1,2)
    },
    'GaussianNB': {
        'var_smoothing': Real(1e-9, 1e+9, prior='log-uniform')
    },
    'SVC': {
         'C': Real(1e-6, 1e+6, prior='log-uniform'),
         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
         'degree': Integer(1,8),
         'kernel': Categorical(['linear', 'poly', 'rbf'])
     }
}


