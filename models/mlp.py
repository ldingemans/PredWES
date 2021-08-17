import numpy as np
from hyperopt import STATUS_OK, fmin, tpe, Trials, hp
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from models.functions import fill_results_array, no_progress_loss_moving_average

def hyperopt_train_test(params):
    """
    The objective function for hyperopt to optimize.

    Parameters
    ----------
    params: dict
        The current parameters to evaluate
        
    Returns
    -------
        The Brier score after cross validation    
    """
    X_train_val = params['X_train_val']
    y_train_val = params['y_train_val']
    del params['X_train_val']
    del params['y_train_val']
    X_ = X_train_val[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = Normalizer().fit_transform(X_)
    del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = StandardScaler().fit_transform(X_)
    del params['scale']
    params['hidden_layer_sizes'] = (int(params['network_depth']), int(params['network_width']))
    del params['network_depth']
    del params['network_width']
    clf = MLPClassifier(**params)
    skf = StratifiedKFold(n_splits=10)
    return cross_val_score(clf, X_, y_train_val, cv=skf, scoring='neg_brier_score').mean()

def f(params):
    """
    Wrapper for he objective function for hyperopt to optimize.

    Parameters
    ----------
    params: dict
        The current parameters to evaluate
        
    Returns
    -------
    trials: trials instance
        Results of evaluation with current parameters
    """
    brier_loss = hyperopt_train_test(params)
    # pickle.dump(trials, open(TRIALS_FILE, "wb"))
    return {'loss': -brier_loss, 'status': STATUS_OK}

def run_mlp(X_train_val, y_train_val, X_real_test, y_real_test, MAX_EVALS, OUTER_FOLD_COUNT):
    """
    Run the hyperparameter tuning for the SVM and use the best parameters to get results
    Parameters
    ----------
    X_train_val: numpy array
        The training and validation data, not splitted (yet)
    y_train_val: numpy array
        The correspondeing labels (in our case the WES results) of the training/validation data
    X_real_test: numpy array
        The test data
    y_real_test: numpy array
        The correspondeing labels (in our case the WES results) of the test data
    MAX_EVALS: int
        Maximum evaluations for the hyperparameter optimisation
    OUTER_FOLD_COUNT: int
        The current outer fold
        
    Returns
    -------
    results: numpy array
        The predictions
    brier_scores: numpy array
        The Brier score after evaluating the model with the current parameters and data 
    """
    results = np.zeros((22,3 + len(X_train_val)), dtype=object)
    brier_scores = np.zeros((22, 3 + len(X_train_val)), dtype=object) 
    result_indexer = 0
    
    space4mlp = {
         'network_depth': hp.quniform('network_depth', 1, 100, 1),
         'network_width': hp.quniform('network_width', 1, 100, 1),
         'activation': 'relu',
         'solver': hp.choice('solver', ['lbfgs', 'adam']),
         'alpha': hp.uniform('alpha', 0, 20),
         'max_iter': 300000,
         'early_stopping': True,
         'max_fun': 300000,
         'scale': hp.choice('scale', [0, 1]),
         'verbose': False,
         'normalize': hp.choice('normalize', [0, 1])
    }

    space4mlp['X_train_val'] = X_train_val
    space4mlp['y_train_val'] = y_train_val
       
    trials = Trials()
    best = fmin(f, space4mlp, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials, early_stop_fn=no_progress_loss_moving_average(iteration_stop_count=20, percent_increase=0.0))
    
    best = trials.best_trial['misc']['vals']
    normalize = best['normalize'][0]
    scale = best['scale'][0]
    del best['normalize']
    del best['scale']
    for k in best:
        best[k] = best[k][0]
    best['solver'] = ['lbfgs', 'adam'][best['solver']]
    best['hidden_layer_sizes'] = (int(best['network_depth']), int(best['network_width']))
    del best['network_depth']
    del best['network_width']
    skf_inner = StratifiedKFold(n_splits=10, shuffle=False)
    skf_inner.get_n_splits(X_train_val, y_train_val)
     
    current_inner_fold = 0
     
    for train_index_inner, test_index_inner in skf_inner.split(X_train_val, y_train_val):
             current_inner_fold += 1
             X_train, X_val = X_train_val[train_index_inner,:], X_train_val[test_index_inner,:]
             y_train, y_val = y_train_val[train_index_inner], y_train_val[test_index_inner]
             
             if scale == 1:
                 scaler = StandardScaler()
                 X_train = scaler.fit_transform(X_train)
                 X_val = scaler.transform(X_val)
                 
             if normalize == 1:
                 normalizer = Normalizer()
                 X_train = normalizer.fit_transform(X_train)
                 X_val = normalizer.transform(X_val)
             
             clf = MLPClassifier(**best)
             clf.fit(X_train, y_train)
             train_means = clf.predict_proba(X_train)[:,1]
             val_means = clf.predict_proba(X_val)[:,1]
             results = fill_results_array(results, 'MLP', str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, False, y_train, y_val)
             brier_scores = fill_results_array(brier_scores, 'MLP', str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, True, y_train, y_val)

             result_indexer += 2
 
    if scale == 1:
         scaler = StandardScaler()
         X_train_val = scaler.fit_transform(X_train_val)
         X_real_test = scaler.transform(X_real_test)
                 
    if normalize == 1:
        normalizer = Normalizer()
        X_train_val = normalizer.fit_transform(X_train_val)
        X_real_test = normalizer.transform(X_real_test)
             
    current_inner_fold = 11
    clf = MLPClassifier(**best)
    clf.fit(X_train_val, y_train_val)
    train_means = clf.predict_proba(X_train_val)[:,1]
    val_means = clf.predict_proba(X_real_test)[:,1]
    results = fill_results_array(results, 'MLP', str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, False, y_train_val, y_real_test)
    brier_scores = fill_results_array(brier_scores, 'MLP', str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, True, y_train_val, y_real_test)

    return results, brier_scores, trials