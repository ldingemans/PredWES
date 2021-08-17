import zlib
import pandas as pd
import numpy as np
from random import sample
import ast
import pickle
from functools import partial
from sklearn.metrics import brier_score_loss
import sys
from hyperopt import STATUS_OK, fmin, tpe, Trials, hp
from sklearn.model_selection import StratifiedKFold
from models.functions import fill_results_array, no_progress_loss_moving_average

def predict_values(X_val, y_train, class_imbalance_correction, training_data, chosen_combis):
    """
    Make predictions for new data

    Parameters
    ----------
    X_val : numpy array
        Unseen data to make predictions for
    y_train: numpy array
        WES results of training data, used to calculate portion of individuals with a positive WES to correct for class imbalance
    class_imbalance_correction: bool
        Whether to correct for the class imbalance
    training_data: numpy array
        The recorded columns of the training data
    chosen_combis: numpy array
        The random chosen columns to select
        
    Returns
    -------
    r_0_uncorr: float
        The uncorrected score for class 0    
    r_1_uncorr: float
        The uncorrected score for class 1   
    r_0: float
        The corrected score for class 0    
    r_1: float
        The corrected score for class 1  
    y_pred: int
        The predicted class
    """
        
    r_0 = [0.0] * X_val.shape[0]
    r_1 = [0.0] * X_val.shape[0]
    r_0_uncorr = [0.0] * X_val.shape[0]
    r_1_uncorr = [0.0] * X_val.shape[0]
    y_pred = [0.0] * X_val.shape[0]  
    
    val_data_np = np.zeros((len(X_val) * len(chosen_combis), 3),dtype=object)
    
    for i in range(len(chosen_combis)):
        combination_dec = ast.literal_eval(zlib.decompress(chosen_combis[i]).decode())
        new_matrix = X_val.take(combination_dec,axis=1)
        for y in range(len(new_matrix)):
            val_data_np[len(new_matrix) * i + y,:2] = chosen_combis[i], ''.join(np.array(compress_string_pattern(new_matrix[y,:]),dtype=str))
            val_data_np[len(new_matrix) * i + y,2] = y
            
    val_data = pd.DataFrame(val_data_np)
    val_data.columns = ["selected_columns", "sign_pattern", "val_index"]
    val_data['selected_columns'] = val_data['selected_columns'].astype("category")
    val_data['sign_pattern'] = val_data['sign_pattern'].astype("category") 
   
    r_0 = 0.0
    r_1 = 0.0
    r_0_uncorr = 0.0
    r_1_uncorr = 0.0
    y_pred = 0.0 
    val_data = val_data.merge(training_data, how='left', on=['selected_columns', 'sign_pattern'])
    val_data = val_data.loc[:,["val_index", "r_0", "r_1"]]
    val_data = val_data.groupby(['val_index'], observed=True, as_index=False).agg('sum').reset_index(drop=True)
    val_data = val_data.sort_values(by='val_index')
    r_0 = val_data.r_0
    r_1 = val_data.r_1
    r_0_uncorr = r_0
    r_1_uncorr = r_1
    if class_imbalance_correction == True:
        r_0 = r_0 * 1/(1-np.mean(y_train))
        r_1 = r_1 * 1/np.mean(y_train)
    y_pred = (r_0.to_numpy() < r_1.to_numpy()).astype(int)
    
    return r_0_uncorr, r_1_uncorr, r_0, r_1, y_pred

def training(l, m, X_train, y_train):
    """
    Make predictions for new data

    Parameters
    ----------
    l : int
        number of levels
    m : int
        number of iterations to do
    X_train: numpy array
        The training data
    y_train: numpy array
        The correspondeing labels (in our case the WES results) of the training data
        
    Returns
    -------
    training_data: numpy array
        The recorded columns of the training data
    chosen_combis: numpy array
        The random chosen columns to select
    """
    chosen_combis = np.empty(0)  
        
    for y in range(l):
        chosen_combis_temp = np.empty(m * 10, dtype=np.object)  
        for i in range(m * 10):
            chosen_combis_temp[i] = str(sorted(sample(range(0, X_train.shape[1]), (y+1))))
        _, idx = np.unique(chosen_combis_temp, return_index=True)
        chosen_combis_temp = chosen_combis_temp[np.sort(idx)]
        chosen_combis_temp = chosen_combis_temp[:m]
        chosen_combis = np.append(chosen_combis, chosen_combis_temp)
    
    chosen_combis_comp = np.empty(len(chosen_combis), dtype=object)

    for index, combi in enumerate(chosen_combis):
        chosen_combis_comp[index] = zlib.compress(combi.encode())
    
    chosen_combis = chosen_combis_comp

    training_data_np = np.zeros((len(X_train) * len(chosen_combis), 4),dtype=object)
    
    for i in range(len(chosen_combis)):
        combination_dec = ast.literal_eval(zlib.decompress(chosen_combis[i]).decode())
        new_matrix = X_train.take(combination_dec,axis=1)
        training_data_np[i*len(new_matrix):i*len(new_matrix)+len(new_matrix),2] = np.array(y_train == 0,dtype='int8')
        training_data_np[i*len(new_matrix):i*len(new_matrix)+len(new_matrix),3]  = np.array(y_train == 1,dtype='int8')
        for y in range(len(new_matrix)):
            training_data_np[len(new_matrix) * i + y,:2] = chosen_combis[i], ''.join(np.array(compress_string_pattern(new_matrix[y,:]),dtype=str))

    training_data = pd.DataFrame(training_data_np)  
    training_data.columns = ['selected_columns', 'sign_pattern', 'class_0', 'class_1']
    training_data['selected_columns'] = training_data['selected_columns'].astype("category")
    training_data['sign_pattern'] = training_data['sign_pattern'].astype("category")
        
    training_data = training_data.groupby(['selected_columns','sign_pattern'], observed=True, as_index=False).agg('sum').reset_index(drop=True)
      
    training_data['r_0'] = '' 
    training_data['r_1'] = '' 
    
    training_data['all_points']= training_data.class_0 + training_data.class_1
    training_data.r_0 = (training_data.class_0/training_data.all_points) * (abs(training_data.class_0- training_data.class_1)/training_data.all_points)
    training_data.r_1 = (training_data.class_1/training_data.all_points) * (abs(training_data.class_1-training_data.class_0)/training_data.all_points)

    return training_data, chosen_combis

def objective(params): 
    """
    The objective function for hyperopt to optimize.

    Parameters
    ----------
    params: dict
        The current parameters to evaluate
        
    Returns
    -------
    trials: trials instance
        Results of evaluation with current parameters
    """
    l = int(params['l'])
    m = int(params['m'])
    if 'k' in params:
        k = int(params['k'])
        if k == 1 : #k cannot be one
            k = 2
        if l > k: # l cannot be larger than all number of columns
            l = k - 1

    X_train_val = params['X_train_val']
    y_train_val = params['y_train_val']
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X_train_val, y_train_val)
    
    val_brier = []
    
    for train_index, test_index in skf.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index,:], X_train_val[test_index,:]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        if 'k' in params:
            X_train, X_val = select_k_best(X_train, X_val,y_train,k)
        
        training_data, chosen_combis = training(l,m, X_train, y_train) 
        training_data = training_data.loc[:, ["selected_columns", "sign_pattern", "r_0", "r_1"]]
        r_0_uncorr_val, r_1_uncorr_val, r_0_val, r_1_val, y_pred_val = predict_values(X_val, y_train, True, training_data, chosen_combis)
        r_total_val = np.array(r_0_val) + np.array(r_1_val)
        score_brier = brier_score_loss(y_val,np.array(r_1_val)/r_total_val)
        val_brier.append(score_brier)
    
    loss = np.mean(val_brier)
    return {'loss': loss, 'brier_loss': np.mean(val_brier), 'l': l, 'm':m, 'number_features':X_train.shape[1], 'status': STATUS_OK}
        
def select_k_best(X_train, X_val,y_train, k):
    """
    Select the K best features, based on the chi-square stat.

    Parameters
    ----------
    X_train: numpy array
        The training data
    X_val: numpy array
        Validation/testing data
    y_train: numpy array
        The correspondeing labels (in our case the WES results) of the training data
    k: int
        Number of features to select    
    
    Returns
    -------
    X_train: numpy array
        The training data, now with the K selected best features
    X_val: numpy array
        The validation/testing data, now with the k selected best features
    """
    from sklearn.feature_selection import SelectKBest, chi2
    clf = SelectKBest(chi2, k=k).fit(X_train, y_train)
    X_train = X_train[:,clf.get_support()]
    X_val= X_val[:,clf.get_support()]
    return X_train, X_val

def compress_string_pattern(pattern):
    """
    Compress the string pattern to save memory

    Parameters
    ----------
    Pattern: numpy array
        The pattern to compress
    
    Returns
    -------
    Pattern: numpy array
        The compressed pattern
    """
    import numpy as np
    return np.append(np.packbits(pattern,axis=-1,bitorder="little"),len(pattern))

def uncompress_string_pattern(pattern):
    """
    Uncompress the compressed string pattern
    
    Parameters
    ----------
    Pattern: numpy array
        The pattern to uncompress
    
    Returns
    -------
    Uncompressed: numpy array
        The uncompressed pattern
    """
    uncompressed = []
    length = pattern[-1]
    pattern = pattern[:-1]
    pattern = np.array(pattern, dtype=np.uint8)
    for bit in pattern:
        uncompressed.extend(np.unpackbits(bit, bitorder= 'little'))
    return uncompressed[:length]

def calc_brier(X_train, y_train, X_val, y_val, l, m, k):
    """
    Evaluate the performance

    Parameters
    ----------
    X_train: numpy array
        The training data
    y_train: numpy array
        The correspondeing labels (in our case the WES results) of the training data
    X_val: numpy array
        The validation/testing data
    y_val: numpy array
        The correspondeing labels (in our case the WES results) of the validation/testing data
    l: int
        Number of levels
    m: int
        Number of iterations to do
    k: int
        Number of features to select
        
    Returns
    -------
    score: float
        The Brier score after evaluating the model with the current parameters and data 
    preds_train: numpy array
        The predictions for the training data
    preds_val: numpy array
        The predictions for the validation/testing data
    """
    if k >0:
        if l > k:
            l = k - 1
        X_train, X_val = select_k_best(X_train, X_val,y_train,k)
    # print("\nStarting training phase")
    # print(l,m)
    training_data, chosen_combis = training(l,m, X_train, y_train) 
    training_data = training_data.loc[:, ["selected_columns", "sign_pattern", "r_0", "r_1"]]
    
    r_0_uncorr_train, r_1_uncorr_train, r_0_train, r_1_train, y_pred_train = predict_values(X_train, y_train, True, training_data, chosen_combis)
    r_total_train = np.array(r_0_train) + np.array(r_1_train)
    preds_train = np.array(r_1_train)/r_total_train
    # print("Calculating validation score")
    r_0_uncorr_val, r_1_uncorr_val, r_0_val, r_1_val, y_pred_val = predict_values(X_val, y_train, True, training_data, chosen_combis)
    r_total_val = np.array(r_0_val) + np.array(r_1_val)
    preds_val = np.array(r_1_val)/r_total_val
    score = brier_score_loss(y_val,np.array(r_1_val)/r_total_val)
    # print("Current Brier :" + str(score))
    return score, preds_train, preds_val

def run_needell(X_train_val, y_train_val, X_real_test, y_real_test, IS_FEATURE_SEL, MAX_EVALS, OUTER_FOLD_COUNT):
    """
    Run the binary classification algorithm as proposed by Needell et al (https://www.jmlr.org/papers/volume19/17-383/17-383.pdf)

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
    IS_FEATURE_SEL: bool
        Whether to select features first
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
    
    if IS_FEATURE_SEL == True:
        name_string = 'Needell_feature_sel'
        space = {
            'k': hp.quniform('k', 1, 200, 1),
            'l': hp.quniform('l', 1, 200, 1),
            'm': hp.quniform('m', 1, 50, 1),
        }
    else:
        name_string = 'Needell'
        space = {
            'l': hp.quniform('l', 1, 400, 1),
            'm': hp.quniform('m', 1, 50, 1),
        }
        
        
    space['X_train_val'] = X_train_val
    space['y_train_val'] = y_train_val
    
    trials = Trials()
    best = fmin(objective,
        space=space,
        algo=partial(tpe.suggest, n_startup_jobs=20),
        max_evals=MAX_EVALS,
        trials=trials,
        catch_eval_exceptions=True,
        early_stop_fn=no_progress_loss_moving_average(iteration_stop_count=20, percent_increase=0.0))
    
    if IS_FEATURE_SEL == True:
        param_k = int(best['k'])
    else:
        param_k = 0
    param_l = int(best['l'])
    param_m = int(best['m'])
    
    skf_inner = StratifiedKFold(n_splits=10, shuffle=False)
    skf_inner.get_n_splits(X_train_val, y_train_val)
    
    current_inner_fold = 0
    
    for train_index_inner, test_index_inner in skf_inner.split(X_train_val, y_train_val):
            current_inner_fold += 1
            X_train, X_val = X_train_val[train_index_inner,:], X_train_val[test_index_inner,:]
            y_train, y_val = y_train_val[train_index_inner], y_train_val[test_index_inner]
            
            score, train_means, val_means = calc_brier(X_train, y_train, X_val, y_val, param_l, param_m, param_k)
            results = fill_results_array(results, name_string, str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, False, y_train, y_val)
            brier_scores = fill_results_array(brier_scores, name_string, str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, True, y_train, y_val)
            result_indexer += 2
    
    current_inner_fold = 11
    score, train_means, val_means = calc_brier(X_train_val, y_train_val, X_real_test, y_real_test, param_l, param_m, param_k)
    results = fill_results_array(results, name_string, str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, False, y_train_val, y_real_test)
    brier_scores = fill_results_array(brier_scores, name_string, str(OUTER_FOLD_COUNT) + '_' + str(current_inner_fold), train_means, val_means, result_indexer, True, y_train_val, y_real_test)

    return results, brier_scores, trials
