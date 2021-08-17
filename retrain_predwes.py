import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.bayes_log_reg import bayes_log_reg
from models.functions import fill_results_array
import os

if __name__=="__main__":
    #line below can be used to generate new random data when needed
    # generated_data = generate_data(1500, 0.3, 1000,False)
    print("Loading random data and one-hot encoding HPO terms")
    print("Since random data is generated, do not use this model - this is purely to demonstrate how this pipeline works!")
    
    generated_data = pd.read_feather(os.path.join(os.path.realpath('.'), "data" , "random_generated_hpo_terms.ftr"))
    
    y = generated_data.WES_result.to_numpy()
    hpo_expanded = pd.get_dummies(pd.DataFrame(generated_data.iloc[:,0].tolist()),prefix_sep ='',prefix='').sum(level=0, axis = 1)
    
    X = hpo_expanded.to_numpy()    
    
    skf_outer = StratifiedKFold(n_splits=10, shuffle=False)
    skf_outer.get_n_splits(X, y)
    
    outer_fold_count = 0
    
    df_all_alg_scores, df_all_brier_scores = pd.DataFrame(), pd.DataFrame()
    
    for train_index_outer, test_index_outer in skf_outer.split(X, y):
            alg_scores_indexer = 0
            outer_fold_count = outer_fold_count + 1
            X_train_val, X_real_test = X[train_index_outer,:], X[test_index_outer,:]
            y_train_val, y_real_test = y[train_index_outer], y[test_index_outer]
            
            alg_scores = np.empty((22, 3 + len(X_train_val)),dtype=object)
            brier_scores = np.empty((22, 3 + len(X_train_val)),dtype=object)
                            
            alg_scores[:] = np.nan
            current_fold = 0
            
            skf = StratifiedKFold(n_splits=10, shuffle=False)
            skf.get_n_splits(X_train_val, y_train_val)
            
            for train_index, test_index in skf.split(X_train_val, y_train_val):
                current_fold = current_fold + 1
                X_train, X_val = X_train_val[train_index,:], X_train_val[test_index,:]
                y_train, y_val = y_train_val[train_index], y_train_val[test_index]
                
                train_means, val_means = bayes_log_reg(X_train, y_train, X_val, 'Log_reg_Finnish_Horseshoe') 
                alg_scores = fill_results_array(alg_scores, 'Log_reg_Finnish_Horseshoe', str(outer_fold_count) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, False, y_train, y_val)
                brier_scores = fill_results_array(brier_scores, 'Log_reg_Finnish_Horseshoe', str(outer_fold_count) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, True, y_train, y_val)
            
                alg_scores_indexer += 2
                
            current_fold = 11
            X_train = X_train_val
            y_train = y_train_val
            X_val = X_real_test
            y_val = y_real_test
            
            train_means, val_means = bayes_log_reg(X_train, y_train, X_val, 'Log_reg_Finnish_Horseshoe') 
            alg_scores = fill_results_array(alg_scores, 'Log_reg_Finnish_Horseshoe', str(outer_fold_count) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, False, y_train, y_val)
            brier_scores = fill_results_array(brier_scores, 'Log_reg_Finnish_Horseshoe', str(outer_fold_count) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, True, y_train, y_val)

            df_all_alg_scores = df_all_alg_scores.append(pd.DataFrame(alg_scores))
            df_all_brier_scores = df_all_brier_scores.append(pd.DataFrame(brier_scores))

    mean_test_brier_all_folds = df_all_brier_scores[df_all_brier_scores.iloc[:,2] == 'test'].iloc[:,3:].mean(axis=1)
    
    print("Mean brier score of the test folds using nested cross validation is " + str(mean_test_brier_all_folds.mean()))
    
    #Now that we have assessed the performance of the model using nested cross validation, lets train our final model
    #Here, X_val does not make sense, since we are not evaluating the performance anymore, but because the function is there, lets use it
    
    train_means, val_means = bayes_log_reg(X, y, X, 'Log_reg_Finnish_Horseshoe', export_model=True) 
