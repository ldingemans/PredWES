import obonet
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from models.needell import run_needell
from models.svm import run_svm
from models.mlp import run_mlp
from models.bayes_log_reg import run_logreg
from models.devries import run_devries
from sklearn.metrics import brier_score_loss
from models.functions import generate_table_1_baseline, generate_table_2_results, generate_fig_1_waffle, generate_fig2_calibration_curve, generate_fig3_sig_features, generate_supp_table2_p_values, generate_supp_table3_hyperparameters, generate_supp_fig1_plot_per_checklist_score, generate_supp_fig2_hpo_number_dist_graph

def get_base_graph():
    """
    Get the update graph/list of HPO terms
        
    Returns
    -------
    graph: graph
        a graph representing the HPO graph
    id_to_name: dict
        a dictionary with all the HPO terms and names
    """

    url = 'http://purl.obolibrary.org/obo/hp.obo'
    graph = obonet.read_obo(url)
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    return graph, id_to_name

def generate_data(n_patients, portion_positive_WES, n_hpo_terms, easy_recognizable=False):
    """
    Generate random patient data

    Parameters
    ----------
    n_patients : int
        Number of individuals to generate data for
    portion_positive_WES: float
        Portion of individuals with a positive WES
    n_hpo_terms: int
        Max number of possible HPO terms
    easy_recognizable: bool
        If True, add some specific terms only to the positive group, making the regression (very) easy
        
    Returns
    -------
    generated_data : pandas DataFrame
        A dataframe with all generated HPO terms and randomly generated label (WES_result)
    """
    import random
    print("Generating random data for " + str(n_patients) + " individuals.")
    print("Since random data is generated, do not use this model - this is purely to demonstrate how this pipeline works!")
    
    hpo_graph, id_to_name = get_base_graph()
    all_hpo_ids = list(id_to_name.keys())
    all_hpo_ids = random.sample(all_hpo_ids, n_hpo_terms)
    
    generated_data = pd.DataFrame()
    generated_data['hpo_terms'] = '' 
    generated_data['hpo_names'] = '' 
    generated_data['WES_result'] = np.random.binomial(1,portion_positive_WES,size=n_patients)
    
    for i in range(n_patients):
        number_hpo_terms = int(np.random.normal(16,8)) #pick random number for the total number of HPO terms
        if number_hpo_terms < 1:
            number_hpo_terms = 1 #make sure number of HPO terms is at least 1
        if number_hpo_terms > n_hpo_terms:
            number_hpo_terms = n_hpo_terms - 1 
            
        generated_hpo_terms = random.sample(all_hpo_ids, number_hpo_terms)
        if (easy_recognizable == True) and generated_data.loc[i,'WES_result'] == 1:
            generated_hpo_terms.append('HP:0001250')
            generated_hpo_terms.append('HP:0030084')
            generated_hpo_terms.append('HP:0002360')
        generated_data.at[i,'hpo_terms'] = generated_hpo_terms
        
        generated_hpo_names = []
        for hpo_id in generated_hpo_terms:
            generated_hpo_names.append(id_to_name[hpo_id])
        generated_data.at[i,'hpo_names'] = generated_hpo_names  
            
    return generated_data    
    
def run_all_models(generated_data, hpo_expanded):
    """
    Run all the models as described in the paper

    Parameters
    ----------
    generated_data: numpy array
        The data with HPO terms for which the table is to be created
    hpo_expanded: numpy array
        The data with HPO terms for which the table is to be created with the HPO terms as seperate columns
        
    Returns
    -------
    df_all_alg_scores: pandas dataframe
        The dataframe with all the predictions
    df_all_brier_scores: pandas dataframe
        The dataframe with all the Brier scores
    brier_devries: pandas series
        The Brier scores of the De Vries score
    trials_svm: trials     
        The Hyperopt trials during hyperparameter optimization of the SVM
    trials_mlp: trials     
        The Hyperopt trials during hyperparameter optimization of the MLP
    trials_needell: trials     
        The Hyperopt trials during hyperparameter optimization of the Needell algorithm
    trials_needell_feature_sel: trials     
        The Hyperopt trials during hyperparameter optimization of the Needell algorithm with feature selection
    """
    X = hpo_expanded.to_numpy()    
    y = generated_data.WES_result.to_numpy()
    
    skf_outer = StratifiedKFold(n_splits=10, shuffle=False)
    skf_outer.get_n_splits(X, y)
    
    outer_fold_count = 0
    
    df_all_alg_scores, df_all_brier_scores = pd.DataFrame(), pd.DataFrame()
    
    for train_index_outer, test_index_outer in skf_outer.split(X, y):
            outer_fold_count = outer_fold_count + 1
            X_train_val, X_real_test = X[train_index_outer,:], X[test_index_outer,:]
            y_train_val, y_real_test = y[train_index_outer], y[test_index_outer]
            
            #use Hyperopt to find the best hyperparameters for the SVM and get the predictions and Brier scores for the SVM
            svm_scores, svm_brier, trials_svm = run_svm(X_train_val, y_train_val, X_real_test, y_real_test, MAX_EVALS=500, OUTER_FOLD_COUNT= outer_fold_count)
            
            #same procedure, now for MLP
            mlp_scores, mlp_brier, trials_mlp = run_mlp(X_train_val, y_train_val, X_real_test, y_real_test, MAX_EVALS=500, OUTER_FOLD_COUNT= outer_fold_count)
            
            #run the Needell algorithm, both with and without feature selection
            needell_scores, needell_brier, trials_needell = run_needell(X_train_val, y_train_val, X_real_test, y_real_test, IS_FEATURE_SEL=False, MAX_EVALS=100, OUTER_FOLD_COUNT= outer_fold_count)
            needell_scores_feature_sel, needell_brier_feature_sel, trials_needell_feature_sel = run_needell(X_train_val, y_train_val, X_real_test, y_real_test,IS_FEATURE_SEL=True, MAX_EVALS=100, OUTER_FOLD_COUNT= outer_fold_count)          
            
            bayes_scores, bayes_brier = run_logreg(X_train_val, y_train_val, X_real_test, y_real_test, OUTER_FOLD_COUNT= outer_fold_count)
         
            all_alg_scores = np.concatenate((svm_scores, mlp_scores, needell_scores, needell_scores_feature_sel, bayes_scores),axis=0)
            all_brier_scores = np.concatenate((svm_brier, mlp_brier, needell_brier, needell_brier_feature_sel, bayes_brier),axis=0)

            df_all_alg_scores = df_all_alg_scores.append(pd.DataFrame(all_alg_scores))
            df_all_brier_scores = df_all_brier_scores.append(pd.DataFrame(all_brier_scores))
            
    #Finally calculate the De Vries score, not in the nested cross-validation, because training is not needed
    devries_score = run_devries(pd.concat([hpo_expanded, generated_data], axis=1)).loc[:,'ChecklistScore']
    devries_score = devries_score/10
    brier_devries = []
    for i, score in enumerate(devries_score):
        brier_devries.append(brier_score_loss([y[i]], [score]))
    brier_devries = pd.Series(brier_devries)
    return df_all_alg_scores, df_all_brier_scores, brier_devries, trials_svm, trials_mlp, trials_needell, trials_needell_feature_sel

if __name__=="__main__":
    #line below can be used to generate new random data when needed
    # generated_data = generate_data(1500, 0.3, 1000,False)
    print("Loading random data and one-hot encoding HPO terms")
    print("Since random data is generated, do not use this model - this is purely to demonstrate how this pipeline works!")
    
    generated_data = pd.read_feather(os.path.join(os.path.join(os.path.realpath('.'), "data" , "random_generated_hpo_terms.ftr")))
    hpo_expanded = pd.get_dummies(pd.DataFrame(generated_data.iloc[:,0].tolist()),prefix_sep ='',prefix='').sum(level=0, axis = 1)
    
    #first, lets run all models to get the results
    df_all_alg_scores, df_all_brier_scores, brier_devries, trials_svm, trials_mlp, trials_needell, trials_needell_feature_sel = run_all_models(generated_data, hpo_expanded)
    
    #now we can generate the tables and figures as in the paper
    generate_table_1_baseline(generated_data, hpo_expanded)
    generate_table_2_results(df_all_brier_scores, np.mean(brier_devries))
    
    df_fh = df_all_alg_scores[df_all_alg_scores.iloc[:,0] == 'Log_reg_Finnish_Horseshoe']
    df_fh_test = df_fh[df_fh.iloc[:,2] == 'test']
    fh_predictions = df_fh_test.to_numpy()[:,3:].flatten()
    fh_predictions = np.array(fh_predictions, dtype=float)
    fh_predictions =  fh_predictions[~np.isnan(fh_predictions)]
    
    generate_fig_1_waffle(generated_data.WES_result.to_numpy(), fh_predictions)
    generate_fig2_calibration_curve(generated_data.WES_result.to_numpy(), fh_predictions)
    generate_fig3_sig_features(hpo_expanded, generated_data.WES_result.to_numpy())
    
    generate_supp_table2_p_values(df_all_brier_scores, len(generated_data), brier_devries)
    generate_supp_table3_hyperparameters([trials_svm, trials_mlp, trials_needell, trials_needell_feature_sel])
    
    generate_supp_fig1_plot_per_checklist_score(generated_data, hpo_expanded)
    generate_supp_fig2_hpo_number_dist_graph(generated_data, fh_predictions)
