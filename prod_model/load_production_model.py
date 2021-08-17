import pymc3 as pm
import pandas as pd
import numpy as np
import pickle
import os
import sys
import math
import ast

def preproc(hpo_input):
    """
    Preprocess the HPO terms, so that the input is exactly correct for the model
    
    Parameters
    ----------
    hpo_input: list
        list of HPO terms
        
    Returns
    -------
    df : pandas DataFrame
        Preprocessed list of HPO terms
    """
    import pandas as pd
    list_cols = pd.read_csv(os.path.join(os.path.realpath('.'), "list_cols.txt"), header=None)
    list_cols = list_cols.iloc[:,0]
    
    df = pd.DataFrame(np.zeros((1,len(list_cols))))
    df.columns = list_cols
        
    for col in df.columns:
        if col in hpo_input:
            df.loc[0, col] = 1

    return df

def get_scores(hpo_input):
    """
    Calculate the scores using the PredWES model
    
    Parameters
    ----------
    hpo_input: list
        List of HPO terms
    
    Returns
    -------
    mean_score: float
        The prediction of PredWES
    """
    
    model_fpath = os.path.join(os.path.realpath('.'), "horseshoe_trained")
    
    with open(model_fpath, 'rb') as buff:
        data = pickle.load(buff)
    model = data['model']
    trace = data['trace']
    X_shared = data['X_shared']
        
    if type(hpo_input) == list:
        processed_input = preproc(hpo_input)
        processed_input = processed_input.append(processed_input)
    elif type(hpo_input) == pd.core.frame.DataFrame:
       hpo_input = hpo_input.reset_index(drop=True)
       processed_input = pd.DataFrame()
       for i in range(len(hpo_input)):
           processed_input = processed_input.append(preproc(hpo_input.loc[i,'hpo_all'].split(';')))
    
    X_shared.set_value(processed_input)
    
    with model:
        ppc = pm.sample_posterior_predictive(trace, samples=10000, model=model)
        
    mean_score = ppc['y_pred'].mean(axis=0).mean()
    return mean_score

if __name__ == '__main__':
    #input command line arguments as "HP:0003808,HP:0001252,HP:0000252"
    try:
        hpo_input = sys.argv[1].split(',')
    except:
        #no command line arguments given   
        print("No command line arguments given, using hard coded HPO terms as input.")
        hpo_input = "HP:0000252,HP:0003808".split(',')
    print("Running PredWES on HPO terms " + str(hpo_input))
    scores = get_scores(hpo_input)
    print(scores)
