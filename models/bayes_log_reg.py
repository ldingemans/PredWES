import pymc3 as pm
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import pickle
from models.functions import fill_results_array

def beta_reg_horseshoe(v, s, prior_relevant_features, shape, n, sigma):
    """
    Finnish horseshoe prior as introduced by Piironen & Vehtari

    Parameters
    ----------
    v: regularizing student-t df
    s: regularizing student-t sd
    prior_relevant_features: expected number of relevant features (must be < total N)
    shape: shape of X
    n: n
    sigma: sigma
        
    Returns
    -------
        The prior for the regression coefficients
    """
    
    half_v = v/2
    
    tau0 = (prior_relevant_features/(shape-prior_relevant_features)) * (sigma/np.math.sqrt(n))
    tau_t = pm.HalfCauchy("tauT", beta = 1)
    tau = tau0*tau_t

    c2_t = pm.InverseGamma("c2_r", half_v, half_v)
    c2 = np.power(s,2) * c2_t

    lambda_m = pm.HalfCauchy("lambdaM", beta = 1, shape = shape)
    lambda_t = (pm.math.sqrt(c2)*lambda_m) / pm.math.sqrt(c2 + pm.math.sqr(tau) * pm.math.sqr(lambda_m))

    beta_t = pm.Normal("betaT", mu=0, sd = 1, shape= shape)
    beta_hs = pm.Deterministic('beta_hs', tau * lambda_t * beta_t)
    return beta_hs

def beta_spike_slab(shape, spike):
    """
    Spike and slab prior for the regression coefficients

    Parameters
    ----------
    shape: numpy array
        Shape of the data
    spike: int
        Standard deviation of the normal distribution for the spike
        
    Returns
    -------
        The spike and slab prior for the regression coefficients
    """
    inclusion_prop = pm.Beta('inclusion_prop', 2, 7)
    beta_spike = pm.Normal('beta_spike', 0, spike, shape=shape)
    
    tau2 = pm.HalfNormal('tau2', 2, shape=shape)
    beta_tilde = pm.Normal('beta_tilde', 0 ,1, shape=shape)
    beta_slab = pm.Deterministic('beta_slab', 0 + tau2 * beta_tilde)
    
    gamma = pm.Bernoulli('gamma', inclusion_prop, shape=shape)

    beta_spike_slab = pm.Deterministic('beta_spike_slab',(beta_spike * (1-gamma)) + ((beta_slab * gamma)))
    return beta_spike_slab

def beta_no_reg(mu, sd, shape):
    """
    Prior for the regression coefficients with regularisation

    Parameters
    ----------
    mu : int
        Mean of the normal distribution of the prior
    sd: int
        Standard deviation of the normal distribution of the prior
    shape: numpy array
        Shape of the data
        
    Returns
    -------
        The prior for the regression coefficients
    """
    beta = pm.Normal('beta', mu=mu, sd=sd, shape=shape)  
    return beta

def bayes_log_reg(X_train, y_train, X_val, reg_method, prior_vars_hs=100, export_model=False):
    """
    Building the Bayesian logistic regression model using PyMC3

    Parameters
    ----------
    X_train : numpy array
        The training data
    y_train: numpy array
        The training labels (WES result in our case)
    X_val: numpy array
        The validation/testing data
    reg_method: str
        Which prior for the regression coefficients to use.
    prior_vars_hs: int
        The estimated number of relevant features for the Finnish horseshoe prior
    export_model: bool
        Whether to export the trained model
        
    Returns
    -------
    train_means: numpy array
        The predictions on the training data
    val_means: numpy array
        The predictions on the validation/test data
    """
    
    with pm.Model() as model: 
      alfa = pm.Normal('alfa', mu=0, sd=3) 
      u = np.mean(y_train)
      sigma = np.sqrt(1/u * (1/(1-u))) # from https://arxiv.org/pdf/1707.01694.pdf p.15, fixed
      
      if reg_method == 'Log_reg_vanilla':
        beta = beta_no_reg(0,2,X_train.shape[1]) #no reg
      elif reg_method == 'Log_reg_Finnish_Horseshoe':
        beta = beta_reg_horseshoe(4,2.5,prior_vars_hs,X_train.shape[1],X_train.shape[0],sigma) #Finnish/regularized horseshoe
      elif reg_method == 'Log_reg_SpikeSlab':
        beta = beta_spike_slab(X_train.shape[1], 0.001) #spike/slab 
      else:
        raise ValueError('Selected reg_method should be Log_reg_vanilla, Log_reg_Finnish_Horseshoe or Log_reg_SpikeSlab')
            
    
      X_shared = pm.Data('x_shared', X_train)
      
      mu = alfa + pm.math.dot(X_shared,beta)
      θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-mu)))      
      y_pred = pm.Bernoulli('y_pred', p=θ, observed=y_train) 
              
      if reg_method == 'Log_reg_vanilla':
        trace = pm.sample(chains=2)
      elif reg_method == 'Log_reg_Finnish_Horseshoe':
        trace = pm.sample(tune=6000, target_accept=0.99, max_treedepth=15, chains=2)
      elif reg_method == 'Log_reg_SpikeSlab':
        trace = pm.sample(2000, tune=8000, nuts={"target_accept":0.99, "max_treedepth":15},chains=2) #spike and slab sampler
    
      X_shared.set_value(X_train)
      ppc = pm.sample_posterior_predictive(trace, samples=50000, model=model, vars=[y_pred,θ])
      train_means = ppc['y_pred'].mean(axis=0)
       
      X_shared.set_value(X_val)
      ppc = pm.sample_posterior_predictive(trace, samples=50000, model=model, vars=[y_pred,θ])
      val_means = ppc['y_pred'].mean(axis=0)
      
    if export_model == True:
      with open(os.path.join(os.path.realpath('.'), "trained_model"), 'wb') as buff:
          pickle.dump({'model': model, 'trace': trace, 'X_shared': X_shared}, buff)
      print("Exported trained model to " + str(os.path.join(os.path.realpath('.'), "trained_model")))
    return train_means, val_means

def run_logreg(X_train_val, y_train_val, X_real_test, y_real_test, OUTER_FOLD_COUNT):
    """
    Run the training and evaluation of the Bayesian logistic regression models
    
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
    OUTER_FOLD_COUNT: int
        The current outer fold
        
    Returns
    -------
    results: numpy array
        The predictions
    brier_scores: numpy array
        The Brier score after evaluating the model with the current parameters and data 
    """
    alg_scores = np.empty((66, 3 + len(X_train_val)),dtype=object)
    brier_scores = np.empty((66, 3 + len(X_train_val)),dtype=object)
                    
    alg_scores[:] = np.nan
    current_fold = 0
    alg_scores_indexer = 0
    
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    skf.get_n_splits(X_train_val, y_train_val)
    
    for train_index, test_index in skf.split(X_train_val, y_train_val):
        current_fold = current_fold + 1
        X_train, X_val = X_train_val[train_index,:], X_train_val[test_index,:]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        
        regs = ['Log_reg_vanilla', 'Log_reg_Finnish_Horseshoe', 'Log_reg_SpikeSlab']
        
        for i in range(3):
            train_means, val_means = bayes_log_reg(X_train, y_train, X_val, regs[i]) 
            alg_scores = fill_results_array(alg_scores, regs[i], str(OUTER_FOLD_COUNT) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, False, y_train, y_val)
            brier_scores = fill_results_array(brier_scores, regs[i], str(OUTER_FOLD_COUNT) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, True, y_train, y_val)
            alg_scores_indexer += 2
            
    for i in range(3):
        current_fold = 11
        X_train = X_train_val
        y_train = y_train_val
        X_val = X_real_test
        y_val = y_real_test
        
        train_means, val_means = bayes_log_reg(X_train, y_train, X_val, regs[i]) 
        alg_scores = fill_results_array(alg_scores, regs[i], str(OUTER_FOLD_COUNT) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, False, y_train, y_val)
        brier_scores = fill_results_array(brier_scores, regs[i], str(OUTER_FOLD_COUNT) + '_' + str(current_fold), train_means, val_means, alg_scores_indexer, True, y_train, y_val)
        alg_scores_indexer += 2
    
    return alg_scores, brier_scores