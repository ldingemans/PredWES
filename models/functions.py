import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from models.devries import run_devries
import os

def no_progress_loss_moving_average(iteration_stop_count=10, percent_increase=0.0):

    """
    Stop function that will stop after X iteration if the loss doesn't increase
    Parameters
    ----------
    iteration_stop_count: int
        search will stop if the loss doesn't improve after this number of iteration
    percent_increase: float
        allow this percentage of variation within iteration_stop_count.
        Early stop will be triggered if the data didn't change for more than this number
        after iteration_stop_count rounds

    """

    def stop_fn(trials, total_loss_last_iterations=None, iteration_no_progress=0):
        new_loss_average = 0
        for i in range(iteration_stop_count):
            if (i < len(trials.trials)):
                new_loss_average  += trials.trials[len(trials.trials) - (i+1)]["result"]["loss"]
                
        if len(trials.trials) > iteration_stop_count:        
            new_loss_average = new_loss_average/iteration_stop_count
        else:
            if len(trials.trials) == 0:
                new_loss_average = new_loss_average
            else:
                new_loss_average = new_loss_average /len(trials.trials)
        if total_loss_last_iterations is None:
            return False, [new_loss_average, iteration_no_progress + 1]                    
        if new_loss_average < total_loss_last_iterations:
            iteration_no_progress = 0
            total_loss_last_iterations = new_loss_average
        else:
            iteration_no_progress += 1
            # print(
            #     "No progress made: %d iteration on %d. total average loss last iterations %.5f, new_loss=%.5f"
            #     % (
            #         iteration_no_progress,
            #         iteration_stop_count,
            #         total_loss_last_iterations,
            #         new_loss_average,
            #     )
            # )
        if len(trials.trials) < 20:
            iteration_no_progress = 0
        return (
            iteration_no_progress >= iteration_stop_count,
            [total_loss_last_iterations, iteration_no_progress],
        )

    return stop_fn

def fill_results_array(results, model, current_fold, train_means, val_means, indexer,brier, y_train, y_val):
    """
    Filling the results array

    Parameters
    ----------
    Results: numpy array
        Current results array
    model: str
        Name of current model
    current_fold: str
        The current fold to add, in form outer_fold_inner_fold (for instance 1_6)
    train_means: numpy array
        The scores on the training data
    val_means: numpy array
        The scores on the validation/test data
    indexer: int
        Current line of the results array to write to
    brier: Boolean
        Use Brier score (if False use actual scores from model)
    y_train: numpy array
        The labels of the training data, needed to calculate Brier score
    y_val: numpy array
        the labels of the validation/test data, needed to calculate Brier score 
        
    Returns
    -------
    results: numpy array
        Updated current results array
    """
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    results[indexer, 0] = model
    results[indexer, 1] = current_fold
    if current_fold[-2:] == '11':
        results[indexer, 2] = 'test'
    else:
        results[indexer, 2] = 'val'
    if brier == False:
        results[indexer, 3:len(val_means) + 3] = val_means
    else:
        val_brier = []
        for i in range(len(val_means)):
            val_brier.append(brier_score_loss([y_val[i]], [val_means[i]]))
        results[indexer, 3:len(val_means) + 3] = val_brier
        
    results[indexer+1, 0] = model
    results[indexer+1, 1] = current_fold
    if current_fold[-2:] == '11':
        results[indexer+1, 2] = 'train_val'
    else:
        results[indexer+1, 2] = 'train'
    if brier == False:
        results[indexer+1, 3:len(train_means) + 3] = train_means
    else:
        train_brier = []
        for i in range(len(train_means)):
            train_brier.append(brier_score_loss([y_train[i]], [train_means[i]]))
        results[indexer+1, 3:len(train_means) + 3] = train_brier
    return results

def generate_table_1_baseline(generated_data, hpo_expanded):
    """
    Create the demographics table as in table 1 of the paper.
    If a specific demographic or symptom is not in the data, zeros are filled in.

    Parameters
    ----------
    generated_data: numpy array
        The data with HPO terms for which the table is to be created
    hpo_expanded: numpy array
        The data with HPO terms for which the table is to be created with the HPO terms as seperate columns
        
    Returns
    -------
    table: pandas dataframe
        The created table
    """
    table = np.zeros((14,3),dtype=object)
    generated_data = pd.concat([hpo_expanded, generated_data], axis=1)
    for i in range(3):
        if i == 0:
            df_subset = generated_data
        if i == 1:
            df_subset = generated_data[generated_data.WES_result == 1]
        if i == 2:
            df_subset = generated_data[generated_data.WES_result == 0]
        
        n = len(df_subset)
        if i > 0:
            n = str(n) + ' (' + str(np.round(len(df_subset) / len(generated_data) * 100,1)) + '%)'
            
        hpo_number = df_subset.hpo_terms.str.len().median()
        
        if 'age' in generated_data.columns:
            med_age = np.round(df_subset.age.median(),1)
            min_age = np.round(df_subset.age.min(),1)
            max_age = np.round(df_subset.age.max(),1)
        else:
            print("No age column found in the supplied data. Please add one if the age needs to be added.")
            med_age, min_age, max_age = 0,0,0
            
        if 'sex' in generated_data.columns:
            m_tot = pheno_ages.sex.value_counts()['m']
            f_tot = pheno_ages.sex.value_counts()['f']
            m_rel = int(np.round(pheno_ages.sex.value_counts(normalize=True)['m'] * 100))
            f_rel = int(np.round(pheno_ages.sex.value_counts(normalize=True)['f']*100))
        else:
            print("No sex/gender column found in the supplied data. Please add one if the gender needs to be added.")
            m_tot, f_tot, m_rel, f_rel = 0,0,0,0
            
        symptoms = ['severe_ID', 'seizures', 'microcephaly', 'macrocephaly','HP:0001363', 'HP:0000717', 'HP:0003808']
        
        table_temp =  [n, hpo_number, (str(m_tot) + '/' + str(f_tot) + ' (' + str(m_rel) + '%/' + str(f_rel) +'%)'), 
                      str(med_age) + ' (' + str(min_age) + ' - ' + str(max_age) +')']
        
        df_subset = run_devries(df_subset)
        
        for symp in symptoms:
            try:
                symp_pos = df_subset[symp].value_counts()[1]
                symp_neg = df_subset[symp].value_counts()[0]
                symp_pos_rel = int(np.round( df_subset[symp].value_counts(normalize=True)[1] * 100))
                symp_neg_rel = int(np.round( df_subset[symp].value_counts(normalize=True)[0] * 100))
            except:
                symp_pos, symp_neg, symp_pos_rel, symp_neg_rel = 0, 0, 0,0
            table_temp.append(str(symp_pos) + '/' + str(symp_neg) + ' (' + str(symp_pos_rel) + '%/' + str(symp_neg_rel)+ '%)')
                    
        table_temp.append(np.round(df_subset['facial_dysmorphy'].mean(),2))
        table_temp.append(np.round(df_subset['congenital_abnormality'].mean(),2))  
        table_temp.append(np.round(df_subset['ChecklistScore'].mean(),2))  
        
        table[:,i] = table_temp
        
    table = pd.DataFrame(table) 
    table.columns = ['Full cohort', 'WES positive', 'WES negative/VUS']
    table.index = ['n (number of individuals)', 'HPO terms per patient (median)', 'Gender (M/F)', 'Age in years (median & range)',
                   'Severe ID (+/-)', 'Seizures (+/-)', 'Microcephaly (+/-)', 'Macrocephaly (+/-)', 'Craniosynostosis (+/-)', 'Autism (+/-)',  'Abnormal muscle tone (+/-)', 'Facial dysmorphisms (mean)', 'Congenital anomalies (mean)', 'De Vries score (mean)']
        
    table.to_csv(os.path.join(os.path.realpath('.'), 'demographics_table_1.csv'))
    table.to_latex(os.path.join(os.path.realpath('.'), "demographics_table_1.tex"))
    
    print("Saved the demographics table!")
    return table
    
def generate_table_2_results(df_all_brier_scores, brier_devries):
    """
    Create the results table as in table 2 of the paper.

    Parameters
    ----------
    df_all_brier_scores: pandas dataframe
        All the evaluated Brier scores
    brier_devries: pandas series
        The Brier scores using De Vries score
        
    Returns
    -------
    results_table: pandas dataframe
        The created table
    """
    #calculate the mean Brier scores
    df_all_brier_scores['mean'] = df_all_brier_scores.iloc[:,3:].mean(axis=1)
    df_all_brier_scores = df_all_brier_scores.loc[:,[0,1,2,'mean']]
    
    test_scores = df_all_brier_scores[df_all_brier_scores.iloc[:,2] == 'test'].groupby(0).mean()
    val_scores = df_all_brier_scores[df_all_brier_scores.iloc[:,2] == 'val'].groupby(0).mean()
    train_scores = df_all_brier_scores[df_all_brier_scores.iloc[:,2] == 'train'].groupby(0).mean()
        
    #Create the results table to be exported
    results_table = pd.DataFrame(np.concatenate((train_scores, val_scores, test_scores),axis=1))
    results_table = results_table.append(pd.Series([np.nan, np.nan, brier_devries]),ignore_index=True)
    results_table.index = np.append(train_scores.index, 'De Vries score')
    results_table.columns = ['Training', 'Validation', 'Test']
    
    results_table.to_csv(os.path.join(os.path.realpath('.'), 'results_table_2.csv'))
    results_table.to_latex(os.path.join(os.path.realpath('.'), "results_table_2.tex"))
    
    print("Saved the results table!")
    return results_table

def generate_fig_1_waffle(y_true, predictions):
    """
    Create the waffle plot as in figure 1 of the paper

    Parameters
    ----------
    y_true: numpy array
        The true WES results
    predictions: numpy array
        The predicted WES results
        
    Returns
    -------
    The saved figures
    """
    import matplotlib.pyplot as plt
    from pywaffle import Waffle
    results = pd.DataFrame()
    results['y_pred'] = predictions
    results['y_val'] = y_true
    results = results.sort_values(by='y_pred')
    results = results.reset_index(drop=True)
    ordered_predictions = results.y_val.to_numpy()[::-1]
    
    ordered_predictions_ = []
    
    bottom_quartile = round(len(ordered_predictions) * 0.1) +1
    ordered_predictions_.append(ordered_predictions[:bottom_quartile-1])
    
    top_quartile = round(len(ordered_predictions) * 0.9)
    ordered_predictions_.append(ordered_predictions[top_quartile:])
    
    titles = ['Highest 10% scores according to PredWES', 'Lowest 10% scores according to PredWES']
    for i, ordered in enumerate(ordered_predictions_):
        data = {'Negative WES result': np.sum(ordered == 0), 'Positive WES result': np.sum(ordered == 1)}
        fig = plt.figure(
            FigureClass=Waffle, 
            rows=12, 
            values=data, 
            ordered_data=ordered,
            colors=("#ABD0E6", "#3787C0"),
            legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
            icons='child', icon_size=16, 
            title={'label': titles[i], 'loc': 'center'},
            vertical=True,
            icon_legend=True
        )
        plt.savefig(os.path.join(os.path.join(os.path.realpath('.'), "waffle_plot" + str(i) + ".png")) , dpi=300, bbox_inches='tight')
        plt.show()
    return

def generate_fig2_calibration_curve(y_true, predictions):
    """
    Create the calibration curve as in figure 2 of the paper

    Parameters
    ----------
    y_true: numpy array
        The true WES results
    predictions: numpy array
        The predicted WES results
        
    Returns
    -------
    The saved figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import calibration
    from math import ceil
    sns.set()
    y,x = calibration.calibration_curve(y_true, predictions, strategy='quantile', n_bins=6)
    plt.plot(np.linspace(0, 0.6, 10), np.linspace(0, 0.6, 10), label='Perfect calibration', color='cornflowerblue', linestyle='--')
    plt.plot(x, y,marker='o', label='PredWES')
    plt.ylabel('Yield of WES');plt.xlabel('Probability according to model'); 
    num_ticks = int(x[-1] // 0.1 + 2)
    ax = plt.gca() # grab the current axis
    ax.set_xticks(np.linspace(0,ceil(x[-1] * 10) / 10.0,num_ticks))
    x_labels = np.array(np.linspace(0,ceil(x[-1] * 10) / 10.0,num_ticks) * 100, dtype=int)
    x_labels = [str(s) + '%' for s in x_labels]
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%']) # set the labels to display at those ticks
    ax.set_xticklabels(x_labels) # set the labels to display at those ticks
    plt.title('Yield of WES per score of PredWES'); 
    plt.legend()
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()
    plt.savefig(os.path.join(os.path.join(os.path.realpath('.'), "calibration_curve.png")) , dpi=300, bbox_inches='tight')
    return


def generate_fig3_sig_features(hpo_expanded, y):
    """
    Create the diagnostic yield of a combination of symptoms as in figure 3 of the paper

    Parameters
    ----------
    hpo_expanded: numpy array
        The data with the HPO terms as separate columns 
    y: numpy array
        The correspondeing WES results
        
    Returns
    -------
    The saved figures
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    X = hpo_expanded
    try:
        hypotonia = X["HP:0003808"]
    except:
        hypotonia = pd.Series([0] * len(X))
        
    try:
        autism = X["HP:000017"]
    except:
        autism = pd.Series([0] * len(X))
        
    try:
        microcephaly = X[" HP:0000252"]
    except:
        microcephaly= pd.Series([0] * len(X))
    
    y, uniques = pd.factorize(y)
    
    results = np.array([[0.0,0.0],[0.0,0.0]])
    results[1,1] = np.mean(y[(autism == 1) & (hypotonia == 1)]) 
    results[1,0] = np.mean(y[(autism == 1) & (hypotonia == 0)])
    results[0,1] = np.mean(y[(autism == 0) & (hypotonia == 1)])
    results[0,0] = np.mean(y[(autism == 0) & (hypotonia == 0)])
    
    fig, axs= plt.subplots(nrows=1,ncols=3,figsize=((20,5)))
    sns.heatmap(results, annot=True, ax = axs[0],fmt='.0%',cmap="Blues"); #annot=True to annotate cells
    
    # labels, title and ticks
    axs[0].set_ylabel('Autism');axs[0].set_xlabel('Abnormal muscle tone'); 
    axs[0].set_title('Yield WES (%)'); 
    axs[0].xaxis.set_ticklabels(['no', 'yes']); axs[0].yaxis.set_ticklabels(['no', 'yes']);
    
    results = np.array([[0.0,0.0],[0.0,0.0]])
    results[1,1] = np.mean(y[(microcephaly == 1) & (hypotonia == 1)]) 
    results[1,0] = np.mean(y[(microcephaly == 1) & (hypotonia == 0)])
    results[0,1] = np.mean(y[(microcephaly == 0) & (hypotonia == 1)])
    results[0,0] = np.mean(y[(microcephaly == 0) & (hypotonia == 0)])
    
    sns.heatmap(results, annot=True, ax = axs[1],fmt='.0%',cmap="Blues"); #annot=True to annotate cells
    
    axs[1].set_ylabel('Microcephaly');axs[1].set_xlabel('Abnormal muscle tone'); 
    axs[1].set_title('Yield WES (%)'); 
    axs[1].xaxis.set_ticklabels(['no', 'yes']); axs[1].yaxis.set_ticklabels(['no', 'yes']);

    results = np.array([[0.0,0.0],[0.0,0.0]])
    results[1,1] = np.mean(y[(autism == 1) & (microcephaly == 1)]) 
    results[1,0] = np.mean(y[(autism == 1) & (microcephaly == 0)])
    results[0,1] = np.mean(y[(autism == 0) & (microcephaly == 1)])
    results[0,0] = np.mean(y[(autism == 0) & (microcephaly == 0)])
    
    sns.heatmap(results, annot=True, ax = axs[2] ,fmt='.0%',cmap="Blues"); #annot=True to annotate cells
    
    axs[2].set_ylabel('Autism');axs[2].set_xlabel('Microcephaly'); 
    axs[2].set_title('Yield WES (%)'); 
    axs[2].xaxis.set_ticklabels(['no', 'yes']); axs[2].yaxis.set_ticklabels(['no', 'yes']);
    plt.show()
    plt.savefig(os.path.join(os.path.join(os.path.realpath('.'), "sig_features.png")) , dpi=300, bbox_inches='tight')
    return

def generate_supp_table2_p_values(df_scores, n_patients, brier_devries):
    """
    Create the p value results table as in supp. table 2 of the paper.

    Parameters
    ----------
    df_all_brier_scores: pandas dataframe
        All the evaluated Brier scores
    n_patients: int
        Number of patients
    brier_devries: pandas series
        The Brier scores using De Vries score
        
    Returns
    -------
    df_p_values: pandas dataframe
        The created table
    """
    from scipy import stats
    
    df_test_scores = (df_scores[df_scores[2] == 'test'])
    del df_test_scores[1]
    del df_test_scores[2]
    
    flat_test_scores = np.zeros((8, n_patients+1), dtype=object)
    
    for i, model in enumerate(df_test_scores[0].unique()):
        flat_test_scores[i, 0] = model
        flat = df_test_scores[df_test_scores[0]  == model].iloc[:,1:-2].to_numpy(dtype=float).flatten()
        flat = flat[flat != 0]
        flat = flat[~np.isnan(flat)]
        flat_test_scores[i, 1:len(flat)+1] =flat
                
    flat_test_scores[-1,0] = 'DeVries'
    flat_test_scores[-1,1:len(brier_devries)+1] = brier_devries
    
    p_values_test = np.zeros((8,8))
    
    for i in range(len(flat_test_scores)):
            for y in range(len(flat_test_scores)):
                s, p = stats.ttest_ind(flat_test_scores[i,1:], flat_test_scores[y,1:], equal_var = False)
                p_values_test[i,y] = p
                
    df_p_values = pd.DataFrame(p_values_test)
    df_p_values.columns, df_p_values.index = flat_test_scores[:,0], flat_test_scores[:,0]
        
    df_p_values.to_csv(os.path.join(os.path.realpath('.'), 'pvalues_supptable_2.csv'))
    df_p_values.to_latex(os.path.join(os.path.realpath('.'), "pvalues_supptable_2.csv"))
    return df_p_values

def generate_supp_table3_hyperparameters(trial_files):
    """
    Create the p value results table as in supp. table 2 of the paper.

    Parameters
    ----------
    trial_files: list
        List of trial files to process
        
    Returns
    -------
    df_all_results: pandas dataframe
        The created table
    """
    df_all_results = pd.DataFrame()
    results = []
    for trial in trial_files:    
        for val in trial.best_trial['misc']['vals']:
            try:
                results.append([val,trial.best_trial['misc']['vals'][val][0]])
            except:
                results.append([val,trial.best_trial['misc']['vals'][val]])
    
    r = pd.DataFrame(results)
    print(r[r[0] == 'kernel'].value_counts())
    print(r[r[0] == 'solver'].value_counts())
    print(r[r[0] == 'normalize'].value_counts())
    print(r[r[0] == 'scale'].value_counts())
    try:
        df_results = pd.DataFrame(results).groupby(0).mean()
    except:
        df_results = pd.DataFrame(results)
    df_results.columns = ['hyperparameter', 'mean']
    try:
        df_results['std'] = pd.DataFrame(results).groupby(0).std()
    except:
        df_results['std'] = 0
    df_all_results = df_all_results.append(df_results)
    index_df=df_all_results['hyperparameter']

    df_all_results = np.round(df_all_results.loc[:,["mean", "std"]],1)
    df_all_results.index = index_df
    
    df_all_results.to_csv(os.path.join(os.path.realpath('.'), 'hyperparameters_supptable_3.csv'))
    df_all_results.to_latex(os.path.join(os.path.realpath('.'), "hyperparameters_supptable_3.tex"))
    return    

def generate_supp_fig1_plot_per_checklist_score(generated_data, hpo_expanded):
    """
    Create the diagnostic yield per De Vries score as in supp. figure 1 of the paper

    Parameters
    ----------
    generated_data: numpy array
        The data with the HPO terms as separate columns 
    hpo_expanded: numpy array
        The data with the HPO terms as separate columns 
        
    Returns
    -------
    The saved figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    sns.set()
    list_of_n = []
    results = pd.DataFrame()
    dataframe = run_devries(pd.concat([hpo_expanded, generated_data], axis=1))
    for i in range(1,11):
        test_score = pd.pivot_table(dataframe[dataframe.ChecklistScore == i].loc[:,["ChecklistScore","WES_result"]], columns="WES_result", aggfunc=np.sum)
        list_of_n.append(str(i) + '\n (n=' + str(np.sum(dataframe.ChecklistScore == i)) +')')
        test_score = test_score.apply(lambda x: x/x.sum(), axis=1)
        results[i] = test_score.iloc[0,:]
    
    results = results.fillna(0)
    negatief = results.iloc[0,:]
    positief = results.iloc[1,:]
    results = pd.DataFrame()
    results["positive WES"] = positief
    results["negative WES"] = negatief
    results.index = list_of_n
    results = results.T
    ax = results.T.plot.bar(stacked=True, colormap=ListedColormap(sns.color_palette("Blues_r", 2)),figsize=(10,7))
    plt.xticks(rotation=0, fontsize = 12)
    plt.ylim([0,1])
    plt.xlabel("Adapted De Vries Score")
    plt.ylabel("WES result")
    
    ax.figure.show()
    ax.figure.savefig(os.path.join(os.path.join(os.path.realpath('.'), "supp_fig1_checklist_score.png")) , dpi=300, bbox_inches='tight')
    return

def generate_supp_fig2_hpo_number_dist_graph(generated_data, predictions):
    """
    Create the distribution of number of HPO terms per PredWES score as in supp. figure 2 of the paper

    Parameters
    ----------
    generated_data: numpy array
        The data with the HPO terms as separate columns 
    predictions: numpy array
        The predicted WES results
        
    Returns
    -------
    The saved figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import binned_statistic
    sns.set()
    generated_data['number_hpo_terms'] = generated_data.hpo_terms.str.len()
    generated_data['preds'] = predictions
    generated_data.sort_values('preds', inplace=True)
    
    bins = binned_statistic(range(len(generated_data)),generated_data.number_hpo_terms.to_numpy(), bins=10, statistic='median')
    ax = sns.barplot(x=list(range(len(bins[0]))),y=bins[0], color='cornflowerblue')
    ax.set_title('Distribution of HPO terms per PredWES score')
    ax.set_ylabel('Median number of HPO terms')
    ax.set_xlabel('Binned PredWES scores')
    ax.set_xticklabels(['0-10%','10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],rotation=90)
    
    plt.savefig(os.path.join(os.path.join(os.path.realpath('.'), "supp_fig2_hpo_distribution.png")),dpi=300,bbox_inches="tight")
    return
