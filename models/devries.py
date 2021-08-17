# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

LIST_HPO_CLASSIFICATION = os.path.join(os.path.join(os.path.realpath('.'), "data" , "list_hpo_class_for_devries_score.csv"))

def check_if_hpo_classified(dataframe):
    """
    Check if all the HPO terms in the current dataframe have already been categorized  into one of the four categories needed to calculate the De Vries score.
    If there are missing classifications, ask for user input to update the list.
    
    Parameters
    ----------
    dataframe: pandas dataframe
        The data with HPO terms
        
    Returns
    -------
         -
    """
    facial_congenital_hpo = pd.read_csv(LIST_HPO_CLASSIFICATION)
    dataframe = dataframe.iloc[:,dataframe.columns.astype(str).str.contains("HP:")]
    df_hpo_only = dataframe.iloc[:,~dataframe.columns.astype(str).str.contains("HP:z")]
    
    sum_hpo = df_hpo_only.sum(axis=0)
    non_zero_hpo = sum_hpo[sum_hpo>0].index
    
    facial_congenital_hpo_this_dataframe = facial_congenital_hpo[facial_congenital_hpo.hpo_id.str.contains('|'.join(non_zero_hpo))]
    not_classified_hpo = facial_congenital_hpo_this_dataframe[facial_congenital_hpo_this_dataframe.facial_congenital.isnull()]
    
    if not_classified_hpo.shape[0] > 0:
        print("There are non classified (congenital/non-facial dysmorphisms/facial dysmorphism/none) HPO terms!")
        print("Please indicate for each one what it is (1=none, 2=congenital anomaly, 3=facial dys, 4=non facial dys)")
        for index, row in not_classified_hpo.iterrows():
            print(not_classified_hpo[not_classified_hpo.index==index].hpoName)
            input_class = input()
            if input_class == "1":
                facial_congenital_hpo.iloc[index,4] = "none"
            elif input_class =="2":
                facial_congenital_hpo.iloc[index,4] = "congenital abnormality"
            elif input_class =="3":
                facial_congenital_hpo.iloc[index,4] = "facial dysmorphism"
            elif input_class =="4":
                facial_congenital_hpo.iloc[index,4] = "non-facial dysmorphism"
        facial_congenital_hpo.to_excel(LIST_HPO_CLASSIFICATION , index=False)
    return
  
def class_de_vries_score(dataframe, column, list_of_symptoms):
    """
    Calculate a specific subcategory of the De Vries score

    Parameters
    ----------
    dataframe: pandas dataframe
        The data with HPO terms
    column: str
        The subcategory of the De Vries score
    list_of_symptoms: list
        List of symptoms to check for this subcategory
        
    Returns
    -------
    dataframe: pandas dataframe
        The data with HPO terms with added de Vries score and separate subcategories
    """
    mask =  dataframe.hpo_names.astype(str).str.contains('|'.join(list_of_symptoms))
    dataframe.loc[:,column] = 0
    dataframe.loc[mask, column] = 1
    return dataframe

def run_devries(dataframe):
    """
    Calculate the De Vries score

    Parameters
    ----------
    dataframe: pandas dataframe
        The data with HPO terms
        
    Returns
    -------
    dataframe: pandas dataframe
        The data with HPO terms with added de Vries score and separate subcategories
    """
    check_if_hpo_classified(dataframe)
    facial_congenital_hpo = pd.read_csv(LIST_HPO_CLASSIFICATION )
    hpo_congenital = facial_congenital_hpo[facial_congenital_hpo.facial_congenital=="congenital abnormality"]["hpo_id"]
    hpo_fac_dys = facial_congenital_hpo[facial_congenital_hpo.facial_congenital=="facial dysmorphism"]["hpo_id"]
    hpo_non_fac_dys = facial_congenital_hpo[facial_congenital_hpo.facial_congenital=="non-facial dysmorphism"]["hpo_id"]
    dataframe["congenital_abnormality"] = dataframe.iloc[:,dataframe.columns.str.contains('|'.join(hpo_congenital))].sum(axis=1)
    dataframe["facial_dysmorphy"] = dataframe.iloc[:,dataframe.columns.str.contains('|'.join(hpo_fac_dys))].sum(axis=1)
    dataframe["non-facial_dysmorphy"] = dataframe.iloc[:,dataframe.columns.str.contains('|'.join(hpo_non_fac_dys))].sum(axis=1)
    dataframe["IDALL"] = 1
    
    names_severe =["Intellectual disability, severe", "Profound global developmental delay", "Severe global developmental delay"]
    dataframe = class_de_vries_score(dataframe, "severe_ID", names_severe)
    
    names_seizures = ["Epilepsy","eizures", "Myoclonic absence"]
    dataframe = class_de_vries_score(dataframe, "seizures", names_seizures)

    try:
        dataframe.loc[:,"mild_ID"] = dataframe.loc[:, "HP:0001256"]
    except:
        #this HPO term is not in the dataframe, so all negative
        dataframe['mild_ID'] = 0
        
    try:
        dataframe.loc[:,"moderate_ID"] = dataframe.loc[:, "HP:0002342"]
    except:
        #this HPO term is not in the dataframe, so all negative
        dataframe['moderate_ID'] = 0
    
    names_gr = ["Intrauterine growth retardation", "Small for gestational age"]
    dataframe = class_de_vries_score(dataframe, "Prenatal GR", names_gr)
    
    names_microcephaly = ['icrocephaly', 'Decreased head circumference']
    dataframe = class_de_vries_score(dataframe, "microcephaly", names_microcephaly)

    names_short_stature = ["length less than 3rd percentile", "hort stature"]
    dataframe = class_de_vries_score(dataframe, "short stature", names_short_stature)
    
    names_macrocephaly = ["acrocephaly", 'Increased head circumference']
    dataframe = class_de_vries_score(dataframe, "macrocephaly", names_macrocephaly)
    
    names_tall_stature = ["length greater than 97th percentile", "all stature"]
    dataframe = class_de_vries_score(dataframe, "tall stature", names_tall_stature)
    dataframe["ChecklistScore"] = 0
    devlop_delay = dataframe["severe_ID"] + dataframe["seizures"] + 1
    devlop_delay[devlop_delay > 2] = 2
    dataframe["ChecklistScore"] = dataframe["ChecklistScore"] + devlop_delay + (dataframe["Prenatal GR"] * 2)
    dataframe.ChecklistScore[(dataframe["congenital_abnormality"] + dataframe["non-facial_dysmorphy"]) > 1] = dataframe.ChecklistScore + 2
    dataframe.ChecklistScore[(dataframe["congenital_abnormality"] + dataframe["non-facial_dysmorphy"]) == 1] = dataframe.ChecklistScore + 1
    postnatal_growth_anomalies = dataframe["microcephaly"] + dataframe["macrocephaly"] + dataframe["short stature"] + dataframe["tall stature"]
    postnatal_growth_anomalies[postnatal_growth_anomalies > 2] = 2
    dataframe["ChecklistScore"] = dataframe["ChecklistScore"] + postnatal_growth_anomalies
    dataframe.ChecklistScore[dataframe["facial_dysmorphy"] > 1] = dataframe.ChecklistScore + 2
    return dataframe