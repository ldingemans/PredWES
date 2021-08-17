# PredWES

This is the repository for the model behind PredWES, available online at https://humandiseasegenes.nl/predwes.

With the scripts in this repository, it is possible to 

* Run the pretrained Finnish Horseshoe yourself
* Retrain the Finnish Horseshoe model with your own data
* Run all models described in the paper (SVM, MLP, De Vries Score, all three Bayesian logistic regression models and the Needell algorithm) to see which one performs best with your data

<h2>Prerequisites</h2>
1) First install the needed dependencies using pip:

```
pip install pandas numpy pymc3 theano sklearn obonet tqdm pywaffle
```

2) Download/clone this repository.
3) Download the pretrained model from https://drive.google.com/file/d/1040zU2GWAU_-N_KtN3E2WYo8rqNn3UNQ/view?usp=sharing and place it in the Prod_model directory.

<h2>Running the pretrained Finnish Horseshoe model</h2>

Run the  `load_production_model.py` from the prod_model directory directly from the command line, giving the HPO terms seperated by semi-colons as input, as in:

```
python3 prod_model/load_production_model.py HP:0003808,HP:0001252,HP:0000252
```

<h2>Retrain the Finnish Horseshoe model</h2>

It is possible as well to retrain the model on another dataset. One should add the data in a feather file with three columns: one column should contain the HPO terms (the IDs) in a list, the second column the HPO terms (the names) in a list, and the third column the corresponding WES result. An example dataset (which is randomly generated using the `generate_data()` function in the `train_eval_model.py file`) is provided to illustrate this.

Then, simply either run `retrain_predwes.py` after replacing the `random_generated_hpo_terms.ftr` file with a real dataset or replacing the path in `retrain_predwes.py`. This will evaluate and train the Finnish Horseshoe model on the data using the nested cross-validation approach as in the paper. The mean Brier score over all test folds is calculated and displayed. 
Finally, a final model is trained on the whole dataset and the model is exported. This can consequently be used in a similar fashion as in the `load_production_model.py` file to generate new predictions.

<h2>Running all models</h2>

It is possible as well to run the nested cross-validation procedure on all models described in the paper on another dataset as well. Again, one should add the data in a feather file with three columns: one column should contain the HPO terms (the IDs) in a list, the second column the HPO terms (the names) in a list, and the third column the corresponding WES result. An example dataset (which is randomly generated using the `generate_data()`function in the `train_eval_model.py file`) is provided to illustrate this.

Next, run `train_eval_model.py` after replacing the `random_generated_hpo_terms.ftr` file with a real dataset or replacing the path in `train_eval_model.py`. This will evaluate train not only the Finnish Horseshoe model, but all other models described in the paper as well on the data using the nested cross-validation approach. 

A table is generated and exported in both latex and csv with the results - similar to table 2 in the paper, the provide an easy overview of the performance of the different classifiers.

Functions to generate the other figures and tables are not included in this repository, since the original patient data can not be supplied - and therefore these cannot be exactly reproduced. Of course, these functions are available on request, if one desires.
