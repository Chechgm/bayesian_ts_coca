# 1.  Introduction
This is the repository for the paper "A Bayesian time series model of Coca leaf production in Colombia" submitted to the LXAI workshop (ICML2020). This repository contains all the code related to the investigation. To create the STAN model, it is necessary to run the train.py.  You can specify the location of the data and the subsequent results; although there are defaults. I will break down the components as sections in the following part :


# 2. Files in the main folder:

## a. Build_data.py
This file is a helper function to modify the raw data. The raw data can be found in the following website: (http://www.odc.gov.co/sidco/oferta/cultivos-ilicitos/departamento-municipio)
		
## b. Train.py

The train function trains a given model specified with STAN with data that has been previously transformed using the Build_data.py. The output of this function is: the run time, and the result of the STAN model saved as 'output_model.txt' in your results folder.

Evaluation.py

The evaluation of the model uses Pareto Smoothed Importance Sampling (PSIS) Leave Future Out (LFO) Cross Validation (CV) (PSIS-LFO-CV) …. Using the model that was specified in STAN in order to evaluate its performance. 


