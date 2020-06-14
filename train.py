#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:32:47 2020

@author: orangebacked and chechgm
"""

import pystan
from numpy import genfromtxt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/formated_data/agregated.csv',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./models/ar_1.stan',
                    help="Directory containing model def")

parser.add_argument('--results_folder', default='./results/',
                    help="Optional, name of the where the previous fit is")

parser.add_argument('--fit', default=None,
                    help="Optional, name of the where the previous fit is")  




if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    data_dir = args.data_dir 
    modeldir = args.model_dir
    results_folder = args.results_folder
    
    with open(modeldir, 'r') as f:
        model_definition = f.read()


    aggregated = genfromtxt(data_dir, delimiter=',')

    data = {
        "N_row": aggregated.shape[0],
        "N_col": aggregated.shape[1], 
        "y": aggregated, 
        "L": 14
    }
    
    sm = pystan.StanModel(model_code=model_definition)

    t0 = time.time()
    
    fit = sm.sampling(data=data, iter=2000, warmup=1000, chains=5, algorithm="NUTS", seed=42, verbose=True, 
                      control={"adapt_delta":0.9, "max_treedepth":15}) # Recommendations from the results
    
    
    t1 = time.time()
    
    total = t1-t0
    
    timefile = results_folder + 'time.txt'
    
    output_model = results_folder + 'output.txt'
# TODO rename it with the name of the model
    with open(timefile, 'w') as f:
        f.write('The run time was: {}'.format(str(total)))
        
        
    with open('output_model.txt', 'w') as f:
        f.write(str(fit))