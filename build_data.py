#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:58:29 2020

@author: orangebacked
"""

import os
import pandas as pd
from copy import copy
import numpy as np

def load_dataset(path_dataset):
    cultivos = pd.read_csv(path_dataset)
    cultivos = cultivos.drop(["Unnamed: 0", "CODDEPTO", "CODMPIO"], axis=1)
    aggregated = cultivos.groupby(["DEPARTAMENTO", "year"]).agg(sum)
    aggregated = aggregated.reset_index().pivot(index="year", columns='DEPARTAMENTO', values='value')
    aggregated = aggregated - aggregated.shift(periods=1)
    aggregated = aggregated.dropna()
    aggg = copy(aggregated)
    # aggregated.head()
    idx_row, idx_col = np.where(np.isclose(aggregated, 0.))
    idx_row, idx_col = np.where(np.isnan(aggregated))
    data_std = np.std(aggregated.values, axis=0)
    aggregated = aggregated.values / data_std
    return aggregated

def save_dataset(dataset, path):
    pathplus = path + '/' + 'agregated.csv'
    np.savetxt(pathplus, dataset, delimiter=",")


if __name__ == "__main__":
    # 
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = './data/clean.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Save the datasets to files
    save_dataset(dataset, './data/formated_data')
    
    
