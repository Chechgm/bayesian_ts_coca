##################################################################################
# This script contains the neccesary function in order to run and evaluate a single
# model.
#
# Ideally we would like to have:
#   1. Some form of result saving scheme
#   2. Some form of outter loop to run this script on all the models (maybe in another script)
##################################################################################

import argparse
import pystan
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_path",  help="Path where the data lies")
parser.add_argument("model_path", help="Path where the model definition lies")
parser.add_argument("L",          help="Initial value from which we estimate the models")
args = parser.parse_args()

# Load the data
data = data_loading(args.path)

# Load stan model
with open (args.model_path, "r") as model_definition:
    model = model_defintion.read()

# Create Stan model object (compile Stan model)
sm = pystan.StanModel(model_code=model)

# Define the data for the model
model_data = {
    "N_row": data.shape[0],
    "N_col": data.shape[1],
    "y": data,
    "L": args.L
}

# Model evaluation
loo, ks, re_i = psis_lfo_cv(model, model_data, args.L)
