import pystan
import pandas as pd
import numpy as np

# Load the data

# Load stan model

# Create Stan model object (compile Stan model)
sm = pystan.StanModel(model_code=model_definition)

data = {
    "N_row": aggregated.shape[0],
    "N_col": aggregated.shape[1],
    "y": aggregated,
    "L": 14 # We use 14 observations to estimate the model, we will then approximate the LFO using this estimation
}
