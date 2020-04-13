##################################################################################
# This script contains the neccesary function to load the data
##################################################################################

import numpy as np
import pandas as pd

def data_loading(path):
    # Reac the data
    cultivos = pd.read_csv(path)

    # Drop irrelevant variables
    cultivos = cultivos.drop(["Unnamed: 0", "CODDEPTO", "CODMPIO"], axis=1)

    # Aggregate the municipios
    aggregated = cultivos.groupby(["DEPARTAMENTO", "year"]).agg(sum)

    # Pivot the table in order to have years as rows and departamentos as columns
    aggregated = aggregated.reset_index().pivot(index="year", columns='DEPARTAMENTO', values='value')

    # If we want differences:
    aggregated = aggregated - aggregated.shift(periods=1)
    aggregated = aggregated.dropna()

    # Standarizing the data
    data_std = np.std(aggregated.values, axis=0)
    aggregated = aggregated.values / data_std

    return aggregated
