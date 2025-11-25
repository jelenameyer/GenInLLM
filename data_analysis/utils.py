# packages --------------------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Loading all data files of one task ------------------------------------------------------------
def load_dataframes(task_name, path = "LLM_data"): #"simulation_semi_random"):   #"simulation_random"):

    # Initialize empty list to store DataFrames
    dataframe = []

    path = path  # folder with CSVs of LLM answers

    for file in glob.glob(os.path.join(path, f"*_{task_name}_prompting_results.csv")):
        model_name = os.path.basename(file).replace(f"*_{task_name}_prompting_results.csv", "")
        
        # Read the CSV
        df = pd.read_csv(file)
        
        # Append to list
        dataframe.append(df)
        
    # Concatenate all DataFrames into one big DataFrame
    merged_data = pd.concat(dataframe, ignore_index=True)

    print(f"Merged DataFrame shape: {merged_data.shape}")
    print(f"Total models: {merged_data['model'].nunique()}")

    return(merged_data)


# filter out probability LLM assigned to real item answer  ------------------------------------------
def filter_pred_prob(data, human_col = "human_number"):
    data["prob_pred"] = data.apply(
        lambda row: row[f"prob_{row[human_col]}"], axis=1
    )
    return(data)
