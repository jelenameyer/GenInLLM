# packages
import pandas as pd
import numpy as np
import glob
import os

# helper
def is_int_like_colname(colname):
    try:
        num = int(colname)
        return 0 <= num <= 99
    except ValueError:
        return False


np.random.seed(13)   # seed for reproducible randomness

# TASK NAMES
task_names = ["AUDIT", "BARRAT", "BART", "CARE", "CCT", "DAST", "DFD", "DFE", 
              "DM", "DOSPERT", "FTND", "GABS", "LOT", "MPL", "PG", "PRI", 
              "SOEP", "SSSV"]

#task_names = ["AUDIT"]     # for testing
path = "LLM_data"          # folder with original CSVs
out_path = "simulation_random" 
os.makedirs(out_path, exist_ok=True)

for task in task_names:
    # Load all files matching pattern
    pattern = os.path.join(path, f"*_{task}_prompting_results.csv")
    for file in glob.glob(pattern):
        
        model_name = os.path.basename(file).replace(f"_{task}_prompting_results.csv", "")
        
        # Read data
        df = pd.read_csv(file)

        # We want to randomize:
        #  - all columns named log_prob_*
        #  - all columns containing integers 0–99 (single number categories)
        
        
        logprob_cols = [c for c in df.columns if c.startswith("log_prob_")]
        intlike_cols = [c for c in df.columns if is_int_like_colname(c)]

        cols_to_randomize = list(set(logprob_cols + intlike_cols))

        # Copy data
        df_work_with = df.copy()

        # Replace selected columns with random log-probs (-1 to -20)
        for c in cols_to_randomize:
            df_work_with[c] = np.random.uniform(-20, -1, size=len(df_work_with))

        # Save output
        out_file = os.path.join(out_path, f"{model_name}_{task}_prompting_results.csv")
        df_work_with.to_csv(out_file, index=False)

        print(f"Saved randomized file: {out_file}")


