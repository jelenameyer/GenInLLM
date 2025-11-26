#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse

# helper, detect int-like column names between 0 and 99
def is_int_like_colname(colname):
    """Detect int-like column names between 0 and 99."""
    try:
        num = int(colname)
        return 0 <= num <= 99
    except ValueError:
        return False


def simulate_logprob_data(path, out_path, semi_random, mean, sd, seed):
    np.random.seed(seed)
    os.makedirs(out_path, exist_ok=True)

    # List of task names
    task_names = [
        "AUDIT", "BARRAT", "BART", "CARE", "CCT", "DAST", "DFD", "DFE",
        "DM", "DOSPERT", "FTND", "GABS", "LOT", "MPL", "PG", "PRI",
        "SOEP", "SSSV"
    ]

    # Collect all unique model names
    all_model_names = set()
    for task in task_names:
        pattern = os.path.join(path, f"*_{task}_prompting_results.csv")
        for file in glob.glob(pattern):
            model = os.path.basename(file).replace(f"_{task}_prompting_results.csv", "")
            all_model_names.add(model)

    all_model_names = sorted(list(all_model_names))
    # for correct ordering, create global mapping: model_name → model_index
    model_index = {model: i+1 for i, model in enumerate(all_model_names)}
    # Offsets only if semi-random
    if semi_random:
        model_offsets = {m: np.random.uniform(-2, 2) for m in all_model_names}
    else:
        model_offsets = {m: 0.0 for m in all_model_names}

    # Simulation loop, read each file all logprob columns with (semi) randomly produced numbers, from same or different distribution per model
    for task in task_names:
        pattern = os.path.join(path, f"*_{task}_prompting_results.csv")

        # Sort files by model number to guarantee same model → same index
        files = []
        for file in glob.glob(pattern):
            model = os.path.basename(file).replace(f"_{task}_prompting_results.csv", "")
            files.append((model_index[model], model, file))
        files.sort()  # Sort by model_index

        # Each task starts again at 1
        for num, (idx, model, file) in enumerate(files, start=1):
            offset = model_offsets[model]

            df = pd.read_csv(file)
            logprob_cols = [c for c in df.columns if c.startswith("log_prob_")]
            intlike_cols = [c for c in df.columns if is_int_like_colname(c)]
            cols_to_randomize = list(set(logprob_cols + intlike_cols))

            df_sim = df.copy()

            # Randomize columns
            for col in cols_to_randomize:
                df_sim[col] = np.random.normal(
                    loc=mean + offset,
                    scale=sd,
                    size=len(df_sim)
                )

            # zero-padded output filename
            out_file = os.path.join(out_path, f"{num:02d}_{task}_prompting_results.csv")
            df_sim.to_csv(out_file, index=False)

            print(f"Saved: {out_file}  (model: {model}, offset: {offset:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="Simulate logprob CSV data.")

    parser.add_argument("--path", type=str, default="LLM_data",
                        help="Input folder containing CSVs.")

    parser.add_argument("--out_path", type=str, default="simulation_random",
                        help="Output folder for simulated files.")

    parser.add_argument("--semi_random", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Use model-specific offsets (True/False).")

    parser.add_argument("--mean", type=float, default=-10,
                        help="Mean of the distribution.")

    parser.add_argument("--sd", type=float, default=1.0,
                        help="Standard deviation.")

    parser.add_argument("--seed", type=int, default=13,
                        help="Random seed.")

    args = parser.parse_args()

    simulate_logprob_data(
        path=args.path,
        out_path=args.out_path,
        semi_random=args.semi_random,
        mean=args.mean,
        sd=args.sd,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
