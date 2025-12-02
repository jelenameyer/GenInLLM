#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse


def is_int_like_colname(colname):
    """Detect int-like column names between 0 and 99."""
    try:
        num = int(colname)
        return 0 <= num <= 99
    except ValueError:
        return False


def get_cols_to_randomize(df):
    """
    Extract and return sorted list of columns to randomize.
    Sorting ensures consistent mapping of column index → option index across tasks.
    """
    logprob_cols = [c for c in df.columns if c.startswith("log_prob_")]
    intlike_cols = [c for c in df.columns if is_int_like_colname(c)]
    # Use sorted() for deterministic ordering
    return sorted(set(logprob_cols + intlike_cols))


def simulate_logprob_data(path, out_path, semi_random, multi_option, mean, sd, seed):
    np.random.seed(seed)
    os.makedirs(out_path, exist_ok=True)

    task_names = [
        "AUDIT", "BARRAT", "BART", "CARE", "CCT", "DAST", "DFD", "DFE",
        "DM", "DOSPERT", "FTND", "GABS", "LOT", "MPL", "PG", "PRI",
        "SOEP", "SSSV"
    ]

    # ────────────────────────────────────────────────────────────────
    # Collect all unique model names
    # ────────────────────────────────────────────────────────────────
    all_model_names = set()
    for task in task_names:
        pattern = os.path.join(path, f"*_{task}_prompting_results.csv")
        for file in glob.glob(pattern):
            model = os.path.basename(file).replace(f"_{task}_prompting_results.csv", "")
            all_model_names.add(model)

    all_model_names = sorted(list(all_model_names))
    model_index = {model: i + 1 for i, model in enumerate(all_model_names)}

    # ────────────────────────────────────────────────────────────────
    # Model-specific offsets (if semi_random mode)
    # ────────────────────────────────────────────────────────────────
    if semi_random:
        model_offsets = {m: np.random.uniform(-2, 2) for m in all_model_names}
    else:
        model_offsets = {m: 0.0 for m in all_model_names}

    # ────────────────────────────────────────────────────────────────
    # FIRST PASS: Find max_K across all files (only if multi_option)
    # ────────────────────────────────────────────────────────────────
    option_means_per_model = {}

    if multi_option:
        max_K = 0
        for task in task_names:
            pattern = os.path.join(path, f"*_{task}_prompting_results.csv")
            for file in glob.glob(pattern):
                df = pd.read_csv(file)
                cols = get_cols_to_randomize(df)
                K = len(cols)
                if K > max_K:
                    max_K = K

        print(f"[INFO] Detected max_K = {max_K} across all tasks")

        # Pre-allocate option means for each model (size = max_K)
        for model in all_model_names:
            offset = model_offsets[model]
            option_means_per_model[model] = np.random.normal(
                loc=mean + offset,
                scale=sd,
                size=max_K
            )

    # ────────────────────────────────────────────────────────────────
    # SECOND PASS: Generate simulated data
    # ────────────────────────────────────────────────────────────────
    for task in task_names:
        pattern = os.path.join(path, f"*_{task}_prompting_results.csv")

        # Collect files and sort by model index
        files = []
        for file in glob.glob(pattern):
            model = os.path.basename(file).replace(f"_{task}_prompting_results.csv", "")
            files.append((model_index[model], model, file))
        files.sort()

        for num, (idx, model, file) in enumerate(files, start=1):
            offset = model_offsets[model]

            df = pd.read_csv(file)
            cols_to_randomize = get_cols_to_randomize(df)
            K = len(cols_to_randomize)

            df_sim = df.copy()

            if multi_option:
                # Use pre-allocated means; index k maps to sorted column k
                option_means = option_means_per_model[model]
                for k, col in enumerate(cols_to_randomize):
                    df_sim[col] = np.random.normal(
                        loc=option_means[k],
                        scale=sd,
                        size=len(df_sim)
                    )
            else:
                # Single distribution per model
                for col in cols_to_randomize:
                    df_sim[col] = np.random.normal(
                        loc=mean + offset,
                        scale=sd,
                        size=len(df_sim)
                    )

            out_file = os.path.join(out_path, f"{num:02d}_{task}_prompting_results.csv")
            df_sim.to_csv(out_file, index=False)

            print(
                f"Saved: {out_file}  "
                f"(model: {model}, offset: {offset:+.3f}, "
                f"K: {K}, multi_option: {multi_option})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Simulate logprob CSV data with optional model/option structure."
    )

    parser.add_argument(
        "--path", type=str, default="LLM_data",
        help="Input folder containing CSVs."
    )
    parser.add_argument(
        "--out_path", type=str, default="simulation_random",
        help="Output folder for simulated files."
    )
    parser.add_argument(
        "--semi_random", type=lambda x: x.lower() == "true", default=False,
        help="Use model-specific offsets (True/False)."
    )
    parser.add_argument(
        "--multi_option", type=lambda x: x.lower() == "true", default=False,
        help="Use one distribution per answer option per model (True/False)."
    )
    parser.add_argument(
        "--mean", type=float, default=-10,
        help="Mean of the distribution."
    )
    parser.add_argument(
        "--sd", type=float, default=1.0,
        help="Standard deviation."
    )
    parser.add_argument(
        "--seed", type=int, default=13,
        help="Random seed."
    )

    args = parser.parse_args()

    simulate_logprob_data(
        path=args.path,
        out_path=args.out_path,
        semi_random=args.semi_random,
        multi_option=args.multi_option,
        mean=args.mean,
        sd=args.sd,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


# Usage examples:
# python3 simulate_logprob_data.py --multi_option True
# python3 simulate_logprob_data.py --semi_random True --multi_option True

