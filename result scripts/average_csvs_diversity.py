import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# === Configure me ===
# INPUT_DIR = Path("results/CIFAR10 nov 9 1000/")   # folder containing your CSV files


def plot_all_columns_mean_std_onefig(
    dfs,
    columns=None,
    x_index_name=None,
    title="Mean ± Std Across DataFrames",
    show_bands=True,
    band_alpha=0.20
):
    """
    dfs: list[pd.DataFrame] with identical columns and aligned indexes
    columns: optional subset list of column names to include
    x_index_name: optional x-axis label
    show_bands: whether to draw ±1σ shaded bands
    band_alpha: transparency for the bands
    """
    if not dfs:
        raise ValueError("Provide at least one DataFrame.")

    base_cols = dfs[0].columns
    base_index = dfs[0].index
    if columns is None:
        columns = list(base_cols)

    # Strictly align everything (safe if indexes/columns are guaranteed identical)
    aligned = [d.reindex(index=base_index, columns=base_cols) for d in dfs]

    x = base_index.to_numpy()

    plt.figure()
    for col in columns:
        # Stack shape: (n_dfs, n_rows)
        stacked = np.vstack([d[col].to_numpy() for d in aligned])
        mean = np.nanmean(stacked, axis=0)
        std  = np.nanstd(stacked,  axis=0)

        # mean line
        plt.plot(x, mean, label=f"{col} mean")

        # ±1σ band
        if show_bands:
            plt.fill_between(x, mean - std, mean + std, alpha=band_alpha)

    plt.title(title)
    plt.xlabel(x_index_name or "index")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()



# def main():
    for method in ["badge_FGSM_bounded_10", "badge_w_samples_10", "DFAL_FGSM_bounded_10", "DFAL_w_samples_10", "FGSM_bounded_10",
                   "FGSM_w_samples_10", "rand_FGSM_bounded_10", "rand_w_samples_10"]:

        files = [INPUT_DIR + f"/{trial_name}_seed{j}/{method}/res.csv" for j in range(5)]
#         # Read the first file to fix the expected header/shape
#         first = pd.read_csv(files[0])
#         if first.empty:
#             raise SystemExit(f"First CSV {files[0]} has no rows.")
#         expected_cols = list(first.columns)
#
#         # Ensure all columns are numeric (except headers, which are column names)
#         try:
#             sum_df = first.astype(float)
#         except ValueError as e:
#             raise SystemExit(f"{files[0]} contains non-float data rows: {e}")
#         # dfs = [first]
#         # Accumulate remaining files
#         for f in files[1:]:
#             df = pd.read_csv(f)
#             # dfs.append(df)
#
#         # plot_all_columns_mean_std_onefig(dfs, ["FGSM_w_samples_10", "rand_w_samples_10"])
#             if list(df.columns) != expected_cols or df.shape != first.shape:
#                 raise SystemExit(
#                     f"Structure mismatch in {f}.\n"
#                     f"Expected columns/shape {expected_cols}, {first.shape}; "
#                     f"got {list(df.columns)}, {df.shape}"
#                 )
#             try:
#                 sum_df += df.astype(float)
#             except ValueError as e:
#                 raise SystemExit(f"{f} contains non-float data rows: {e}")
#
#         avg_df = sum_df / len(files)
#
#         # Write the averaged CSV (keep original headers, no index)
#         out_file = INPUT_DIR + f"{method}_avg.csv"
#         avg_df.to_csv(out_file, index=False)
#         print(f"Wrote {out_file} (averaged over {len(files)} files)")



if __name__ == "__main__":
    for trial_name in ["trial_MNIST_oct_26_32_10epochs50samples", "trial_fashionMNIST_oct_26_32_10epochs50samples", "trial_CIFAR10_nov_9_1000samples"]:
        INPUT_DIR = f"diversity/diversity_measure/{trial_name}/" # folder containing your CSV files

        for method in ["badge_FGSM_bounded_10", "badge_w_samples_10", "DFAL_FGSM_bounded_10", "DFAL_w_samples_10",
                       "FGSM_bounded_10",
                       "FGSM_w_samples_10", "rand_FGSM_bounded_10", "rand_w_samples_10"]:
            files = [INPUT_DIR + f"/{trial_name}_seed{j}/{method}/res.csv" for j in range(5)]

            # Read all CSVs into DataFrames
            dfs = [pd.read_csv(f) for f in files]

            # Optional sanity check: all shapes equal
            n_rows = dfs[0].shape[0]
            n_cols = dfs[0].shape[1]
            for f, df in zip(files, dfs):
                if df.shape != (n_rows, n_cols):
                    raise ValueError(f"File {f} has shape {df.shape}, expected {(n_rows, n_cols)}")

            # If your columns are not named 'mean' and 'std', adjust here:
            mean_col = "mean"
            std_col = "std"

            # Stack means and stds across files: shape (n_rows, n_files)
            means_matrix = np.stack([df[mean_col].to_numpy() for df in dfs], axis=1)
            stds_matrix = np.stack([df[std_col].to_numpy() for df in dfs], axis=1)

            # Number of files
            k = means_matrix.shape[1]

            # Combined mean per row (configuration)
            combined_mean = means_matrix.mean(axis=1)

            # Combined std per row using law of total variance:
            # Var_total = E[Var] + Var[E]
            combined_var = (stds_matrix ** 2 + (means_matrix - combined_mean[:, None]) ** 2).mean(axis=1)
            combined_std = np.sqrt(combined_var)

            # Build result DataFrame with same number of rows
            result = pd.DataFrame({
                mean_col: combined_mean,
                std_col: combined_std,
            })

            # Save to CSV
            output_path = INPUT_DIR + f"/{method}_avg.csv"
            result.to_csv(output_path, index=False)
            print(f"Saved combined results to {output_path}")
