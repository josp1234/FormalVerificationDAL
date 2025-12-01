import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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



def main():
    files = sorted(p for p in INPUT_DIR.glob("/*.csv") if p.name != OUTPUT_FILE.name)
    if not files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR.resolve()}")

    # Read the first file to fix the expected header/shape
    first = pd.read_csv(files[0])
    if first.empty:
        raise SystemExit(f"First CSV {files[0]} has no rows.")
    expected_cols = list(first.columns)

    # Ensure all columns are numeric (except headers, which are column names)
    try:
        sum_df = first.astype(float)
    except ValueError as e:
        raise SystemExit(f"{files[0]} contains non-float data rows: {e}")
    # dfs = [first]
    # Accumulate remaining files
    for f in files[1:]:
        df = pd.read_csv(f)
        # dfs.append(df)

    # plot_all_columns_mean_std_onefig(dfs, ["FGSM_w_samples_10", "rand_w_samples_10"])
        if list(df.columns) != expected_cols or df.shape != first.shape:
            raise SystemExit(
                f"Structure mismatch in {f}.\n"
                f"Expected columns/shape {expected_cols}, {first.shape}; "
                f"got {list(df.columns)}, {df.shape}"
            )
        try:
            sum_df += df.astype(float)
        except ValueError as e:
            raise SystemExit(f"{f} contains non-float data rows: {e}")

    avg_df = sum_df / len(files)

    # Write the averaged CSV (keep original headers, no index)
    avg_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE.resolve()} (averaged over {len(files)} files)")

if __name__ == "__main__":
    for trial_name in ["MNIST", "fashionMNIST", "CIFAR10"]:
        INPUT_DIR = Path(f"diversity/diversity_measure/{trial_name}/")  # folder containing your CSV files
        OUTPUT_FILE = INPUT_DIR / "average.csv"  # write output *inside* INPUT_DIR
        main()
