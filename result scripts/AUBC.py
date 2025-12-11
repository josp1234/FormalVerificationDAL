import sys
from typing import Optional, Dict
import numpy as np
import pandas as pd


def trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area under curve using the trapezoid rule."""
    return np.trapezoid(y, x)


def compute_aubc_frame(
        df: pd.DataFrame,
        n_col: Optional[str] = None,
        normalize: bool = False,
        max_score: float = 1.0,
        dropna: bool = True,
        interpolate: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute AUBC for all method columns.

    Args:
        df: DataFrame with rows=AL rounds; columns=methods (+ optional n_col).
        n_col: name of column giving labeled counts per round; if None, uses 0..T-1.
        normalize: if True, divide each AUBC by the maximum possible area
                   (= (x_T - x_0) * max_score).
        max_score: upper bound of the metric (1.0 for accuracy).
        dropna: if True, drop rows where the method is NaN (per-method basis).
        interpolate: one of {None, 'linear', 'nearest', 'pad', 'bfill'} to fill NaNs.

    Returns:
        Dict mapping method name -> AUBC (or normalized AUBC if normalize=True).
    """
    # Build x-axis (labeled counts or round indices)
    if n_col is not None:
        if n_col not in df.columns:
            raise ValueError(f"n_col '{n_col}' not in CSV columns: {df.columns.tolist()}")
        x = df[n_col].to_numpy(dtype=float)
    else:
        x = np.arange(len(df), dtype=float)

    # Basic validation for x
    if len(np.unique(x)) != len(x):
        raise ValueError("x values (rounds or n) contain duplicates; expected strictly increasing.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing.")

    # Candidate method columns
    candidate_cols = [c for c in df.columns if c != n_col]

    results = {}
    for col in candidate_cols:
        y = df[col].astype(float).copy()

        # Handle NaNs
        if interpolate:
            y = y.interpolate(method=interpolate, limit_direction="both")
        if dropna:
            mask = ~y.isna()
            x_col = x[mask.to_numpy()]
            y_col = y[mask].to_numpy(dtype=float)
        else:
            # If NaNs remain, np.trapz will propagate them; warn explicitly
            if y.isna().any():
                raise ValueError(f"Column '{col}' contains NaNs; use --interpolate or --no-dropna.")
            x_col, y_col = x, y.to_numpy(dtype=float)

        if len(x_col) < 2:
            results[col] = np.nan
            continue

        area = trapezoid_area(x_col, y_col)
        if normalize:
            denom = (x_col[-1] - x_col[0]) * max_score
            area = area / denom if denom > 0 else np.nan
        results[col] = float(area)

    return results


def main():
    to_normalize = True
    res = {}
    for trial_name in ("MNIST_oct_26_32_10epochs50samples", "fashionMNIST_oct_26_32_10epochs50samples",
                       "CIFAR10_nov_9_1000samples"):
        csv_path = f"results/combined_total_retrained_trial_{trial_name}_avg.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read CSV: {e}", file=sys.stderr)
            sys.exit(1)
        # if args.out is None:
        out_path = "results/AUBC/all.csv"
        to_drop = ["coreset_w_samples_5", "coreset_w_samples_10", "FGSM_bounded_5", "FVAAL_rand_w_samples_5",
                   "FGSM_w_samples_5", "DFAL_w_samples_5",
                   "DFAL_FGSM_bounded_5",
                   "badge_w_samples_5",
                   "badge_FGSM_bounded_5",
                   "rand_w_samples_5", "rand_FGSM_bounded_5"
                   ]
        to_drop = [col for col in to_drop if col in df.columns]
        df = df.drop(to_drop, axis=1)
        rename_d = {}
        for col in df.columns:
            suffix = col.rsplit('_', 1)[-1]
            try:
                suffix = int(suffix)  # works with leading/trailing spaces and +/-
            except ValueError:
                pass
            if type(suffix) == int and suffix > 10:
                to_drop.append(col)
                continue
            if col.startswith('FGSM_w_samples'):
                rename_d[col] = "FVAAL"
            elif col == "FGSM_0":
                rename_d[col] = "FVAAL-No-Adv"
            elif col == "DFAL_0":
                rename_d[col] = "DFAL-No-Adv"
            elif col.startswith('DFAL_w_samples'):
                rename_d[col] = f"DFAL+FV-Adv"
            elif col.startswith('DFAL_FGSM'):
                rename_d[col] = f"DFAL+FGSM-Adv"
            elif col.startswith('badge_w_samples'):
                rename_d[col] = f"BADGE+FV-Adv"
            elif col.startswith('badge_FGSM_bounded'):
                rename_d[col] = f"BADGE+FGSM-Adv"
            elif col == "badge":
                rename_d[col] = "BADGE"
            elif col.startswith("coreset_w_samples"):
                rename_d[col] = f"CoreSet+FV-Adv"
            elif col == "coreset":
                rename_d[col] = "CoreSet"
            elif col.startswith("rand_w_samples"):
                rename_d[col] = f"Random+FV-Adv"
            elif col.startswith("rand_FGSM"):
                rename_d[col] = f"Random+FGSM-Adv"
            elif col == "rand":
                rename_d[col] = "Random"
        # df = df.drop(to_drop, axis=1)
        df = df.rename(columns=rename_d)
        try:
            results = compute_aubc_frame(
                df,
                n_col=None,
                normalize=to_normalize,
                max_score=1.0,
                dropna=False,
                interpolate=None,
            )
        except Exception as e:
            print(f"Error computing AUBC: {e}", file=sys.stderr)
            sys.exit(2)

        # Print nicely
        norm_tag = " (normalized)" if to_normalize else ""
        print(f"AUBC{norm_tag}:")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}" if pd.notna(v) else f"  {k}: NaN")
        dataset = trial_name.rsplit('_')[0]
        res[dataset] = results
    res_df = pd.DataFrame.from_dict(res)

    # Optional: sort for nicer order
    res_df = res_df.sort_index().sort_index(axis=1)

    # Add row-wise average (ignores NaNs by default)
    res_df["avg"] = res_df.mean(axis=1)

    # Save to CSV
    res_df = res_df.round(5)
    res_df = res_df[["MNIST", "fashionMNIST", "CIFAR10", "avg"]]

    res_df.to_csv(out_path)


if __name__ == "__main__":
    main()
