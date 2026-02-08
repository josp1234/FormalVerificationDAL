import sys
import tensorflow as tf
import numpy as np
import os
from scipy.spatial.distance import pdist
import pandas as pd


def retrain_single_model(_x_train, _y_train, _model=None, _epoch_num=10, _num_classes=10,
                         lr=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    _model.fit(x=_x_train, y=_y_train, epochs=_epoch_num)
    return _model


def compute_feature_diversity(
        model,
        images,
        layer_name=None,
        metric="euclidean",
        return_distances=False,
):
    """
    Compute feature-space diversity statistics for a set of images.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    images : np.ndarray
        Array of inputs of shape (N, H, W, C) or (N, ...), already preprocessed
        as expected by `model`.
    layer_name : str or None, optional
        Name of the layer to use as feature extractor. If None, uses the last
        Dense layer if available, otherwise the second-to-last layer.
    metric : str, optional
        Distance metric to use for pairwise distances (passed to `scipy.spatial.distance.pdist`).
    return_distances : bool, optional
        If True, also return the raw 1D array of pairwise distances.

    Returns
    -------
    stats : dict
        Dictionary with summary statistics over pairwise distances:
        {
            "mean": ...,
            "median": ...,
            "p10": ...,
            "p90": ...,
            "min": ...,
            "max": ...,
            "num_points": N
        }
    distances : np.ndarray (optional)
        1D array of pairwise distances of length N*(N-1)/2, returned only if
        `return_distances=True`.
    """

    # --- Build feature extractor ---
    if layer_name is None:
        # Try to use last Dense layer as penultimate representation
        penultimate = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                penultimate = layer
                break
        if penultimate is None:
            # Fallback: just take the second-to-last layer
            if len(model.layers) < 2:
                raise ValueError("Model has fewer than 2 layers; specify layer_name explicitly.")
            penultimate = model.layers[-2]
        layer_name = penultimate.name

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )

    # --- Extract features ---
    features = feature_model.predict(images)
    N = features.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples to compute pairwise distances.")

    # --- Pairwise distances in feature space ---
    d = pdist(features, metric=metric)  # 1D array of length N*(N-1)/2

    # --- Summary statistics ---
    stats = {
        "num_points": int(N),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "median": float(np.median(d)),
        "p10": float(np.percentile(d, 10)),
        "p90": float(np.percentile(d, 90)),
        "min": float(d.min()),
        "max": float(d.max()),
    }

    if return_distances:
        return stats, d
    else:
        return stats


if __name__ == "__main__":
    exp_name = sys.argv[1]
    base_path = os.path.join("experiments", exp_name)
    method = sys.argv[2]
    n_samples_to_add = int(sys.argv[3])
    os.makedirs(f"diversity_measure/{exp_name}/{method}", exist_ok=True)

    base_data_path = os.path.join(base_path, "init_data")
    base_model_path = os.path.join(base_path, "init_model")
    x_unlabeled = np.load(os.path.join(base_data_path, "x_unlabeled_total.npy"))
    y_unlabeled = np.load(os.path.join(base_data_path, "y_unlabeled_total.npy"))
    last_model_path = base_path + f"/cycle_19/models/{method}_model"
    x_advs = np.load(base_path + f"/cycle_19/{method}_advs/x_all_advs.npy")
    y_advs = np.load(base_path + f"/cycle_19/{method}_advs/y_all_advs.npy")

    results = []
    for cycle in [0, 19]:
        print(f"starting cycle {cycle}", flush=True)
        indices_to_add = np.load(base_path + f"/cycle_{cycle}/data/{method}_lab_indices.npy")
        if cycle == 0:
            x_advs = x_advs[:n_samples_to_add*10]
            y_advs = y_advs[:n_samples_to_add*10]
        elif cycle == 19:
            indices_to_add = indices_to_add[-n_samples_to_add:]
            x_advs = x_advs[-n_samples_to_add*10:]
            y_advs = y_advs[-n_samples_to_add*10:]
        else:
            print("no model found", flush=True)
            sys.exit(1)
        print(f"x_advs.shape = {x_advs.shape}", flush=True)
        print(f"y_advs.shape = {y_advs.shape}", flush=True)
        x_train = x_unlabeled[indices_to_add]
        y_train = y_unlabeled[indices_to_add]
        if cycle == 0:
            x_temp = np.concatenate((x_train, x_advs), axis=0)
            y_temp = np.concatenate((y_train, y_advs), axis=0)
            model = tf.keras.models.load_model(base_model_path)
            model = retrain_single_model(x_temp, y_temp, model)
        elif cycle == 19:
            model = tf.keras.models.load_model(last_model_path)

        print(f"x_advs.shape: {x_advs.shape}", flush=True)
        results.append(compute_feature_diversity(model, x_advs))

    df = pd.DataFrame(results)
    df.to_csv(f"diversity_measure/{exp_name}/{method}/res.csv", index=False)
