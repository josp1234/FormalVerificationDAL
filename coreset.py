import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def query(
        _model,
        _x_unlabeled,
        _pool_indices,
        n,
        already_selected=None,
        batch_size=256,
        embedding_layer=None,  # can be None (penultimate), int (layer idx), or str (layer name)
        l2_normalize=False,
        rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Resolve which layer to use for embeddings
    if embedding_layer is None:
        if len(_model.layers) < 2:
            raise ValueError("Model must have at least 2 layers (hidden + output).")
        embed_out = _model.layers[-2].output  # penultimate layer
    elif isinstance(embedding_layer, int):
        embed_out = _model.layers[embedding_layer].output
    elif isinstance(embedding_layer, str):
        embed_out = _model.get_layer(embedding_layer).output
    else:
        raise TypeError("embedding_layer must be None, int, or str.")

    embedding_model = Model(inputs=_model.input, outputs=embed_out)

    # Compute embeddings in inference mode
    embeddings = embedding_model.predict(_x_unlabeled[_pool_indices], batch_size=batch_size, verbose=0)
    embeddings = np.asarray(embeddings)

    if embeddings.ndim > 2:
        # Flatten non-vector embeddings (not needed for your simple MLP, but robust)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

    N = embeddings.shape[0]
    if n <= 0:
        return []
    if n > N:
        raise ValueError(f"Requested n={n} but pool size is {N}.")

    selected_indices = []
    if already_selected is not None and len(already_selected) > 0:
        # Start with distances to the first provided selection
        selected_embeddings = embedding_model.predict(np.array(already_selected), batch_size=batch_size, verbose=0)
        selected_embeddings = np.asarray(selected_embeddings)
        min_distances = np.linalg.norm(embeddings - selected_embeddings[0], axis=1)
        # Update using the rest
        for i in range(1, len(already_selected)):
            dist_new = np.linalg.norm(embeddings - selected_embeddings[i], axis=1)
            min_distances = np.minimum(min_distances, dist_new)

    else:
        # Seed with a random point
        first = int(rng.integers(0, N))
        selected_indices.append(first)
        min_distances = np.linalg.norm(embeddings - embeddings[first], axis=1)

    # Greedy k-center
    to_pick = n - len(selected_indices)
    for _ in range(to_pick):
        idx = int(np.argmax(min_distances))
        selected_indices.append(idx)
        dist_new = np.linalg.norm(embeddings - embeddings[idx], axis=1)
        min_distances = np.minimum(min_distances, dist_new)

    return _pool_indices[selected_indices]


if __name__ == "__main__":
    node_list = os.environ.get('SLURM_NODELIST', 'No node information available')
    print(f"Job is running on node: {node_list}", flush=True)
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('trial', type=str, help='Trial name', default='default_value')
    parser.add_argument('cycle', type=int, help='cycle num')
    parser.add_argument('n_samples', type=int, help='Num of samples to acquire')
    parser.add_argument('method', type=str, help='Name of current method')

    # Parse the arguments
    args = parser.parse_args()

    trial_name = args.trial
    cycle_num = args.cycle
    n_samples = args.n_samples
    method = args.method

    base_path = "."
    cur_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num}/"
    prev_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num - 1}/"
    base_model_path = base_path + f"/experiments/trial_{trial_name}/init_model/"
    base_data_path = base_path + f"/experiments/trial_{trial_name}/init_data/"

    x_unlabeled = np.load(base_data_path + "x_unlabeled_total.npy")

    if cycle_num == 0:
        model = tf.keras.models.load_model(base_model_path)
        labeled_samples = None
    else:
        model = tf.keras.models.load_model(prev_exp_path + f"models/{method}_model")
        labeled_indices = np.load(prev_exp_path + f"data/{method}_lab_indices.npy")
        labeled_samples = x_unlabeled[labeled_indices]

    pool_indices = np.load(cur_exp_path + f"data/{method}_pool_indices.npy")
    indices_to_add = query(model, x_unlabeled, pool_indices, n_samples, labeled_samples)
    np.save(cur_exp_path + f"data/{method}_chosen_indices.npy", indices_to_add)

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTotal job runtime: {:.6f} seconds".format(runtime), flush=True)
