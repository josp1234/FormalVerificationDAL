import os
import numpy as np
from gradient_attacks import get_advs_FGSM
import tensorflow as tf
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('_trial_name', type=str, help='Trial Name (not used)')
    parser.add_argument('_cycle_idx', type=str, help='cycle idx (not used)')
    parser.add_argument('method', type=str, help='Method to calculate')
    parser.add_argument('model_path', type=str, help='Model path', default='default_value')
    parser.add_argument('data_path', type=str, help='Data path')
    parser.add_argument('cur_dir_path', type=str, help='cur_dir_path')
    parser.add_argument('n_advs', type=int, help='Num of args to calculate')
    parser.add_argument('indices', type=int, nargs='*', help='Indices to calculate')

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    cur_dir_path = args.cur_dir_path
    n_advs = args.n_advs
    method = args.method
    indices = args.indices
    if not method.startswith(('rand', "badge")):
        df = pd.read_csv(cur_dir_path + f"rows/{method}_total.csv", dtype= {"Index": int, "Method": float})

    for idx in indices:
        model = tf.keras.models.load_model(model_path)
        x_unlabeled = np.load(os.path.join(data_path, "x_unlabeled_total.npy"))
        img = x_unlabeled[idx]
        cur_pred = np.argmax(model.predict(np.expand_dims(img, axis=0))[0])
        if not method.startswith(('rand', 'badge')):
            cur_eps = df.loc[df['Index'] == idx, "Method"].values[0]
        else:
            cur_eps = 0.05
        if "bounded" in method:
            bound = 0.1
        else:
            bound = 0.35
        get_advs_FGSM(model, cur_eps, img, cur_pred, idx, cur_dir_path, n_advs, method,
                      start_0=method.startswith(("rand_", "badge")), upper_bound=bound)
