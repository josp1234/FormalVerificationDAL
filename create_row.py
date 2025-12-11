import numpy as np
import sys
import time
import argparse
import os
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from deep_fool_sampling import deepfool
import pandas as pd
from gradient_attacks import get_epsilon_FGSM


def get_entropy(logits):
    print("logits: ", logits)
    probs = softmax(logits)
    entropies = entropy(probs)
    print("Probabilities:\n", probs)
    print("Entropy:\n", entropies)
    return entropies


def create_table_entry(img_idx, _model_path, _img, cur_dir_path, _out_path, _method, _num_classes=10):
    method_value = 0

    cur_model = tf.keras.models.load_model(_model_path)
    predictions = cur_model.predict(np.expand_dims(_img, axis=0))[0]
    if predictions is None:
        return -1
    print("predictions:")
    print(predictions)
    top_two_indices = np.argsort(predictions)[-2:]
    model_winner = top_two_indices[1]

    if _method == "entropy":
        print("### Calculating entropy ###", flush=True)
        method_value = get_entropy(predictions)

    elif _method.startswith("DFAL"):
        print("### Calculating DeepFool ###", flush=True)
        adv_sample, min_pert, timeout = deepfool(cur_model, _img, num_classes=_num_classes)
        method_value = np.linalg.norm(min_pert)
        adv_sample = np.clip(adv_sample, 0, 1)
        np.save(cur_dir_path + f"{_method}_advs/idx_{img_idx}_0.npy", adv_sample)

    elif _method.startswith(("FGSM", "FVAAL_rand")):
        print("### Calculating FGSM ###", flush=True)
        method_value = get_epsilon_FGSM(cur_model, _img, model_winner, img_idx, cur_dir_path, _method)
        if method_value is None:
            print("No adversarial example found!", flush=True)
    else:
        print("Method not recognized!", flush=True)
        sys.exit(1)

    data = {"Index": img_idx, "Method": method_value}
    dtypes = {"Index": int, "Method": float}
    result_df = pd.DataFrame([data])
    result_df = result_df.astype(dtypes)

    result_df.to_csv(_out_path, index=False)
    print(f"Done! CSV file created successfully in {_out_path}", flush=True)
    return 0


if __name__ == "__main__":
    node_list = os.environ.get('SLURM_NODELIST', 'No node information available')
    print(f"Job is running on node: {node_list}", flush=True)
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('trial', type=str, help='Trial name', default='default_value')
    parser.add_argument('cycle', type=int, help='cycle num')
    parser.add_argument('job_idx', type=int, help='Job idx')
    parser.add_argument('to_calculate', type=int, help='Num of images to calculate in job')
    parser.add_argument('--methods', type=str, nargs='*', help='Query methods to use', default=["rand"])
    parser.add_argument('--num_classes', type=int, help='num of classes in benchmark', default=10)

    # Parse the arguments
    args = parser.parse_args()

    trial_name = args.trial
    cycle_num = args.cycle
    job_idx_arg = args.job_idx
    to_calculate = args.to_calculate
    query_methods = args.methods

    print(f"trial_name: {trial_name}", flush=True)
    print(f"cycle_num: {cycle_num}", flush=True)
    print(f"job_idx_arg: {job_idx_arg}", flush=True)
    print(f"to_calculate: {to_calculate}", flush=True)
    print(f"query_methods: {query_methods}", flush=True)
    print(f"num_classes: {args.num_classes}", flush=True)

    if cycle_num < 0 or job_idx_arg < 0:
        print(f"Invalid args\ntrial_name: {trial_name}, cycle_num: {cycle_num}, cur_idx: {job_idx_arg}", flush=True)
        sys.exit()

    base_path = "./"
    cur_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num}/"
    base_model_path = base_path + f"/experiments/trial_{trial_name}/init_model/"
    base_data_path = base_path + f"/experiments/trial_{trial_name}/init_data/"

    x_unlabeled = np.load(base_data_path + "x_unlabeled_total.npy")
    y_unlabeled = np.load(base_data_path + "y_unlabeled_total.npy")

    all_done = False
    for job_idx in range(job_idx_arg, job_idx_arg + to_calculate):
        for method in query_methods:
            indices = np.load(cur_exp_path + f"data/{method}_pool_indices.npy")
            num_unlabeled = len(indices)
            idx = indices[job_idx]

            if cycle_num == 0:
                model_path = base_model_path

            else:
                prev_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num - 1}/"
                model_path = prev_exp_path + f"models/{method}_model"

            if job_idx >= num_unlabeled:
                print(
                    f"Current job idx {job_idx} is higher than number of remaining unlabeled samples {num_unlabeled}",
                    flush=True)
                all_done = True
                break

            img = x_unlabeled[idx]
            true_label = y_unlabeled[idx]
            out_path = cur_exp_path + f"rows/{method}_{job_idx}.csv"
            res = create_table_entry(idx, model_path, img, cur_exp_path, out_path, method,
                                     _num_classes=args.num_classes)
            if res != 0:
                print(f"an error occurred when running {method}!", flush=True)
                sys.exit(1)
            print(f"\n#### {method} run ended successfully - job idx {job_idx} ####\n", flush=True)
        if all_done:
            break

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTotal job runtime: {:.6f} seconds".format(runtime), flush=True)
