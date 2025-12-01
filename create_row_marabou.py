import numpy as np
import sys
import time
import argparse
import os
from maraboupy import Marabou
from main import get_epsilon, get_epsilon_full
import pandas as pd


def create_table_entry(img_idx, _model_path, _img, cur_dir_path, _out_path, _method, _num_classes=10):
    cur_model = Marabou.read_tf(_model_path, modelType="savedModel_v2")
    options = Marabou.createOptions(verbosity=0)
    predictions = cur_model.evaluate(np.expand_dims(_img, axis=0), useMarabou=False, options=options)[0][0]
    if predictions is None:
        return -1
    print("predictions:")
    print(predictions)
    top_two_indices = np.argsort(predictions)[-2:]
    model_winner = top_two_indices[1]
    runner_up = top_two_indices[0]

    print("### Calculating epsilon ###", flush=True)
    _start_time = time.time()
    if _method.startswith("eps_full"):
        method_value = get_epsilon_full(_model_path, _img, model_winner, _num_classes, img_idx, cur_dir_path, _method)
    else:
        method_value = get_epsilon(_model_path, _img, model_winner, runner_up, img_idx, cur_dir_path, _method)
    _end_time = time.time()
    _runtime = _end_time - _start_time
    print(f"total epsilon runtime: {_runtime} seconds", flush=True)

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
    parser.add_argument('stage', type=int, help='Stage num')
    parser.add_argument('job_idx', type=int, help='Job idx')
    parser.add_argument('to_calculate', type=int, help='Num of images to calculate in job')
    parser.add_argument('--methods', type=str, nargs='*', help='Query methods to use', default=["rand"])
    parser.add_argument('--num_classes', type=int, help='num of classes in benchmark', default=10)

    # Parse the arguments
    args = parser.parse_args()

    trial_name = args.trial
    stage_num = args.stage
    job_idx_arg = args.job_idx
    to_calculate = args.to_calculate
    query_methods = args.methods

    print(f"trial_name: {trial_name}", flush=True)
    print(f"stage_num: {stage_num}", flush=True)
    print(f"job_idx_arg: {job_idx_arg}", flush=True)
    print(f"to_calculate: {to_calculate}", flush=True)
    print(f"query_methods: {query_methods}", flush=True)
    print(f"num_classes: {args.num_classes}", flush=True)

    if stage_num < 0 or job_idx_arg < 0:
        print(f"Invalid args\ntrial_name: {trial_name}, stage_num: {stage_num}, cur_idx: {job_idx_arg}", flush=True)
        sys.exit()

    base_path = "."
    cur_exp_path = base_path + f"/experiments/trial_{trial_name}/stage_{stage_num}/"
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

            if stage_num == 0:
                model_path = base_model_path

            else:
                prev_exp_path = base_path + f"/experiments/trial_{trial_name}/stage_{stage_num - 1}/"
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
                sys.exit()
            print(f"\n#### {method} run ended successfully - job idx {job_idx} ####\n", flush=True)
        if all_done:
            break

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTotal job runtime: {:.6f} seconds".format(runtime), flush=True)
