import numpy as np
import sys
import time
import argparse
import os
from maraboupy import Marabou, MarabouCore, MarabouUtils
import subprocess
import re
from main import delete_file
import pandas as pd

MARABOU_PATH = "./Marabou_w_Gurobi/Marabou/build/Marabou" # Set up Marabou path


def exclude_found_example(_network, _adv_example, delta=0.001):
    input_vars = _network.inputVars[0][0]
    for i in np.ndindex(input_vars.shape):
        eq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        eq1.addAddend(1, input_vars[i])
        eq1.setScalar(_adv_example[i] - delta)

        eq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1, input_vars[i])
        eq2.setScalar(_adv_example[i] + delta)

        _network.addDisjunctionConstraint([[eq1], [eq2]])
    return _network


def get_adversarial_inputs(n_advs, network_path, img, model_winner, runner_up, img_idx, cur_dir_path, _method, _epsilon,
                           _adv, marabou_wait=1800):
    if _adv is None:
        advs = []
    else:
        advs = [_adv]
    _epsilon += 0.05
    counter = 0
    iter_limit = 1000
    _network = Marabou.read_tf(network_path, modelType="savedModel_v2")
    output_vars = _network.outputVars[0][0]
    input_vars = _network.inputVars[0][0]
    for i in np.ndindex(input_vars.shape):
        _network.setLowerBound(input_vars[i], max(0, img[i] - _epsilon))
        _network.setUpperBound(input_vars[i], min(1, img[i] + _epsilon))
    _network.addInequality([output_vars[model_winner],
                            output_vars[runner_up]],
                           [1, -1], -0.001)
    if _adv is not None:
        _network = exclude_found_example(_network, _adv)
    while len(advs) < n_advs and counter < iter_limit:
        _cur_ipq_path = cur_dir_path + f"ipqs/{_method}_idx_{img_idx}_adv_{len(advs)}.ipq"
        _sum_path = cur_dir_path + f"sums/{_method}_idx_{img_idx}_adv_{len(advs)}.txt"
        _network.saveQuery(_cur_ipq_path)
        process = subprocess.Popen(
            [MARABOU_PATH, f"--input-query={_cur_ipq_path}", f"--summary-file={_sum_path}", "--verbosity=0", "--milp",
             "--num-workers=10"],
            stdout=subprocess.PIPE, text=True)
        marabou_response = ""
        try:
            process.wait(marabou_wait)
            marabou_response, _ = process.communicate()
        except subprocess.TimeoutExpired:
            print(f"Stopped Marabou query after {marabou_wait} seconds", flush=True)
            delete_file(_cur_ipq_path)
            delete_file(_sum_path)
            return advs

        first_word = None
        with open(_sum_path, 'r') as file:
            first_line = file.readline()
            first_word = first_line.split()[0]

        # remove ipq and summary files
        delete_file(_cur_ipq_path)
        delete_file(_sum_path)

        if first_word == 'sat':
            print(f"found adv #{len(advs)}", flush=True)
            x_values = re.findall(r'x\d+ = ([\d\.\-e]+)', marabou_response)
            adv_numpy = np.array(x_values, dtype=float)
            np.save(cur_dir_path + f"{_method}_advs/idx_{img_idx}_{len(advs)}.npy", adv_numpy)
            advs.append(adv_numpy)
            _network = exclude_found_example(_network, adv_numpy)

        elif first_word == 'unsat':
            print("UNSAT. Increasing epsilon", flush=True)
            _epsilon += 0.05
            if _epsilon > 1:
                print("epsilon too high", flush=True)
                return len(advs)
            _network = Marabou.read_tf(network_path, modelType="savedModel_v2")
            output_vars = _network.outputVars[0][0]
            input_vars = _network.inputVars[0][0]
            for i in np.ndindex(input_vars.shape):
                _network.setLowerBound(input_vars[i], max(0, img[i] - _epsilon))
                _network.setUpperBound(input_vars[i], min(1, img[i] + _epsilon))
            _network.addInequality([output_vars[model_winner],
                                    output_vars[runner_up]],
                                   [1, -1], -0.001)
            for cur_adv in advs:
                _network = exclude_found_example(_network, cur_adv)
        else:
            print("an error occurred while reading summary file", flush=True)
            return -1
        counter += 1
    print("Found advs", flush=True)
    return len(advs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trial', type=str, help='Trial name', default='default_value')
    parser.add_argument('stage', type=int, help='Stage num')
    parser.add_argument('job_idx', type=int, help='job idx')
    parser.add_argument('method', type=str, help='Method name')
    parser.add_argument('indices', type=int, nargs='*', help='Indices to run on', default=None)

    # Parse the arguments
    args = parser.parse_args()
    trial_name = args.trial
    stage_num = args.stage
    indices = args.indices
    method = args.method
    n_advs = int(method.rsplit('_', 1)[-1])
    wait_for = 2700

    print(f"trial_name: {trial_name}", flush=True)
    print(f"stage_num: {stage_num}", flush=True)
    print(f"method: {method}", flush=True)
    print(f"n_advs: {n_advs}", flush=True)
    print(f"indices: {indices}", flush=True)

    if stage_num < 0 or indices is None or len(indices) == 0 or n_advs <= 0:
        print(f"Invalid args\ntrial_name: {trial_name}, stage_num: {stage_num}, indices: {indices}", flush=True)
        sys.exit()

    base_path = "./"
    cur_exp_path = base_path + f"/experiments/trial_{trial_name}/stage_{stage_num}/"
    base_model_path = base_path + f"/experiments/trial_{trial_name}/init_model/"
    base_data_path = base_path + f"/experiments/trial_{trial_name}/init_data/"

    x_unlabeled = np.load(base_data_path + "x_unlabeled_total.npy")

    c = 0

    if stage_num == 0:
        model_path = base_model_path

    else:
        prev_exp_path = base_path + f"/experiments/trial_{trial_name}/stage_{stage_num - 1}/"
        model_path = prev_exp_path + f"models/{method}_model"

    for cur_idx in indices:
        model = Marabou.read_tf(model_path, modelType="savedModel_v2")
        img = x_unlabeled[cur_idx]

        options = Marabou.createOptions(verbosity=0)
        predictions = model.evaluate(np.expand_dims(img, axis=0), useMarabou=False, options=options)[0][0]
        print("predictions:")
        print(predictions)
        top_two_indices = np.argsort(predictions)[-2:]
        model_winner = top_two_indices[1]
        runner_up = top_two_indices[0]
        if method.startswith(("rand_w_samples", "badge_w_samples", "coreset_w_samples")):
            epsilon = 0.01
            adv_sample = None

        else:
            df = pd.read_csv(cur_exp_path + f"rows/{method}_total.csv", dtype={"Index": int, "Method": float})
            epsilon = df.loc[df['Index'] == cur_idx, 'Method'].values[0]
            if epsilon > 0.5:
                epsilon = 0.01
            try:
                adv_sample = np.load(cur_exp_path + f"{method}_advs/idx_{cur_idx}_0.npy")
            except Exception as e:
                epsilon = 0.01
                adv_sample = None

        if n_advs > 1:
            get_adversarial_inputs(n_advs, model_path, img, model_winner, runner_up, cur_idx, cur_exp_path, method,
                                   epsilon, adv_sample)
        else:
            print("adv_sample is None", flush=True)

        print(f"\n#### {method} run ended successfully - cur idx {cur_idx} ####\n", flush=True)
        c += 1
        if c % 2 == 0:
            print(f"\n######## Indices done: {c} / {len(indices)} ########\n", flush=True)


if __name__ == "__main__":
    node_list = os.environ.get('SLURM_NODELIST', 'No node information available')
    print(f"Job is running on node: {node_list}", flush=True)
    start_time = time.time()

    main()

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTotal job runtime: {:.6f} seconds".format(runtime), flush=True)
