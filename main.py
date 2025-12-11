import os
import time
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
import logging
import my_utils
import subprocess
import argparse
import re
import shutil
import json

MARABOU_PATH = "./Marabou/Marabou/build/Marabou"  # Set up Marabou path


def delete_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        print(f"{filename} does not exist.", flush=True)
    except PermissionError:
        print(f"Permission denied: unable to delete {filename}.", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)


def retrain_single_model(_x_train, _y_train, _x_test, _y_test, _model=None, _epoch_num=1, _num_classes=10,
                         lr=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    _model.fit(x=_x_train, y=_y_train, validation_data=(_x_test, _y_test), epochs=_epoch_num)
    return _model.evaluate(_x_test, _y_test)[1]


def combine_entries(in_path, methods, expected_num_files, _to_delete=True):
    for prefix in methods:
        dfs = []
        paths = []
        c = 0
        pattern = rf"^{re.escape(prefix)}_(\d+)\.csv$"
        for filename in os.listdir(in_path):
            if re.match(pattern, filename):
                filepath = os.path.join(in_path, filename)
                df = pd.read_csv(filepath)
                dfs.append(df)
                paths.append(filepath)
                c += 1
        print(f"found {c} {prefix} files", flush=True)
        if c < expected_num_files:
            print("expected number of files not found!", flush=True)
            sys.exit(1)
        print(f"len(dfs): {len(dfs)}", flush=True)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            out_path = str(os.path.join(in_path, prefix + "_total.csv"))
            combined_df.to_csv(out_path, index=False)
            print(f"Combined {prefix} saved in {out_path}", flush=True)
        else:
            print(f"No files found for {prefix}", flush=True)
        if _to_delete:
            for cur_file_path in paths:
                os.remove(cur_file_path)
            print(f"{prefix} files removed successfully", flush=True)


def retrain_models(trial_name, cur_dir_path, prev_dir_path, first_model_path, first_data_path, _num_samples_to_retrain,
                   _cycle_idx,
                   _x_unlabeled, _y_unlabeled, _x_test, _y_test, methods, _batch_methods=None, _run_rand=True,
                   _rand_adv_methods=None, _num_classes=10, _epoch_num=10, _dataset_name=None,
                   jobs_wait=1800):
    res_dict = dict()
    for _cur_method in methods:
        if prev_dir_path is None:
            _model_path = first_model_path
        else:
            _model_path = prev_dir_path + f"models/{_cur_method}_model"

        print(f"Starting {_cur_method}")
        dtypes = {"Index": int, "Method": float}
        cur_table = pd.read_csv(cur_dir_path + f"rows/{_cur_method}_total.csv", dtype=dtypes)
        if _cur_method == "entropy":
            sorted_df = cur_table.sort_values(by='Method', ascending=True)
            indices_to_add = sorted_df.head(_num_samples_to_retrain)['Index']
        elif _cur_method.startswith(("FGSM", "DFAL", "FVAAL_rand")):
            sorted_df = cur_table.sort_values(by='Method', ascending=True)
            if _cur_method.startswith("FVAAL_rand"):
                indices_to_add = sorted_df.head(int(_num_samples_to_retrain / 2))['Index'].to_numpy()
                unlab_idxs = np.load(cur_dir_path + f"/data/{_cur_method}_pool_indices.npy")
                unlab_idxs = np.setdiff1d(unlab_idxs, indices_to_add)
                if _num_samples_to_retrain / 2 >= len(unlab_idxs):
                    print("Warning: too few samples in rand_unlab_idxs", flush=True)
                    num_samples = len(unlab_idxs)
                else:
                    num_samples = int(_num_samples_to_retrain / 2)
                r_indices_to_add = np.random.choice(unlab_idxs, size=num_samples, replace=False)
                indices_to_add = np.append(indices_to_add, r_indices_to_add)
                print(f"Combined random and FVAAL, indices_to_add shape: {indices_to_add.shape}", flush=True)
            else:
                indices_to_add = sorted_df.head(_num_samples_to_retrain)['Index']
            cur_samples_path = cur_dir_path + f"{_cur_method}_advs/"

            adv_file_paths = []
            # generating adv samples by Marabou
            if "_w_samples_" in _cur_method:
                send_get_Marabou_advs_jobs(_cur_method, indices_to_add, _cycle_idx, 10, trial_name,
                                           cur_dir_path, jobs_wait)

                n_advs = int(_cur_method.rsplit('_', 1)[-1])
                if n_advs > 0:
                    y_advs = []
                    missing_files_d = dict()
                    missing_files_l = []
                    for cur_idx in indices_to_add:
                        for adv_idx in range(n_advs):
                            cur_path = os.path.join(cur_samples_path, f"idx_{cur_idx}_{adv_idx}.npy")
                            if os.path.exists(cur_path):
                                adv_file_paths.append(cur_path)
                                y_advs.append(_y_unlabeled[cur_idx])
                            else:
                                if adv_idx in missing_files_d:
                                    missing_files_d[adv_idx] += 1
                                else:
                                    missing_files_d[adv_idx] = 1
                                missing_files_l.append(cur_path)
                                break
                    if len(missing_files_l) > 0:
                        print("Missing files found!", flush=True)
                        missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.json"
                        print(f"Saving in {missing_summary_path}", flush=True)
                        with open(missing_summary_path, 'w') as f:
                            json.dump(missing_files_d, f)
                        missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.txt"
                        print(f"Saving in {missing_summary_path}", flush=True)
                        with open(missing_summary_path, "w") as f:
                            for path in missing_files_l:
                                f.write(path + "\n")
                        print("saved missing files", flush=True)

                    y_advs = np.array(y_advs)

            # generating adv samples by FGSM
            elif _cur_method.startswith("DFAL_FGSM") or (
                    _cur_method.startswith("FGSM") and "w_samples" not in _cur_method):
                n_advs = int(_cur_method.rsplit('_', 1)[-1])
                if n_advs > 0:
                    send_get_FGSM_advs_jobs(trial_name, _cycle_idx, _model_path, first_data_path, cur_dir_path,
                                            n_advs - 1, _cur_method, indices_to_add, jobs_wait)
                    y_advs = []
                    missing_files_d = dict()
                    missing_files_l = []
                    for cur_idx in indices_to_add:
                        temp_count = 0
                        for adv_idx in range(n_advs):
                            cur_path = os.path.join(cur_samples_path, f"idx_{cur_idx}_{adv_idx}.npy")
                            if os.path.exists(cur_path):
                                adv_file_paths.append(cur_path)
                                y_advs.append(_y_unlabeled[cur_idx])
                                temp_count += 1
                            else:
                                missing_files_l.append(cur_path)
                        if temp_count < n_advs:
                            if temp_count in missing_files_d:
                                missing_files_d[temp_count] += 1
                            else:
                                missing_files_d[temp_count] = 1
                    if len(missing_files_l) > 0:
                        print("Missing files found!", flush=True)
                        missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.json"
                        print(f"Saving in {missing_summary_path}", flush=True)
                        with open(missing_summary_path, 'w') as f:
                            json.dump(missing_files_d, f)
                        missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.txt"
                        print(f"Saving in {missing_summary_path}", flush=True)
                        with open(missing_summary_path, "w") as f:
                            for path in missing_files_l:
                                f.write(path + "\n")
                        print("saved missing files", flush=True)

                    y_advs = np.array(y_advs)

            else:
                adv_file_paths = [
                    os.path.join(cur_samples_path, f"idx_{num}_0.npy") for num in indices_to_add
                    if os.path.exists(os.path.join(cur_samples_path, f"idx_{num}_0.npy"))
                ]
                y_advs = _y_unlabeled[indices_to_add]

            x_advs = None
            if not os.path.exists(cur_samples_path + "x_all_advs.npy") and len(adv_file_paths) > 0:

                with open(cur_samples_path + "paths.txt", "w") as file:
                    for path in adv_file_paths:
                        file.write(path + "\n")
                    print("saved adv_file_paths", flush=True)

                # Load the filtered .npy files into a list of arrays
                arrays = [np.load(file) for file in adv_file_paths]
                # Combine all arrays into a single NumPy array (rows = number of files, columns = array shape)
                x_advs = np.stack(arrays, axis=0)
        else:
            print(f"Error! method name {_cur_method} is unidentified")
            return

        x_to_add = _x_unlabeled[indices_to_add]
        y_to_add = _y_unlabeled[indices_to_add]

        if _cur_method.startswith(("FGSM", "DFAL", "FVAAL_rand")) and _cur_method not in ["FGSM_0", "DFAL_0"]:
            cur_samples_path = cur_dir_path + f"{_cur_method}_advs/"
            if x_advs is not None:
                print(f"x_to_add.shape: {x_to_add.shape}", flush=True)
                print(f"x_advs.shape: {x_advs.shape}", flush=True)
                x_to_add = np.concatenate((x_to_add, x_advs), axis=0)
                y_to_add = np.concatenate((y_to_add, y_advs), axis=0)
                if prev_dir_path is None:
                    x_all_advs = x_advs
                    y_all_advs = y_advs
                else:
                    prev_samples_path = prev_dir_path + f"{_cur_method}_advs/"
                    x_all_advs = np.load(prev_samples_path + "x_all_advs.npy")
                    x_all_advs = np.concatenate((x_all_advs, x_advs), axis=0)
                    y_all_advs = np.load(prev_samples_path + "y_all_advs.npy")
                    y_all_advs = np.concatenate((y_all_advs, y_advs), axis=0)
                np.save(cur_samples_path + "x_all_advs.npy", x_all_advs)
                np.save(cur_samples_path + "y_all_advs.npy", y_all_advs)
            else:
                x_all_advs = np.load(cur_samples_path + "x_all_advs.npy")
                y_all_advs = np.load(cur_samples_path + "y_all_advs.npy")
                x_to_add = x_all_advs[-_num_samples_to_retrain:]
                y_to_add = y_all_advs[-_num_samples_to_retrain:]
        print(f"x_to_add shape: {x_to_add.shape}")
        print(f"y_to_add shape: {y_to_add.shape}")
        if isinstance(indices_to_add, (pd.Series, pd.DataFrame)):
            indices_to_add = indices_to_add.to_numpy()
        if prev_dir_path is None:
            cur_unlab_indices = np.arange(len(_y_unlabeled))
            labeled_indices = indices_to_add
        else:
            cur_unlab_indices = np.load(prev_dir_path + f"data/{_cur_method}_unlab_indices.npy")
            labeled_indices = np.load(prev_dir_path + f"data/{_cur_method}_lab_indices.npy")
            labeled_indices = np.concatenate((labeled_indices, indices_to_add))

        updated_unlab_indices = np.setdiff1d(cur_unlab_indices, indices_to_add)
        print(f"Saving {_cur_method} new indices of shape {updated_unlab_indices.shape}")
        np.save(cur_dir_path + f"data/{_cur_method}_unlab_indices.npy", updated_unlab_indices)
        np.save(cur_dir_path + f"data/{_cur_method}_lab_indices.npy", labeled_indices)
        # retraining from scratch
        print(f"retraining from scratch - {_cur_method}", flush=True)
        init_model = tf.keras.models.load_model(first_model_path)
        if _cur_method.startswith(("FGSM", "DFAL", "FVAAL_rand")) and _cur_method not in ["FGSM_0", "DFAL_0"]:
            print(f"x_all_advs.shape {x_all_advs.shape}")
            print(f"labeled_indices.shape {labeled_indices.shape}")
            print(f"y_all_advs.shape {y_all_advs.shape}")
            x_train_labeled = np.concatenate((_x_unlabeled[labeled_indices], x_all_advs))
            y_train_labeled = np.concatenate((_y_unlabeled[labeled_indices], y_all_advs))
        else:
            x_train_labeled = _x_unlabeled[labeled_indices]
            y_train_labeled = _y_unlabeled[labeled_indices]
        print(f"x_train_labeled.shape: {x_train_labeled.shape}", flush=True)
        print(f"y_train_labeled.shape: {y_train_labeled.shape}", flush=True)
        res_dict[_cur_method] = [
            retrain_single_model(x_train_labeled, y_train_labeled, _x_test, _y_test, _model=init_model,
                                 _epoch_num=_epoch_num)]
        init_model.save(cur_dir_path + f"models/{_cur_method}_model", save_format='tf')

    # run batch methods and retrain
    if _batch_methods is not None and len(_batch_methods) > 0:
        for _cur_method in _batch_methods:
            print(f"Running {_cur_method}...")
            if prev_dir_path is None:
                cur_model = tf.keras.models.load_model(first_model_path)
                cur_unlab_indices = np.arange((len(_y_unlabeled)))
            else:
                cur_model = tf.keras.models.load_model(prev_dir_path + f"models/{_cur_method}_model")
                cur_unlab_indices = np.load(prev_dir_path + f"data/{_cur_method}_unlab_indices.npy")
            send_batch_job(trial_name, _cycle_idx, _num_samples_to_retrain, _cur_method)
            indices_to_add = np.load(cur_dir_path + f"data/{_cur_method}_chosen_indices.npy")

            if "_w_samples_" in _cur_method or "FGSM" in _cur_method:
                n_advs = int(_cur_method.rsplit('_', 1)[-1])
                cur_samples_path = cur_dir_path + f"{_cur_method}_advs/"

                if "_w_samples_" in _cur_method:
                    send_get_Marabou_advs_jobs(_cur_method, indices_to_add, _cycle_idx, 10, trial_name,
                                               cur_dir_path, jobs_wait)
                else:
                    _model_path = first_model_path if prev_dir_path is None else prev_dir_path + f"models/{_cur_method}_model"
                    send_get_FGSM_advs_jobs(trial_name, _cycle_idx, _model_path, first_data_path, cur_dir_path,
                                            n_advs, _cur_method, indices_to_add, jobs_wait)

                adv_file_paths = []
                y_advs = []
                missing_files_d = dict()
                missing_files_l = []
                for cur_idx in indices_to_add:
                    for adv_idx in range(n_advs):
                        cur_path = os.path.join(cur_samples_path, f"idx_{cur_idx}_{adv_idx}.npy")
                        if os.path.exists(cur_path):
                            adv_file_paths.append(cur_path)
                            y_advs.append(_y_unlabeled[cur_idx])
                        else:
                            if adv_idx in missing_files_d:
                                missing_files_d[adv_idx] += 1
                            else:
                                missing_files_d[adv_idx] = 1
                            missing_files_l.append(cur_path)
                            break
                if len(missing_files_l) > 0:
                    print("Missing files found!", flush=True)
                    missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.json"
                    print(f"Saving in {missing_summary_path}", flush=True)
                    with open(missing_summary_path, 'w') as f:
                        json.dump(missing_files_d, f)
                    missing_summary_path = cur_dir_path + f"{_cur_method}_missing_files.txt"
                    print(f"Saving in {missing_summary_path}", flush=True)
                    with open(missing_summary_path, "w") as f:
                        for path in missing_files_l:
                            f.write(path + "\n")
                    print("saved missing files", flush=True)

                x_advs = None
                if not os.path.exists(cur_samples_path + "x_all_advs.npy"):

                    with open(cur_samples_path + "paths.txt", "w") as file:
                        for path in adv_file_paths:
                            file.write(path + "\n")
                        print("saved adv_file_paths", flush=True)

                    # Load the filtered .npy files into a list of arrays
                    arrays = [np.load(file) for file in adv_file_paths]
                    # Combine all arrays into a single NumPy array (rows = number of files, columns = array shape)
                    x_advs = np.stack(arrays, axis=0)
                y_advs = np.array(y_advs)

            x_to_add = _x_unlabeled[indices_to_add]
            y_to_add = _y_unlabeled[indices_to_add]

            if "w_samples_" in _cur_method or "FGSM" in _cur_method:
                cur_samples_path = cur_dir_path + f"{_cur_method}_advs/"
                if x_advs is not None:
                    print(f"x_to_add.shape: {x_to_add.shape}", flush=True)
                    print(f"x_advs.shape: {x_advs.shape}", flush=True)
                    x_to_add = np.concatenate((x_to_add, x_advs), axis=0)
                    y_to_add = np.concatenate((y_to_add, y_advs), axis=0)
                    if prev_dir_path is None:
                        x_all_advs = x_advs
                        y_all_advs = y_advs
                    else:
                        prev_samples_path = prev_dir_path + f"{_cur_method}_advs/"
                        x_all_advs = np.load(prev_samples_path + "x_all_advs.npy")
                        x_all_advs = np.concatenate((x_all_advs, x_advs), axis=0)
                        y_all_advs = np.load(prev_samples_path + "y_all_advs.npy")
                        y_all_advs = np.concatenate((y_all_advs, y_advs), axis=0)
                    np.save(cur_samples_path + "x_all_advs.npy", x_all_advs)
                    np.save(cur_samples_path + "y_all_advs.npy", y_all_advs)
                else:
                    x_all_advs = np.load(cur_samples_path + "x_all_advs.npy")
                    y_all_advs = np.load(cur_samples_path + "y_all_advs.npy")
                    x_to_add = x_all_advs[-_num_samples_to_retrain:]
                    y_to_add = y_all_advs[-_num_samples_to_retrain:]

            print(f"x_to_add shape: {x_to_add.shape}")
            print(f"y_to_add shape: {y_to_add.shape}")
            updated_unlab_indices = np.setdiff1d(cur_unlab_indices, indices_to_add)
            if prev_dir_path is not None:
                labeled_indices = np.load(prev_dir_path + f"data/{_cur_method}_lab_indices.npy")
                labeled_indices = np.concatenate((labeled_indices, indices_to_add))
            else:
                labeled_indices = indices_to_add

            print(f"Saving {_cur_method} new indices of shape {updated_unlab_indices.shape}")
            np.save(cur_dir_path + f"data/{_cur_method}_unlab_indices.npy", updated_unlab_indices)
            np.save(cur_dir_path + f"data/{_cur_method}_lab_indices.npy", labeled_indices)
            print(f"retraining from scratch - {_cur_method}", flush=True)
            init_model = tf.keras.models.load_model(first_model_path)
            if "w_samples_" in _cur_method or "FGSM" in _cur_method:
                x_train_labeled = np.concatenate((_x_unlabeled[labeled_indices], x_all_advs))
                y_train_labeled = np.concatenate((_y_unlabeled[labeled_indices], y_all_advs))
            else:
                x_train_labeled = _x_unlabeled[labeled_indices]
                y_train_labeled = _y_unlabeled[labeled_indices]
            res_dict[_cur_method] = [
                retrain_single_model(x_train_labeled, y_train_labeled, _x_test, _y_test, _model=init_model,
                                     _epoch_num=_epoch_num)]
            init_model.save(cur_dir_path + f"models/{_cur_method}_model", save_format='tf')

    # run rand and retrain
    if _run_rand:
        cur_r_method = "rand"
        num_samples = _num_samples_to_retrain
        print(f"Running {cur_r_method}...")
        if prev_dir_path is None:
            model_rand = tf.keras.models.load_model(first_model_path)
            rand_unlab_idxs = np.arange(_x_unlabeled.shape[0])
        else:
            model_rand = tf.keras.models.load_model(prev_dir_path + f"models/{cur_r_method}_model")
            rand_unlab_idxs = np.load(prev_dir_path + f"/data/{cur_r_method}_unlab_indices.npy")
        if num_samples >= len(rand_unlab_idxs):
            print("Warning: too few samples in rand_unlab_idxs", flush=True)
            num_samples = len(rand_unlab_idxs)
        indices_to_add = np.random.choice(rand_unlab_idxs, size=num_samples, replace=False)
        x_to_add = _x_unlabeled[indices_to_add]
        y_to_add = _y_unlabeled[indices_to_add]
        print(f"x_to_add shape: {x_to_add.shape}")
        print(f"y_to_add shape: {y_to_add.shape}")
        updated_unlab_indices = np.setdiff1d(rand_unlab_idxs, indices_to_add)
        if prev_dir_path is not None:
            labeled_indices = np.load(prev_dir_path + f"data/{cur_r_method}_lab_indices.npy")
            labeled_indices = np.concatenate((labeled_indices, indices_to_add))
        else:
            labeled_indices = indices_to_add

        print(f"Saving rand new indices of shape {updated_unlab_indices.shape}")
        np.save(cur_dir_path + f"data/{cur_r_method}_unlab_indices.npy", updated_unlab_indices)
        np.save(cur_dir_path + f"data/{cur_r_method}_lab_indices.npy", labeled_indices)

        # retraining from scratch - rand
        print(f"retraining from scratch - {cur_r_method}", flush=True)
        init_model = tf.keras.models.load_model(first_model_path)
        x_train_labeled = _x_unlabeled[labeled_indices]
        y_train_labeled = _y_unlabeled[labeled_indices]
        res_dict[cur_r_method] = [
            retrain_single_model(x_train_labeled, y_train_labeled, _x_test, _y_test, _model=init_model,
                                 _epoch_num=_epoch_num)]
        init_model.save(cur_dir_path + f"models/{cur_r_method}_model", save_format='tf')

    # run rand with samples and retrain
    if _rand_adv_methods is not None and len(_rand_adv_methods) > 0:
        for cur_r_method in _rand_adv_methods:
            print(f"Running {cur_r_method}...")
            if prev_dir_path is None:
                rand_model = tf.keras.models.load_model(first_model_path)
                cur_unlab_idxs = np.arange(_x_unlabeled.shape[0])
            else:
                rand_model = tf.keras.models.load_model(prev_dir_path + f"models/{cur_r_method}_model")
                cur_unlab_idxs = np.load(
                    prev_dir_path + f"/data/{cur_r_method}_unlab_indices.npy")
            indices_to_add = np.random.choice(cur_unlab_idxs, size=_num_samples_to_retrain,
                                              replace=False)
            x_to_add = _x_unlabeled[indices_to_add]
            y_to_add = _y_unlabeled[indices_to_add]
            print(f"x_to_add shape: {x_to_add.shape}")
            print(f"y_to_add shape: {y_to_add.shape}")
            updated_unlab_indices = np.setdiff1d(cur_unlab_idxs, indices_to_add)
            if prev_dir_path is not None:
                labeled_indices = np.load(prev_dir_path + f"data/{cur_r_method}_lab_indices.npy")
                labeled_indices = np.concatenate((labeled_indices, indices_to_add))
            else:
                labeled_indices = indices_to_add

            print(f"Saving {cur_r_method} new indices", flush=True)
            np.save(cur_dir_path + f"data/{cur_r_method}_unlab_indices.npy", updated_unlab_indices)
            np.save(cur_dir_path + f"data/{cur_r_method}_lab_indices.npy", labeled_indices)
            _n_advs_to_calc = int(cur_r_method.rsplit('_', 1)[-1])
            if cur_r_method.startswith("rand_w_samples"):
                send_get_Marabou_advs_jobs(cur_r_method, indices_to_add, _cycle_idx,
                                           10, trial_name, cur_dir_path, jobs_wait)
            elif cur_r_method.startswith("rand_FGSM"):
                _model_path = first_model_path if prev_dir_path is None else prev_dir_path + f"models/{cur_r_method}_model"
                send_get_FGSM_advs_jobs(trial_name, _cycle_idx, _model_path, first_data_path, cur_dir_path,
                                        _n_advs_to_calc, cur_r_method, indices_to_add, jobs_wait)
            else:
                print("Unknown method: ", cur_r_method, flush=True)
                sys.exit(1)

            all_npy_files = []
            missing_files_d = dict()
            missing_files_l = []
            y_advs = []
            for cur_idx in indices_to_add:
                for adv_idx in range(_n_advs_to_calc):
                    cur_file_path = cur_dir_path + f"{cur_r_method}_advs/idx_{cur_idx}_{adv_idx}.npy"
                    if os.path.exists(cur_file_path):
                        all_npy_files.append(np.load(cur_file_path))
                        y_advs.append(_y_unlabeled[cur_idx])
                    else:
                        if adv_idx in missing_files_d:
                            missing_files_d[adv_idx] += 1
                        else:
                            missing_files_d[adv_idx] = 1
                        missing_files_l.append(cur_file_path)
            if len(missing_files_l) > 0:
                print("Missing files found!", flush=True)
                missing_summary_path = cur_dir_path + f"{cur_r_method}_missing_files.json"
                print(f"Saving in {missing_summary_path}", flush=True)
                with open(missing_summary_path, 'w') as f:
                    json.dump(missing_files_d, f)
                missing_summary_path = cur_dir_path + f"{cur_r_method}_missing_files.txt"
                print(f"Saving in {missing_summary_path}", flush=True)
                with open(missing_summary_path, "w") as f:
                    for path in missing_files_l:
                        f.write(path + "\n")
                print("saved missing files", flush=True)

            x_advs = np.stack(all_npy_files, axis=0)
            cur_samples_path = cur_dir_path + f"{cur_r_method}_advs/"

            if x_advs is not None:
                print(f"x_to_add.shape: {x_to_add.shape}", flush=True)
                print(f"x_advs.shape: {x_advs.shape}", flush=True)
                x_to_add = np.concatenate((x_to_add, x_advs), axis=0)
                y_to_add = np.concatenate((y_to_add, y_advs), axis=0)

                if prev_dir_path is None:
                    x_all_advs = x_advs
                    y_all_advs = y_advs
                else:
                    prev_samples_path = prev_dir_path + f"{cur_r_method}_advs/"
                    x_all_advs = np.load(prev_samples_path + "x_all_advs.npy")
                    x_all_advs = np.concatenate((x_all_advs, x_advs), axis=0)
                    y_all_advs = np.load(prev_samples_path + "y_all_advs.npy")
                    y_all_advs = np.concatenate((y_all_advs, y_advs), axis=0)
                np.save(cur_samples_path + "x_all_advs.npy", x_all_advs)
                np.save(cur_samples_path + "y_all_advs.npy", y_all_advs)
            else:
                x_all_advs = np.load(cur_samples_path + "x_all_advs.npy")
                y_all_advs = np.load(cur_samples_path + "y_all_advs.npy")
                x_to_add = x_all_advs[-_num_samples_to_retrain:]
                y_to_add = y_all_advs[-_num_samples_to_retrain:]

            # retraining from scratch for "rand + samples" method
            print(f"retraining from scratch - {cur_r_method}", flush=True)
            init_model = tf.keras.models.load_model(first_model_path)
            x_train_labeled = np.concatenate((_x_unlabeled[labeled_indices], x_all_advs))
            y_train_labeled = np.concatenate((_y_unlabeled[labeled_indices], y_all_advs))
            res_dict[cur_r_method] = [
                retrain_single_model(x_train_labeled, y_train_labeled, _x_test, _y_test, _model=init_model,
                                     _epoch_num=_epoch_num)]
            init_model.save(cur_dir_path + f"models/{cur_r_method}_model", save_format='tf')

    if not os.path.exists(cur_dir_path + "total_retrained.csv"):
        retrained_out_path = cur_dir_path + "total_retrained.csv"
    else:
        retrained_out_path = cur_dir_path + f"total_retrained_{int(time.time())}.csv"
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(retrained_out_path, index=False)


def is_job_running(_job_id):
    """Checks if the job with the given job ID is still running."""
    squeue = subprocess.Popen(['squeue', '--job', _job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = squeue.communicate()[0].decode('utf-8')
    return _job_id in output


def send_batch_job(_trial_name, _cycle_idx, _num_samples_to_retrain, _cur_method):
    print(f"Sending batch job - {_cur_method}......", flush=True)
    if _cur_method.startswith("badge"):
        sbatch_file = "badgeSampling.sbatch"
    elif _cur_method.startswith("coreset"):
        sbatch_file = "coresetSampling.sbatch"
    else:
        print(f"Unknown method - {_cur_method}", flush=True)
        sys.exit(1)
    proc = subprocess.Popen(
        ["bash", sbatch_file, _trial_name, str(_cycle_idx), str(_num_samples_to_retrain), _cur_method],
        stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()

    # Extract job ID from sbatch output (usually the last part of the output)
    job_id = stdout.decode('utf-8').strip().split()[-1]

    # Wait for job to finish
    while True:
        if not is_job_running(job_id):
            print("Job has completed.", flush=True)
            break

        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S", current_time)
        print(f"{formatted_time} - job is still running", flush=True)
        time.sleep(120)  # Wait for x seconds before checking again


def send_get_FGSM_advs_jobs(_trial_name, _cycle_idx, _model_path, _data_path, cur_dir_path, n_advs, _method, indices,
                            jobs_wait):
    print(f"Sending FGSM adv job for {_method}, len(indices)={len(indices)}......", flush=True)
    proc = subprocess.Popen(
        ["bash", "getFGSMAdvs.sbatch", _trial_name, str(_cycle_idx), _method, _model_path, _data_path, cur_dir_path,
         str(n_advs)] + [
            str(x) for x in indices],
        stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()

    # Extract job ID from sbatch output (usually the last part of the output)
    job_id = stdout.decode('utf-8').strip().split()[-1]

    # Wait for job to finish
    while True:
        if not is_job_running(job_id):
            print("All jobs have completed.", flush=True)
            break

        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S", current_time)
        print(f"{formatted_time} - job is still running", flush=True)
        time.sleep(jobs_wait)  # Wait for x seconds before checking again


def send_get_Marabou_advs_jobs(_method, indices, _cycle_idx, _num_idxs_in_job, _trial_name, _cur_exp_path, jobs_wait):
    print(f"Sending adv jobs, len(indices)={len(indices)}......", flush=True)
    job_ids = set()
    for i in range(0, len(indices), _num_idxs_in_job):
        cur_indices = indices[i:i + _num_idxs_in_job]
        proc = subprocess.Popen(
            ["bash", "getAdvs.sbatch", str(_trial_name), str(_cycle_idx), str(i), str(_method)] + [str(x) for x in
                                                                                                   cur_indices],
            stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()

        # Extract job ID from sbatch output (usually the last part of the output)
        job_id = stdout.decode('utf-8').strip().split()[-1]
        job_ids.add(job_id)

    # Wait for all jobs to finish
    while True:
        job_ids = set([job_id for job_id in job_ids if is_job_running(job_id)])

        if not job_ids:
            print("All jobs have completed.", flush=True)
            break

        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S", current_time)
        print(f"{formatted_time} - {len(job_ids)} jobs are still running", flush=True)
        time.sleep(jobs_wait)  # Wait for x seconds before checking again


def send_create_row_jobs(_total_num_samples, _cycle_num, _samples_to_retrain, _num_idxs_in_job, _trial_name,
                         _query_methods, _cur_exp_path, _num_classes=10, _num_jobs=110,
                         _specific_idxs=None, running_marabou=False, _jobs_wait=1800):
    if _specific_idxs is None or _specific_idxs[0] != _cycle_num:
        print(f"Sending jobs......", flush=True)
        all_idxs = set(range(0, _total_num_samples, _num_idxs_in_job))

    else:
        print(f"Sending {len(_specific_idxs[1])} jobs......", flush=True)
        all_idxs = _specific_idxs[1]

    job_ids = set()
    sbatch_file = "createRow.sbatch"
    # Submit each sbatch file and collect the job IDs
    for i in range(_num_jobs):
        if not all_idxs:
            break
        cur_job_idx = all_idxs.pop()
        proc = subprocess.Popen(
            ["bash", sbatch_file, str(_trial_name), str(_cycle_num), str(cur_job_idx), str(_num_idxs_in_job),
             "--methods"] + _query_methods + ["--num_classes", str(_num_classes)], stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()

        # Extract job ID from sbatch output (usually the last part of the output)
        job_id = stdout.decode('utf-8').strip().split()[-1]
        job_ids.add(job_id)

    # Wait for all jobs to finish
    while True:
        job_ids = set([job_id for job_id in job_ids if is_job_running(job_id)])

        if not job_ids and not all_idxs:
            print("All jobs have completed.", flush=True)
            break
        if len(job_ids) < _num_jobs and len(all_idxs) > 0:
            jobs_to_add = min(_num_jobs - len(job_ids), len(all_idxs))
            for i in range(jobs_to_add):
                cur_job_idx = all_idxs.pop()
                proc = subprocess.Popen(
                    ["bash", sbatch_file, str(_trial_name), str(_cycle_num), str(cur_job_idx),
                     str(_num_idxs_in_job), "--methods"] + _query_methods + ["--num_classes", str(_num_classes)],
                    stdout=subprocess.PIPE)
                stdout, _ = proc.communicate()

                # Extract job ID from sbatch output (usually the last part of the output)
                job_id = stdout.decode('utf-8').strip().split()[-1]
                job_ids.add(job_id)
            current_time = time.localtime()
            formatted_time = time.strftime("%H:%M:%S", current_time)
            print(f"{formatted_time} - added {jobs_to_add} jobs", flush=True)

        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S", current_time)
        print(f"{formatted_time} - {len(job_ids)} jobs still running, {len(all_idxs)} jobs pending\n", flush=True)
        time.sleep(_jobs_wait)  # Wait for x seconds before checking again


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial', type=str, help='Trial name (e.g., MNIST_nov_9_50samples_seed0)', default='trial_name')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=["MNIST", "fashionMNIST", "CIFAR10"])
    parser.add_argument('--delete', type=str, help='Whether to delete stdout files, previous models and advs',
                        default="True")
    parser.add_argument('--methods', type=str, nargs='*', help='Query methods to use', default=["rand"])
    parser.add_argument('--emb_size', type=int, help='MLP embedding size', default=128)
    parser.add_argument('--cycles', type=int, help='Num of cycles to run', default=20)
    parser.add_argument('--num_samples', type=int, help='Num of samples to acquire each iteration', default=-1)
    parser.add_argument('--start_cycle', type=int, help='Cycle to start in (optional)', default=0)
    parser.add_argument('--retrain_only', type=str, help='Whether to only retrain', default="False")
    parser.add_argument('--lr', type=float, help='learning rate for retraining', default=0.001)
    parser.add_argument('--pool_size', type=int, help='Num of samples in pool chosen each iteration', default=10000)
    parser.add_argument('--seed', type=int, help='Random seed for numpy', default=0)
    parser.add_argument('--epoch_num', type=int, help='Num of epochs for training', default=10)

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    trial_name = args.trial
    dataset_name = args.dataset
    query_methods = args.methods
    emb_size = args.emb_size
    start_cycle = args.start_cycle
    num_cycles = args.cycles
    np_seed = args.seed
    np.random.seed(np_seed)
    retrain_only = False if args.retrain_only == "False" else True
    to_delete = False if args.delete == "False" else True
    learning_rate = args.lr
    epoch_num = args.epoch_num
    initial_labeled_size = args.num_samples
    samples_to_retrain = args.num_samples

    # Identifying batch method (BADGE, CoreSet, etc.)
    batch_methods = [s for s in query_methods if s.startswith(("badge", "coreset"))]
    if len(batch_methods) > 0:
        for method in batch_methods:
            query_methods.remove(method)

    # Identifying rand + adv methods
    rand_adv_methods = [s for s in query_methods if s.startswith(("rand_FGSM", "rand_w_samples"))]
    if len(rand_adv_methods) > 0:
        for method in rand_adv_methods:
            query_methods.remove(method)

    # Identifying rand methods
    if "rand" in query_methods:
        run_rand = True
        query_methods.remove("rand")
    else:
        run_rand = False

    base_path = "./"
    model_path = base_path + f"/experiments/trial_{trial_name}/init_model/"
    data_path = base_path + f"/experiments/trial_{trial_name}/init_data/"

    # train initial model and save data
    num_classes = 10
    if start_cycle == 0 and not retrain_only and not os.path.exists(model_path):
        my_utils.save_data(initial_labeled_size, out_path=data_path, dataset_name=dataset_name)
        my_utils.load_data_and_train_model(model_path, data_path, num_classes, num_neurons=emb_size,
                                           epoch_num=epoch_num)
    x_unlabeled = np.load(data_path + "x_unlabeled_total.npy")
    y_unlabeled = np.load(data_path + "y_unlabeled_total.npy")
    num_unlabeled = len(y_unlabeled)
    pool_size = min(args.pool_size, num_unlabeled)
    x_test = np.load(data_path + "x_test.npy")
    y_test = np.load(data_path + "y_test.npy")

    # create dirs for stdouts
    os.makedirs(base_path + f"/stdouts/create_row_trial_{trial_name}", exist_ok=True)
    os.makedirs(base_path + f"/stdouts/get_advs_trial_{trial_name}", exist_ok=True)

    # other args
    num_idxs_in_job = 1000
    jobs_wait = 300
    # specific_indices usage: (cycle_num, idxs)
    specific_indices = None

    log_file = base_path + f"/experiments/trial_{trial_name}/configuration.txt"
    if os.path.exists(log_file):
        log_file = base_path + f"/experiments/trial_{trial_name}/configuration_{int(time.time())}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(stream=sys.stdout),
        ],
        force=True,
    )

    print(f"\n###### Starting experiment ######\n\n", flush=True)
    logging.info(f"trial_name: {trial_name} ")
    logging.info(f"initial_labeled_size: {initial_labeled_size} ")
    logging.info(f"embedding size: {emb_size} ")
    logging.info(f"samples_to_retrain: {samples_to_retrain} ")
    logging.info(f"start_cycle: {start_cycle} ")
    logging.info(f"num_cycles: {num_cycles} ")
    logging.info(f"num_idxs_in_job: {num_idxs_in_job} ")
    logging.info(f"jobs_wait: {jobs_wait} ")
    logging.info(f"epoch_num for initial training: {epoch_num} ")
    logging.info(f"query methods: {query_methods} ")
    logging.info(f"batch methods: {batch_methods} ")
    logging.info(f"run_rand: {run_rand} ")
    logging.info(f"extended_rand_methods: {rand_adv_methods} ")
    logging.info(f"Delete files: {to_delete} ")
    logging.info(f"Retrain only: {retrain_only} ")
    logging.info(f"learning_rate: {learning_rate} ")
    logging.info(f"pool_size: {pool_size} ")
    logging.info(f"seed: {np_seed} ")

    for cycle_num in range(start_cycle, num_cycles):
        print(f"\n###### Starting cycle {cycle_num} ######\n\n", flush=True)

        if cycle_num == 0:
            prev_exp_path = None
        else:
            prev_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num - 1}/"

        cur_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num}/"
        os.makedirs(cur_exp_path, exist_ok=True)
        os.makedirs(cur_exp_path + "rows", exist_ok=True)
        os.makedirs(cur_exp_path + "models", exist_ok=True)
        os.makedirs(cur_exp_path + "data", exist_ok=True)
        os.makedirs(cur_exp_path + "ipqs", exist_ok=True)
        os.makedirs(cur_exp_path + "sums", exist_ok=True)

        for cur_method in query_methods:
            if cur_method.startswith(("FGSM", "DFAL", "FVAAL_rand")):
                os.makedirs(cur_exp_path + f"{cur_method}_advs", exist_ok=True)
        if len(rand_adv_methods) > 0:
            for method in rand_adv_methods:
                if method.startswith(("rand_w_samples", "rand_FGSM")):
                    os.makedirs(cur_exp_path + f"{method}_advs", exist_ok=True)
        if len(batch_methods) > 0:
            for method in batch_methods:
                if "w_samples" in method or "FGSM" in method:
                    os.makedirs(cur_exp_path + f"{method}_advs", exist_ok=True)

        # save pool indices
        if len(query_methods) > 0:
            for method in query_methods:
                if not os.path.exists(cur_exp_path + f"data/{method}_pool_indices.npy"):
                    if cycle_num == 0:
                        indices = np.arange(num_unlabeled)
                    else:
                        indices = np.load(prev_exp_path + f"data/{method}_unlab_indices.npy")
                    my_utils.save_random_samples(indices, pool_size, cur_exp_path + "data/", method)
        if len(batch_methods) > 0:
            for method in batch_methods:
                if not os.path.exists(cur_exp_path + f"data/{method}_pool_indices.npy"):
                    if cycle_num == 0:
                        indices = np.arange(num_unlabeled)
                    else:
                        indices = np.load(prev_exp_path + f"data/{method}_unlab_indices.npy")
                    my_utils.save_random_samples(indices, pool_size, cur_exp_path + "data/", method)

        # create table rows (rank samples according to each DAL method)
        if len(query_methods) > 0 and not retrain_only:
            reg_methods = [s for s in query_methods if not s.startswith("eps_")]
            if len(reg_methods) > 0:
                print(f"Regular methods: {reg_methods} ", flush=True)
                send_create_row_jobs(pool_size, cycle_num, samples_to_retrain, num_idxs_in_job, trial_name,
                                     reg_methods, cur_exp_path, num_classes,
                                     _specific_idxs=specific_indices, running_marabou=False, _jobs_wait=jobs_wait)

            # combine table
            print("\n###### Combining tables ######\n", flush=True)
            combine_entries(cur_exp_path + "rows/", query_methods, pool_size, to_delete)

        # retrain
        print("\n###### Retraining ######\n", flush=True)
        retrain_models(trial_name, cur_exp_path, prev_exp_path, model_path, data_path, samples_to_retrain, cycle_num,
                       x_unlabeled, y_unlabeled, x_test, y_test, query_methods, batch_methods, run_rand,
                       rand_adv_methods,
                       _num_classes=num_classes, _epoch_num=epoch_num, _dataset_name=dataset_name, jobs_wait=jobs_wait)

        # delete unnecessary files
        if to_delete:
            print(f"deleting files - cycle {cycle_num}", flush=True)
            # remove stdout files
            directory_path = base_path + f"/stdouts/create_row_trial_{trial_name}"
            for file_name in os.listdir(directory_path):
                temp_path = os.path.join(directory_path, file_name)
                # Check if it's a file and delete it
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            directory_path = base_path + f"/stdouts/get_advs_trial_{trial_name}"
            for file_name in os.listdir(directory_path):
                temp_path = os.path.join(directory_path, file_name)
                # Check if it's a file and delete it
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            print("removed stdout files", flush=True)
            directory_path = cur_exp_path + "ipqs/"
            for file_name in os.listdir(directory_path):
                temp_path = os.path.join(directory_path, file_name)
                # Check if it's a file and delete it
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            print("removed ipq files", flush=True)
            directory_path = cur_exp_path + "sums/"
            for file_name in os.listdir(directory_path):
                temp_path = os.path.join(directory_path, file_name)
                # Check if it's a file and delete it
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            print("removed sums files", flush=True)
            for method_name in (query_methods + rand_adv_methods + batch_methods):
                if method_name.startswith(
                        ("rand_w_samples", "rand_FGSM", "eps_w_samples", "eps_FGSM", "eps_full",
                         "FGSM", "DFAL", "badge_w_samples", "coreset_w_samples", "FVAAL_rand")):
                    for np_file_name in os.listdir(cur_exp_path + f"{method_name}_advs/"):
                        file_path = os.path.join(cur_exp_path + f"{method_name}_advs/", np_file_name)
                        if os.path.isfile(file_path) and np_file_name.endswith('.npy') and np_file_name.startswith(
                                'idx'):  # making sure we don't delete generated.npy files
                            os.remove(file_path)
                    # removing previous *_all_advs.npy
                    if cycle_num > 0:
                        for np_file_name in os.listdir(prev_exp_path + f"{method_name}_advs/"):
                            file_path = os.path.join(prev_exp_path + f"{method_name}_advs/", np_file_name)
                            if os.path.isfile(file_path) and np_file_name.endswith('advs.npy'):
                                os.remove(file_path)
            print("removed advs files", flush=True)
            if cycle_num > 0:
                path_to_check = f"{prev_exp_path}/models/"
                for item in os.listdir(path_to_check):
                    item_path = os.path.join(path_to_check, item)
                    if os.path.isdir(item_path):
                        print(f"Deleting folder: {item_path}")
                        shutil.rmtree(item_path)
                print("removed previous models", flush=True)
        print("Done! \n", flush=True)


if __name__ == "__main__":
    start_time = time.time()
    main()

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTrial ended successfully!", flush=True)
    print(f"\nRuntime: {runtime:.6f} seconds, ~{int(runtime / 3600)} hours", flush=True)
