from scipy import stats
import numpy as np
import tensorflow as tf
import os
import time
import argparse


def distance(X1, X2, mu):
    Y1, Y2 = mu
    probs, prob_norms_square = X1
    embs, emb_norms_square = X2
    mu_probs, mu_probs_norm_square = Y1
    mu_embs, mu_embs_norm_square = Y2
    dist = prob_norms_square * emb_norms_square + mu_probs_norm_square * mu_embs_norm_square - 2 * (
            probs @ mu_probs) * (embs @ mu_embs)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist


def init_centers(X1, X2, chosen, chosen_list, mu, D2):
    probs = X1[0]
    prob_norms_square = X1[1]
    embs = X2[0]
    emb_norms_square = X2[1]
    if len(chosen) == 0:
        ind = np.argmax(prob_norms_square * emb_norms_square)
        mu = [((probs[ind], prob_norms_square[ind]), (embs[ind], emb_norms_square[ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((probs[ind], prob_norms_square[ind]), (embs[ind], emb_norms_square[ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2


def query_badge(_model, _x_unlabeled, unlabeled_idxs, n):
    penultimate_layer = tf.keras.Model(inputs=_model.input, outputs=_model.layers[-2].output)
    embs = penultimate_layer.predict(_x_unlabeled[unlabeled_idxs])
    logits = _model.predict(_x_unlabeled[unlabeled_idxs])
    probs = tf.nn.softmax(logits, axis=-1).numpy()

    # the logic below reflects a speedup proposed by Zhang et al.
    # see Appendix D of https://arxiv.org/abs/2306.09910 for more details

    # how many unlabeled there are
    m = unlabeled_idxs.shape[0]
    mu = None
    D2 = None
    chosen = set()
    chosen_list = []
    # L2 norm of each embedding: ||v_i||
    emb_norms_square = np.sum(embs ** 2, axis=-1)
    # index of "winner" in each sample
    max_inds = np.argmax(probs, axis=-1)

    probs = -1 * probs
    probs[np.arange(m), max_inds] += 1
    # L2 norm of q_i: ||q_i||
    prob_norms_square = np.sum(probs ** 2, axis=-1)

    for iter_num in range(n):
        # D2 is min dist of all samples from one of the centroids
        # mu is the probs & embs of chosen idxs
        if iter_num % 10 == 0:
            print(f"Running BADGE iteration {iter_num}...\n", flush=True)
        chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen,
                                                   chosen_list, mu, D2)
    return unlabeled_idxs[chosen_list]


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

    base_path = "./"
    cur_exp_path = base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num}/"
    base_model_path = base_path + f"/experiments/trial_{trial_name}/init_model/"
    base_data_path = base_path + f"/experiments/trial_{trial_name}/init_data/"

    x_unlabeled = np.load(base_data_path + "x_unlabeled_total.npy")

    if cycle_num == 0:
        model = tf.keras.models.load_model(base_model_path)
    else:
        model = tf.keras.models.load_model(base_path + f"/experiments/trial_{trial_name}/cycle_{cycle_num - 1}/models/{method}_model")

    pool_indices = np.load(cur_exp_path + f"data/{method}_pool_indices.npy")
    indices_to_add = query_badge(model, x_unlabeled, pool_indices, n_samples)
    np.save(cur_exp_path + f"data/{method}_chosen_indices.npy", indices_to_add)

    end_time = time.time()
    runtime = end_time - start_time
    print("\nTotal job runtime: {:.6f} seconds".format(runtime), flush=True)
