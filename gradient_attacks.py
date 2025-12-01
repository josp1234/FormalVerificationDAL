import tensorflow as tf
import numpy as np
# from PIL import Image


def fgsm_attack(model, x_input, epsilons):
    """
    Generate FGSM adversarial examples for multiple epsilon values using model's prediction (logits),
    and return whether each attack succeeded.

    Args:
        model (tf.keras.Model): Model that outputs logits.
        x_input (tf.Tensor or np.array): Input image(s), shape (N, H, W, C).
        epsilons (list or array): List of epsilon values.

    Returns:
        adversarial_examples: List of adversarial examples (one per epsilon).
        success_flags: List of booleans indicating if the attack changed the prediction.
    """
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
    x_input = tf.expand_dims(x_input, 0)

    # Get original prediction (logits) and class
    logits_orig = model(x_input, training=False)
    y_pred = tf.one_hot(tf.argmax(logits_orig, axis=1), depth=logits_orig.shape[1])
    orig_class = tf.argmax(logits_orig, axis=1)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    adversarial_examples = []
    success_flags = []

    for eps in epsilons:
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            logits = model(x_input, training=False)
            loss = loss_fn(y_pred, logits)

        gradient = tape.gradient(loss, x_input)
        signed_grad = tf.sign(gradient)
        x_adv = x_input + eps * signed_grad
        x_adv = tf.clip_by_value(x_adv, 0, 1)

        # Check if prediction changed
        logits_adv = model(x_adv, training=False)
        adv_class = tf.argmax(logits_adv, axis=1)
        success = adv_class[0] != orig_class[0]
        adversarial_examples.append(x_adv)
        success_flags.append(success)

    return adversarial_examples, success_flags


def get_epsilon_FGSM(_model, _image, _pred, img_idx, cur_dir_path, method):
    adv = None
    epsilon = 0.5
    precision = 0.0005

    left = 0
    right = 1
    iter_idx = 0
    while right - left > precision:
        _raw_adv, _success = fgsm_attack(_model, _image, [epsilon])
        _success_np = np.array(_success)
        if _success_np[0]:
            right = epsilon
            _raw_adv_tensor = tf.stack(_raw_adv)
            adv = _raw_adv_tensor[0, 0].numpy()
        else:
            left = epsilon
        epsilon = left + (right - left) / 2
        epsilon = max(0, min(epsilon, 1))

        iter_idx += 1
    cur_path = cur_dir_path + f"{method}_advs/idx_{img_idx}_0.npy"
    np.save(cur_path, adv)
    return epsilon



def get_advs_FGSM(_model, _epsilon, _image, _pred, img_idx, cur_dir_path, _n_samples, method, start_0=False, upper_bound=0.35):
    print(f"Starting FGSM for idx {img_idx}...", flush=True)
    if _epsilon >= upper_bound:
        _epsilon = 0
    epsilons = list(np.linspace(_epsilon, upper_bound, num=_n_samples * 2))

    # Run FGSM attack
    _raw_advs, _success = fgsm_attack(_model, _image, epsilons)

    # Collect successful adversarials
    _raw_advs_tensor = tf.stack(_raw_advs)  # shape: (num_epsilons, 1, H, W, C)
    _success_np = np.array(_success).reshape(len(epsilons))

    # Take the ones in odd indices. If adv generation didn't succeed, try to take the previous one or next one.
    c = -1 if start_0 else 0
    for i, succeeded in enumerate(_success_np):
        if i % 2 == 1:
            if succeeded:
                adv = _raw_advs_tensor[i, 0].numpy()
                _success_np[i] = False
            else:
                if _success_np[i - 1]:
                    adv = _raw_advs_tensor[i - 1, 0].numpy()
                    _success_np[i - 1] = False
                elif i < len(_success_np) - 1 and _success_np[i + 1]:
                    adv = _raw_advs_tensor[i + 1, 0].numpy()
                    _success_np[i + 1] = False
                else:
                    continue
            c += 1
            np.save(cur_dir_path + f"{method}_advs/idx_{img_idx}_{c}.npy", adv)
            print(f"found adv #{c}", flush=True)