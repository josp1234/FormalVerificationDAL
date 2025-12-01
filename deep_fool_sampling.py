import tensorflow as tf


def deepfool(
        model,
        x0,
        num_classes=10,
        max_iter=20,
        overshoot=0.02,
        tol=1e-6
):
    """
    Compute a (near-)minimal adversarial perturbation with DeepFool.

    Args:
        x0:          Input image as a float32 Tensor of shape (H, W, C) or (1, H, W, C).
        model:       Keras model; should output logits (preferred). If it outputs
                     probabilities, the method still works but is less ideal.
        num_classes: Number of classes to consider when searching the closest boundary.
                     (You can set this <= total classes for speed.)
        max_iter:    Maximum DeepFool iterations.
        overshoot:   Small factor to go slightly past the boundary.
        tol:         Early-stop threshold on step size.

    Returns:
        x_adv:       Adversarial example (Tensor, same shape as x0).
        r_tot:       Total perturbation added (Tensor).
        iters:       Number of iterations performed.
        orig_label:  Original predicted label (int).
        new_label:   New predicted label (int).
    """
    # Ensure batch dimension
    input_image = tf.convert_to_tensor(x0, dtype=tf.float32)
    vector_input = (input_image.shape.rank == 1)
    if vector_input:
        input_image = input_image[None, :]  # (1, 784)

    # Remember original prediction
    logits0 = model(input_image, training=False)
    orig_label = tf.argmax(logits0[0]).numpy().item()

    # For efficiency, restrict to top-k classes each iteration
    total_classes = logits0.shape[-1]
    k_eff = min(num_classes, total_classes)

    r_tot = tf.zeros_like(input_image)
    perturbed_image = tf.identity(input_image)
    iters = 0

    for it in range(max_iter):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(perturbed_image)
            logits = model(perturbed_image, training=False)

            # current predicted class k
            k = tf.argmax(logits[0])
            k = tf.cast(k, tf.int32)

            # Stop if already misclassified
            if k.numpy().item() != orig_label:
                break

            # pick top-k classes to compare against (including current)
            top_vals, top_idx = tf.math.top_k(logits[0], k=k_eff)
            # Ensure current class index is included
            if not tf.reduce_any(top_idx == k):
                top_idx = tf.concat([top_idx[:-1], tf.reshape(k, (1,))], axis=0)

        # Compute minimal step to the closest linearized boundary
        # We’ll look for class j != k that minimizes |f_j - f_k| / ||∇(f_j - f_k)||
        w_min = None
        f_min = None
        dist_min = tf.constant(float('inf'), dtype=tf.float32)

        # Gradient of current class logit
        with tf.GradientTape() as tape_k:
            tape_k.watch(perturbed_image)
            f_k = model(perturbed_image, training=False)[:, k]
        grad_k = tape_k.gradient(f_k, perturbed_image)  # shape like x_adv

        for j in tf.reshape(top_idx, [-1]):
            j = tf.cast(j, tf.int32)
            if j == k:
                continue

            with tf.GradientTape() as tape_j:
                tape_j.watch(perturbed_image)
                f_j = model(perturbed_image, training=False)[:, j]
            grad_j = tape_j.gradient(f_j, perturbed_image)

            w = grad_j - grad_k  # ∇(f_j - f_k)
            f = (logits[:, j] - logits[:, k])  # (f_j - f_k), shape (1,)
            w_norm = tf.norm(tf.reshape(w, (w.shape[0], -1)), ord=2, axis=1) + 1e-12
            dist = tf.abs(f) / w_norm  # distance to boundary

            # keep minimal distance
            if dist[0] < dist_min:
                dist_min = dist[0]
                w_min = w
                f_min = f

        # If gradients vanished or we didn't find a direction, bail out
        if w_min is None:
            break

        w_flat = tf.reshape(w_min, (w_min.shape[0], -1))  # (B, D)
        w_norm_sq = tf.reduce_sum(w_flat * w_flat, axis=1, keepdims=True) + 1e-12  # (B, 1)

        # f_min has shape (1,), make it (B,1)
        f_abs = tf.reshape(tf.abs(f_min), (-1, 1))  # (B, 1)

        # r_i in flat space, then reshape back to w_min's shape
        r_i_flat = (f_abs / w_norm_sq) * w_flat  # (B, D)
        r_i = tf.reshape(r_i_flat, tf.shape(w_min))  # same shape as w_min (matches input)

        # Accumulate and update
        r_tot = r_tot + (1.0 + overshoot) * r_i
        step_norm = tf.norm(tf.reshape(r_i, (r_i.shape[0], -1)), ord=2)

        perturbed_image = input_image + r_tot

        iters += 1
        # Early stop on tiny steps (helps avoid infinite loops on flat regions)
        if step_norm.numpy().item() < tol:
            break

        # Check if label flipped; loop will also check at the top next iteration
        curr_label = tf.argmax(model(perturbed_image, training=False)[0]).numpy().item()
        if curr_label != orig_label:
            break

    perturbed_image = tf.squeeze(perturbed_image, axis=0) if x0.ndim == 3 else perturbed_image
    r_tot = tf.squeeze(r_tot, axis=0) if x0.ndim == 3 else r_tot

    if vector_input:
        perturbed_out = tf.squeeze(perturbed_image, axis=0)  # (784,)
        r_tot_out = tf.squeeze(r_tot, axis=0)  # (784,)
    else:
        perturbed_out = tf.identity(perturbed_image)
        r_tot_out = tf.identity(r_tot)

    return perturbed_out, r_tot_out, iters
