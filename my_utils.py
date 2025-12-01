import sys
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.datasets import cifar10
import os


def get_random_samples(x_train, y_train, labeled_size, uniform):
    selected_indices = []
    if uniform:
        class_size = int(labeled_size / 10)
        for class_label in range(10):
            # Get indices of samples belonging to the current class
            class_indices = np.where(y_train == class_label)[0]

            # Randomly shuffle the indices to ensure randomness
            np.random.shuffle(class_indices)

            # Select the first 'samples_per_class' indices from the shuffled list
            selected_indices.extend(class_indices[:class_size])
    else:
        selected_indices.extend(np.random.choice(len(y_train), size=labeled_size, replace=False))

    x_labeled = x_train[selected_indices]
    y_labeled = y_train[selected_indices].squeeze()
    x_unlabeled = np.delete(x_train, selected_indices, axis=0)
    y_unlabeled = np.delete(y_train, selected_indices).squeeze()
    return x_labeled, y_labeled, x_unlabeled, y_unlabeled


def load_dataset(dataset):
    if dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "fashionMNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        print("dataset not recognized!", flush=True)
        sys.exit(1)

    # Rescale the images from [0,255] to the [0.0,1.0] range
    x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    return x_train, y_train, x_test, y_test


def get_processed_data(labeled_size, dataset):
    x_train, y_train, x_test, y_test = load_dataset(dataset)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print('Train shape:')
    print(x_train.shape)
    print('Test shape:')
    print(x_test.shape)
    print("labels range:")
    print(f"[{np.min(y_train)}, {np.max(y_train)}]")
    print("x range:")
    print(f"[{np.min(x_test)}, {np.max(x_test)}]")

    # divide training set to initial labeled pool and unlabeled pool
    x_labeled, y_labeled, x_unlabeled, y_unlabeled = get_random_samples(x_train, y_train, labeled_size, uniform=False)
    print("x_labeled shape: " + str(x_labeled.shape))
    print("y_labeled shape: " + str(y_labeled.shape))
    print("x_unlabeled shape: " + str(x_unlabeled.shape))
    print("y_unlabeled shape: " + str(y_unlabeled.shape))
    return x_unlabeled, x_labeled, y_unlabeled, y_labeled, x_test, y_test


def train_model(x_train, y_train, x_test, y_test, model_path, num_classes=10, num_neurons=30,
                epoch_num=10):
    d_shape = int(x_train.shape[1])
    print(f"d_shape: {d_shape}")
    print(f"x_train.shape: {x_train.shape}")
    model = tf.keras.models.Sequential()
    model.add(Dense(units=num_neurons, input_shape=(d_shape,), activation='relu'))
    model.add(Dense(units=num_classes, activation=None, use_bias=False))

    # logits loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    # train model & check test accuracy
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epoch_num, verbose=True)
    acc = model.evaluate(x_test, y_test)[1]
    print(f"Model accuracy: {acc}")
    model.save(model_path, save_format='tf')
    print("Model saved in " + model_path)
    return acc


def load_data_and_train_model(model_path, data_path, num_classes=10, num_neurons=30, epoch_num=10):
    x_labeled = np.load(os.path.join(data_path, "x_labeled.npy"))
    y_labeled = np.load(os.path.join(data_path, "y_labeled.npy"))
    x_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    train_model(x_labeled, y_labeled, x_test, y_test, model_path, num_classes=num_classes,
                num_neurons=num_neurons, epoch_num=epoch_num)


def save_random_samples(indices, num_samples, out_path, method):
    chosen_indices = np.random.choice(len(indices), size=num_samples, replace=False)
    chosen = indices[chosen_indices]
    np.save(out_path + f"{method}_pool_indices.npy", chosen)


def save_data(labeled_size, out_path, dataset_name):
    x_unlabeled, x_labeled, y_unlabeled, y_labeled, x_test, y_test = get_processed_data(labeled_size, dataset_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    np.save(os.path.join(out_path, "x_unlabeled_total.npy"), x_unlabeled)
    np.save(os.path.join(out_path, "y_unlabeled_total.npy"), y_unlabeled)
    np.save(os.path.join(out_path, "x_labeled.npy"), x_labeled)
    np.save(os.path.join(out_path, "y_labeled.npy"), y_labeled)
    np.save(os.path.join(out_path, "x_test.npy"), x_test)
    np.save(os.path.join(out_path, "y_test.npy"), y_test)
    print("data saved in " + out_path, flush=True)
