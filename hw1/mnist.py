"""Problem 3 - Training on MNIST"""
import numpy as np
import pdb

from mytorch.optim.sgd import SGD
from mytorch.nn.activations import ReLU
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.linear import Linear
from mytorch.nn.sequential import Sequential
from mytorch.tensor import Tensor

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100


def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)

    Args:
        train_x (np.array): training data (55000, 784)
        train_y (np.array): training labels (55000,)
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    # convert training and validation datasets to tensors
    # create MLP with provided architecture
    # MLP architecture -> Linear(784, 20) -> BatchNorm1d(20) -> ReLU() -> Linear(20, 10)
    model = Sequential(Linear(784, 20), ReLU(), Linear(20, 10))
    # Set the learning rate of your optimizer to lr=0.1.
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.1)
    # Initialize your criterion (CrossEntropyLoss)
    criterion = CrossEntropyLoss()

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(
        model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3
    )
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    print("Training...")
    # model.activate_train_mode()
    model.train()
    # for each epoch:
    for epoch in range(num_epochs):
        # shuffle_train_data()
        shuffle = True
        if shuffle:
            dataset_size = np.arange(len(train_x))
            permutation = np.random.shuffle(dataset_size)
            train_x = np.squeeze(train_x[permutation])
            train_y = np.squeeze(train_y[permutation])

        # break into bactches
        # batches = split_data_into_batches()

        BATCH_SIZE = 100
        current_batch = 0
        # batches = []
        # batches = split_data_into_batches(train_x, train_y)
        x_batches = np.array_split(train_x, 100)
        y_batches = np.array_split(train_y, 100)
        batches = zip(x_batches, y_batches)

        val_accuracies = []
        # for i, (batch_data, batch_labels) in enumerate(batches):
        for i, (batch_data, batch_labels) in enumerate(batches):

            optimizer.zero_grad()  # clear any previous gradients
            out = model.forward(Tensor(batch_data))
            loss = criterion.forward(out, Tensor(batch_labels))
            loss.backward()
            optimizer.step()  # update weights with new gradients
            # loss = CrossEntropyLoss(out, batch_labels)
            # loss.backwards()
            if BATCH_SIZE % 100 == 0:
                accuracy = validate(model=model, val_x=val_x, val_y=val_y)
                # store_validation_accuracy(accuracy)
                val_accuracies.append(accuracy)
                model.train()

    # TODO: Implement me! (Pseudocode on writeup)
    return val_accuracies


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    # TODO: implement validation based on pseudocode
    # model.activate_eval_mode()
    model.eval()
    # batches = split_data_into_batches()
    # batches = split_data_into_batches(features=val_x, labels=val_y)

    x_batches = np.array_split(val_x, 100)
    y_batches = np.array_split(val_y, 100)
    batches = zip(x_batches, y_batches)
    num_correct = 0
    # for (batch_data, batch_labels) in batches:
    for i, (batch_data, batch_labels) in enumerate(batches):
        # out = forward_pass(batch_data)
        # pdb.set_trace()
        out = model.forward(Tensor(batch_data))

        # batch_preds = get_idxs_of_largest_values_per_batch(out)
        batch_preds = np.argmax(out.data, axis=1)
        # print(f"Batch preds are {batch_preds}")
        num_correct += batch_preds == batch_labels
    accuracy = num_correct.sum() / len(val_x)

    return accuracy
