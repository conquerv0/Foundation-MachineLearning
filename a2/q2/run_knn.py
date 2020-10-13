from q2.l2_distance import l2_distance
from q2.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    k_range = [1, 3, 5, 7, 9]
    class_rate = []
    for k in k_range:
        valid_pred = knn(k, train_inputs, train_targets, valid_inputs)
        correct_set = []
        for i in range(len(valid_pred)):
            pred = valid_pred[i]
            targ = valid_targets[i]
            if pred == targ:
                correct_set.append(pred)
        rate = len(correct_set) / len(valid_pred)
        class_rate.append(rate)

    # generate a plot showing the validation classification rate
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, class_rate, color='blue', label='Validation Classification Rate')
    plt.legend()
    plt.xlabel('K-value')
    plt.ylabel('Classification Rate')
    plt.title('Classification Rate on Validation Set of Different K Value')
    plt.savefig("classification_rate.png")  # save the figure
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    """
    The performance of the classifier increased as k increased from 1 to 3,
    peaked around k-value of 3, 5 and 7; and sharply decreased at k=9. Considering the fact
    that smaller k may overfit the data, while a larger k tend to underfit as it fails to 
    capture important regularities; we choose its median, k=5 as our k*.
    
    As reported in the classification rate plot v2, the corresponding classification rate for k*+2
    and k*-2 is around 0.94 and 0.92 respectively. These test performance corresponds to the validation
    set in that it 
    """
    # TODO: 2.1-b Above Comments To be completed.


def run_knn_v2():
    train_inputs, train_targets = load_train()
    test_inputs, test_targets = load_test()

    k_range = [3, 5, 7]
    class_rate = []
    for k in k_range:
        test_pred = knn(k, train_inputs, train_targets, test_inputs)
        correct_set = []
        for i in range(len(test_pred)):
            pred = test_pred[i]
            targ = test_targets[i]
            if pred == targ:
                correct_set.append(pred)
        rate = len(correct_set) / len(test_pred)
        class_rate.append(rate)

    # generate a plot showing the test classification rate
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, class_rate, color='red', label='Test Classification Rate')
    plt.legend()
    plt.xlabel('K-value')
    plt.ylabel('Classification Rate-V2')
    plt.title('Classification Rate on Test Set of Different K Value')
    plt.savefig("classification_rate-v2.png")  # save the figure
    plt.show()


if __name__ == "__main__":
    run_knn()
    run_knn_v2()