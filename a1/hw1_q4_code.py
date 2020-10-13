import json
from typing import List
from typing import Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_train = {'X': np.genfromtxt('C:/Users/conqu/PycharmProjects/csc311/a1/data/data_train_X.csv', delimiter=','),
              't': np.genfromtxt('C:/Users/conqu/PycharmProjects/csc311/a1/data/data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('C:/Users/conqu/PycharmProjects/csc311/a1/data/data_test_X.csv', delimiter=','),
             't': np.genfromtxt('C:/Users/conqu/PycharmProjects/csc311/a1/data/data_test_y.csv', delimiter=',')}
# Note: np.genfromtxt returns an ndarray.


def shuffle_data(data: object) -> object:
    """
    This returns permuted version of the data along the samples.
    :rtype: object
    :return:
    """
    n_x = np.random.permutation(data['X'])
    n_t = np.random.permutation(data['t'])
    n_data = {'X': n_x, 't': n_t}
    return n_data


def split_data(data: object, num_folds, fold) -> object:
    """
    This function return both the selected partition fold as data_fold, and remaining data
    as data_rest.
    :param data:
    :param num_folds:
    :param fold:
    :return:
    """
    len_fold = len(data['t']) // num_folds
    end = fold*len_fold
    start = end - len_fold
    all_range = [i for i in range(0, len(data['X']))]
    fold_range = [i for i in range(start, end)]
    rest_range = list(set(all_range).difference(set(fold_range)))
    data_fold = {'X': data['X'][fold_range], 't': data['t'][fold_range]}
    data_rest = {'X': data['X'][rest_range], 't': data['t'][rest_range]}

    return data_fold, data_rest


def train_model(data, lambd) -> object:
    """
    This function returns the coefficient of ridge regression with penalty level lambd.
    :param data:
    :param lambd:
    :return:
    """
    X = data['X']
    t = data['t']
    N = len(X)
    D = len(X.T)
    A = X.T.dot(X) + np.identity(D) * lambd * N
    w = np.linalg.inv(A).dot(X.T.dot(t))
    return w


def predict(data: object, model: object) -> object:
    """
    This function returns model prediction based on the data and model
    :param data:
    :param lamb:
    :return:
    """
    X = data['X']
    return X.dot(model)


def loss(data, model) -> object:
    """
    Return the average squared error loss based on model.
    :param data:
    :param model:
    :return:
    """
    t = data['t']
    w = model
    N = len(t)
    pred = predict(data, model)
    avg_loss = np.linalg.norm(np.subtract(pred, t))/ (N*2)
    return avg_loss


def cross_validation(data, num_folds, lambd_seq) -> List[Union[float, Any]]:
    """
    Return cross validation error across all lambda value in lambd_sequence.
    :param data:
    :param num_folds:
    :param lambd_seq:
    :return:
    """
    cv_error = []
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error += [cv_loss_lmd / num_folds]
    return cv_error


if __name__ == '__main__':
    lambd_seq = np.linspace(0.00005, 0.005, 50)
    train_errors = []
    test_errors = []
    fold5_errors = cross_validation(data_train, 5, lambd_seq)
    fold10_errors = cross_validation(data_train, 10, lambd_seq)
    for lambd in lambd_seq:
        model = train_model(data_train, lambd)
        train_errors += [loss(data_train, model)]
        test_errors += [loss(data_test, model)]
    raw = {'lamb': lambd_seq, 'train_error': train_errors, 'test_error': test_errors}
    report = pd.DataFrame(raw)
    print(report)

    plt.figure(figsize=(10, 6))
    plt.plot(lambd_seq, train_errors, label='Training Errors')
    plt.plot(lambd_seq, test_errors, label='Testing Errors')
    plt.plot(lambd_seq, fold5_errors, label='5-Fold CV Error')
    plt.plot(lambd_seq, fold10_errors, label='10-Fold CV Error')

    plt.xlabel('Lambda Value for Ridge')
    plt.ylabel('Average Squared Error Loss')
    plt.title('Testing, Training Error on Ridge Regression with Different Lambda')
    plt.legend()

    plt.savefig('CV_Error_Report')
    plt.show()
