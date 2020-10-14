from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }
    weights = np.zeros((M+1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train_ce = []
    valid_ce = []
    train_rate, valid_rate, t_ce, v_ce = 0, 0, 0, 0
    i_range = range(hyperparameters["num_iterations"])

    for _ in i_range:
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights -= df*hyperparameters["learning_rate"]  # update df

        t_ce, train_rate = evaluate(train_targets, logistic_predict(weights, train_inputs))
        train_ce.append(t_ce)

        v_ce, valid_rate = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
        valid_ce.append(v_ce)

    plt.figure(figsize=(10, 6))
    plt.plot(i_range, train_ce, color='blue', label='Training Set')
    plt.plot(i_range, valid_ce, color='red', label='Validation Set')
    plt.legend()
    plt.title('Cross Entropy in Training and Validation Set v.s. Num_Iterations, MNIST')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy')
    plt.savefig("CE_MNIST.png")  # save the figure
    plt.show()

    test_ce, test_rate = evaluate(test_targets, logistic_predict(weights, test_inputs))
    print(train_rate, t_ce, valid_rate, v_ce, test_rate, test_ce)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape
    lambd_seq = [0, 0.001, 0.01, 0.1, 1.0]
    stats = {}

    for lambd in lambd_seq:
        hyperparameters = {
            "learning_rate": 0.1,
            "weight_regularization": lambd,
            "num_iterations": 1000
        }
        i_range = range(hyperparameters["num_iterations"])
        train_total_ce = 0
        valid_total_ce = 0
        train_total_rate = 0
        valid_total_rate = 0
        train_ce_set = []  # The set of all CE for different iterations
        valid_ce_set = []
        train_rt_set = []
        valid_rt_set = []

        for _ in range(5):
            weights = np.zeros((M+1, 1))
            train_rate, valid_rate = 0, 0
            train_ce, valid_ce = 0, 0
            for _ in i_range:
                f, df, y = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                weights -= df * hyperparameters["learning_rate"]  # update df

                train_ce, train_rate = evaluate(train_targets, logistic_predict(weights, train_inputs))
                train_ce_set.append(train_ce)
                train_rt_set.append(train_rate)

                valid_ce, valid_rate = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
                valid_ce_set.append(valid_ce)
                valid_rt_set.append(valid_rate)

            train_total_rate += train_rate
            valid_total_rate += valid_rate
            train_total_ce += train_ce
            valid_total_ce += valid_ce

        stats[str(lambd)] = {}
        stats[str(lambd)]['avg_train_acc'] = train_total_rate/5
        stats[str(lambd)]['avg_valid_acc'] = valid_total_rate/5
        stats[str(lambd)]['avg_train_ce'] = train_total_ce/5
        stats[str(lambd)]['avg_valid_ce'] = valid_total_ce/5

        # Take the first run
        plot_train_ce = train_ce_set[:hyperparameters["num_iterations"]]
        plot_valid_ce = valid_ce_set[:hyperparameters["num_iterations"]]

        plt.figure(figsize=(10, 6))
        plt.plot(i_range, plot_train_ce, color='blue', label='Training Set')
        plt.plot(i_range, plot_valid_ce, color='red', label='Validation Set')
        plt.legend()
        plt.title('Cross Entropy in Training and Validation Set v.s. Num_Iterations, MNIST-small for Lambda' + str(lambd))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cross Entropy')
        plt.savefig("Penalty" + str(lambd)+"CE_MNIST-Small.png")  # save the figure
        plt.show()
    print(stats)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    # run_pen_logistic_regression()
    run_logistic_regression()
