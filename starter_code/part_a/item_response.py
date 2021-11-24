from starter_code.utils import *

import numpy as np
import scipy.special as sp


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    N, D = theta.shape[0], beta.shape[0]
    #
    # s = 0
    # for i in range(N):
    #     for j in range(D):
    #         if data["is_correct"][i] == 1:
    #             s += theta[i] - np.log(np.exp(theta[i]) + np.exp(beta[j]))
    #     print(s)
    # log_lklihood = s[0]

    # vec = np.zeros((N, 1))
    # for i in range(N):
    #     vec[i] = np.sum(np.log(np.exp(theta[i]) + np.exp(beta)))

    # log_lklihood = np.sum(D * theta - vec)

    vec = np.zeros((N, 1))
    for i in range(N):
        theta_i = np.tile(theta[i], (D, 1))
        if data["is_correct"][i] == 1:
            vec[i] = np.sum(theta_i) - np.sum(np.log(np.exp(theta_i) + np.exp(beta)))
    log_lklihood = np.sum(vec)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_iterations = 50
    N, D = theta.shape[0], beta.shape[0]

    print(len(data["user_id"]), len(data["question_id"]), len(data["is_correct"]))
    # need to include the indicator somehow
    for k in range(num_iterations):
        dL_dtheta = 0
        for i in range(N):
            if data["is_correct"][i] == 1:
                dL_dtheta += np.sum(np.exp(beta) / (np.exp(theta[i]) + np.exp(beta)))

        dL_dbeta = 0
        for j in range(D):
            if data["is_correct"][j] == 1:
                dL_dbeta += - np.exp(beta[j]) / (np.exp(theta) + np.exp(beta[j]))
        theta = theta - lr * dL_dtheta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = None
    beta = None

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    N = len(train_data["user_id"])
    theta = np.zeros((N, 1))
    beta = np.zeros((N, 1))

    # theta[0], theta[1], theta[2] = 0.4, 0.3, 0.1
    # beta[0], beta[1] = 0.1, 0.3

    print(neg_log_likelihood(train_data, theta, beta))

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    update_theta_beta(train_data, 0.01, theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
