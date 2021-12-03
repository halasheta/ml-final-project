from starter_code.utils import *

import numpy as np
import scipy.special as sp


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def stable_sigmoid(z):
    """
    A more stable sigmoid
    """

    return np.exp(z) / np.exp(np.logaddexp(0, z))


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
    N = len(data["is_correct"])

    vec = np.zeros((N, 1))
    for i in range(N):
        term_1 = data["is_correct"][i] * (theta[data["user_id"][i]] - beta[data["question_id"][i]])
        term_2 = np.logaddexp(0, theta[data["user_id"][i]] - beta[data["question_id"][i]])
        vec[i] = term_1 - term_2

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
    N = len(data["is_correct"])

    dL_dtheta = np.zeros((542, 1))
    for i in range(N):
        dL_dtheta[data["user_id"][i]] += data["is_correct"][i] - \
                                      sigmoid(theta[data["user_id"][i]] - beta[data["question_id"][i]])

    theta += lr * dL_dtheta

    dL_dbeta = np.zeros((1774, 1))
    for i in range(N):
        dL_dbeta[data["question_id"][i]] += -1 * data["is_correct"][i] + \
                                      sigmoid(theta[data["user_id"][i]] - beta[data["question_id"][i]])

    beta += lr * dL_dbeta

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
    theta = np.random.uniform(low=0, high=1,
                              size=(542, 1))
    beta = np.random.uniform(low=0, high=1,
                             size=(1774, 1))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        print('theta: ', theta)
        print('beta: ', beta)

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
        # p_a = stable_sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print('irt:')
    print(irt(train_data, val_data, lr=0.01, iterations=20))
    # print(neg_log_likelihood(train_data, tta, beta))

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # update_theta_beta(train_data, 0.01, theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
