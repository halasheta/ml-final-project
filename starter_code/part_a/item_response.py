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
        if data["is_correct"][i] == 1:
            theta_i = np.tile(theta[i], (D, 1))
            # vec[i] = np.sum(theta_i) - np.sum(np.log(np.exp(theta_i) + np.exp(beta)))
            vec[i] = np.sum(theta_i - np.logaddexp(theta_i, beta))
    log_lklihood = np.sum(vec)

    # is_correct = np.array(data['is_correct']).reshape(-1, 1)
    # theta_vect = is_correct * theta
    # beta_vect = is_correct * beta

    # theta_matrix = np.tile(theta, (D, 1))
    # beta_matrix = np.tile(beta, (D, 1))
    #
    # print(theta_matrix.shape)
    # print(beta_matrix.shape)
    #
    # term2 = np.sum(np.log(np.exp(theta_matrix.T) + np.exp(beta_matrix)), axis=0)
    # print(term2.shape)
    # # log_likelihood_matrix = theta_matrix - term2
    # print(theta_matrix.shape)
    # print(theta_matrix)


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
    # num_iterations = 1
    N, D = theta.shape[0], beta.shape[0]

    # num_it
    # for k in range(num_iterations):
    dL_dtheta = np.zeros((N, 1))
    for i in range(N):
        # if data["is_correct"][i] == 1:
        theta_i = np.tile(theta[i], (D, 1))
        # dL_dtheta[i] = np.sum(np.exp(beta) / (np.exp(theta_i) + np.exp(beta)))
        dL_dtheta[i] = np.sum(np.exp(beta) / np.exp(np.logaddexp(theta_i, beta)))

    # gradient of the sum of the losses = sum of individual losses * lr
    theta = theta - (lr * dL_dtheta)

    dL_dbeta = np.zeros((D, 1))
    for j in range(D):
        if data["is_correct"][j] == 1:
            beta_j = np.tile(beta[j], (N, 1))
            # dL_dbeta[j] = np.sum(- np.exp(beta_j) / (np.exp(theta) + np.exp(beta_j)))
            dL_dbeta[j] = np.sum(- np.exp(beta_j) / np.exp(np.logaddexp(theta, beta_j)))
    beta = beta - (lr * dL_dbeta)

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
    # theta = np.random.uniform(low=0, high=1,
    #                           size=(len((data["user_id"])), 1))
    # beta = np.random.uniform(low=0, high=1,
    #                          size=(len((data["question_id"])), 1))
    theta = np.zeros((len(data['user_id']), 1))
    beta = np.zeros((len(data['question_id']), 1))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        print(theta, beta)

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

    theta = np.random.uniform(low=0, high=1,
                              size=(len((train_data["user_id"])), 1))
    beta = np.random.uniform(low=0, high=1,
                             size=(len((train_data["question_id"])), 1))

    print('loglik:')
    print(irt(train_data, val_data, lr=0.1, iterations=4))
    # print(neg_log_likelihood(train_data, theta, beta))

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
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
