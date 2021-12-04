from starter_code.utils import *
from matplotlib import pyplot as plt
import numpy as np


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
    # Implement the function as described in the docstring.             #
    #####################################################################
    N = len(data["is_correct"])

    vec = np.zeros((N, 1))
    for i in range(N):
        term_1 = data["is_correct"][i] * (theta[data["user_id"][i]] -
                                          beta[data["question_id"][i]])
        term_2 = np.logaddexp(0, theta[data["user_id"][i]] -
                              beta[data["question_id"][i]])
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
    # Implement the function as described in the docstring.             #
    #####################################################################
    N = len(data["is_correct"])

    dL_dtheta = np.zeros((542, 1))
    for i in range(N):
        dL_dtheta[data["user_id"][i]] += data["is_correct"][i] - \
                                         sigmoid(theta[data["user_id"][i]] -
                                                 beta[data["question_id"][i]])

    theta += lr * dL_dtheta

    dL_dbeta = np.zeros((1774, 1))
    for i in range(N):
        dL_dbeta[data["question_id"][i]] += -1 * data["is_correct"][i] + \
                                            sigmoid(theta[data["user_id"][i]] -
                                                    beta[data["question_id"][i]]
                                                    )

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

    #####################################################################

    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    val_theta, val_beta, val_acc_lst = irt(train_data, val_data, lr=0.01,
                                           iterations=15)
    test_theta, test_beta, test_acc_lst = irt(train_data, test_data, lr=0.01,
                                              iterations=15)

    print('Validation Accuracy: ', val_acc_lst)
    print('Test Accuracy: ', test_acc_lst)

    # plotting the training and validation log-likelihoods as a function of
    # iteration

    train_theta, v_theta = np.random.uniform(low=0, high=1, size=(542, 1)), \
                           np.random.uniform(low=0, high=1, size=(542, 1))
    train_beta, v_beta = np.random.uniform(low=0, high=1, size=(1774, 1)), \
                         np.random.uniform(low=0, high=1, size=(1774, 1))

    train_nllk, val_nllk = [], []
    itr = [i for i in range(15)]
    for i in range(15):
        train_nllk.append(
            neg_log_likelihood(train_data, theta=train_theta, beta=train_beta))
        val_nllk.append(
            neg_log_likelihood(val_data, theta=v_theta, beta=v_beta))
        train_theta, train_beta = update_theta_beta(train_data, 0.01,
                                                    train_theta, train_beta)
        v_theta, v_beta = update_theta_beta(val_data, 0.01, v_theta, v_beta)

    fig = plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-likelihood')
    # plt.plot(itr, train_nllk, color='orange', label='training')
    plt.title('Validation Negative Log-likelihood vs Iteration')
    plt.plot(itr, val_nllk, color='green', label='validation')
    # fig.savefig("q2_b2.png")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################

    questions = [358, 7, 1720]
    prob_dict = {358: [[], []], 7: [[], []], 1720: [[], []]}

    for i, q in enumerate(train_data["question_id"]):
        if q in questions:
            u = train_data["user_id"][i]
            x = (val_theta[u] - val_beta[q]).sum()
            p_a = sigmoid(x)
            prob_dict[q][0].append(val_theta[u])
            prob_dict[q][1].append(p_a)

    fig = plt.figure()
    plt.xlabel('Theta')
    plt.ylabel('Probability')
    plt.title('Probability of Correct Response vs Theta Given a Question j')

    plt.scatter(prob_dict[358][0], prob_dict[358][1], color='orange',
                label='j1 = 358')
    plt.scatter(prob_dict[7][0], prob_dict[7][1], color='green', label='j2 = 7')
    plt.scatter(prob_dict[1720][0], prob_dict[1720][1], color='blue',
                label='j3 = 1720')
    plt.legend(title='Legend')
    # fig.savefig("q2_d.png")
    plt.show()

    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
