from matplotlib import pyplot

from starter_code.utils import *
from scipy.linalg import sqrtm
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu

    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # fix z[q] and derive wrt u[n]
    dL_du = -z[q].reshape(-1, 1) @ (
                c - (u[n].reshape(-1, 1).T @ z[q].reshape(-1, 1)))
    u[n] -= lr * dL_du[0]

    # fix u[n] and derive wrt z[q]
    dL_dz = -u[n].reshape(-1, 1) @ (
                c - (u[n].reshape(-1, 1).T @ z[q].reshape(-1, 1)))
    z[q] -= lr * dL_dz[0]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################

    for i in range(num_iteration):
        update_u_z(train_data, lr, u, z)

    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, u, z


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [1, 5, 7, 9, 11]
    accuracies = []
    for k in k_values:
        predictions = []
        curr_svd_matrix = svd_reconstruct(train_matrix, k)
        for i in range(len(val_data['user_id'])):
            curr_user_id = val_data['user_id'][i]
            curr_ques_id = val_data['question_id'][i]
            if curr_svd_matrix[curr_user_id][curr_ques_id] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        correct = 0
        for j in range(len(predictions)):
            if predictions[j] == val_data['is_correct'][j]:
                correct += 1
        accuracies.append(correct / len(val_data['is_correct']))

    print(accuracies)

    # Test for k*=9:
    predictions = []
    curr_svd_matrix = svd_reconstruct(train_matrix, 9)
    for i in range(len(test_data['user_id'])):
        curr_user_id = test_data['user_id'][i]
        curr_ques_id = test_data['question_id'][i]
        if curr_svd_matrix[curr_user_id][curr_ques_id] >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    correct = 0
    for j in range(len(predictions)):
        if predictions[j] == test_data['is_correct'][j]:
            correct += 1
    test_acc = correct / len(test_data['is_correct'])
    print(test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.
    lr = 0.12
    num_iterations = 50000
    iteration = [i for i in range(num_iterations)]
    als_k = [1, 4, 5, 15, 22, 30, 35, 40, 50, 70, 80, 150]

    losses = []
    for k in als_k:
        curr_als_matrix, u, z = als(train_data, k, lr, num_iterations)
        losses.append(squared_error_loss(val_data, u, z))

    valid_losses =[]
    for i in range(50000):
        curr_als_matrix, u, z = als(train_data, 35, lr, i)
        valid_losses.append(squared_error_loss(val_data, u, z))
    pyplot.plot(iteration, valid_losses)
    pyplot.title("Graph of Validation losses vs number of iterations")
    pyplot.xlabel("Number of iterations")
    pyplot.ylabel("Validation losses")
    pyplot.show()

    k_star = als_k[losses.index(min(losses))]
    test_als, u, z = als(train_data, k_star, lr, num_iterations)
    test_loss = squared_error_loss(test_data, u, z)
    print('Test squared-error loss: ', test_loss)

    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
