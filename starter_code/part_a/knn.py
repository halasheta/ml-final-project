from matplotlib import pyplot
from sklearn.impute import KNNImputer
from utils import *

from starter_code.utils import sparse_matrix_evaluate, load_train_sparse, \
    load_valid_csv, load_public_test_csv


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    #####################################################################
    ks = [1, 6, 11, 16, 21, 26]
    kstar = 0
    best_accuracy = 0
    accuracies = []
    for k in ks:
        # item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracies.append(user_acc)
        # if item_acc > best_accuracy:
        #    best_accuracy = item_acc
        #    kstar = k
        if user_acc > best_accuracy:
            best_accuracy = user_acc
            kstar = k
    pyplot.plot(ks, accuracies)
    pyplot.show()
    print(kstar, best_accuracy)
    return knn_impute_by_user(sparse_matrix, test_data, kstar)
    # what is left is to plot/report results - nm

    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    print(main())
