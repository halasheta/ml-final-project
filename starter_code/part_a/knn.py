from matplotlib import pyplot
from sklearn.impute import KNNImputer

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
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)

    # sparse matrix evaluate
    total_prediction = 0
    total_accurate = 0
    for i in range(len(valid_data["is_correct"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        if mat[cur_question_id, cur_user_id] >= 0.5 and valid_data["is_correct"][i]:
            total_accurate += 1
        if mat[cur_question_id, cur_user_id] < 0.5 and not valid_data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    acc = total_accurate / float(total_prediction)
    print("Validation Accuracy: {}".format(acc))
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
        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        # user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracies.append(item_acc)
        # accuracies.append(user_acc)
        if item_acc > best_accuracy:
            best_accuracy = item_acc
            kstar = k
        # if user_acc > best_accuracy:
        #     best_accuracy = user_acc
        #     kstar = k

    pyplot.plot(ks, accuracies)
    print(kstar, best_accuracy)
    pyplot.title("Accuracy of Item-based collaborative filtering")
    pyplot.xlabel("k")
    pyplot.ylabel("Accuracy")
    pyplot.show()
    # pyplot.savefig('user_based.png')
    print("Test accuracy: {}".
          format(knn_impute_by_item(sparse_matrix, test_data, kstar)))

    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
