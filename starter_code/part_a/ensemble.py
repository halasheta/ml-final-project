# TODO: complete this file.
import math

from sklearn.impute import KNNImputer

from starter_code.utils import *
import numpy as np


def load_data(path):
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    data = []
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data.append((int(row[0]), int(row[1]), int(row[2])))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def perform_bagging(data, m):
    """
    Returns a list of m datasets sampled from data.
    """
    datasets = []
    arr = np.empty(len(data), dtype=object)
    arr[:] = data
    for i in range(m):
        datasets.append(np.random.choice(arr, len(data)))
    return datasets


def sparse_matrix_convert(datasets):
    """
    Returns a list of sparse matrices for each dataset.
    """
    matrices = []
    for ds in datasets:
        data = {"user_id": [], "question_id": [], "is_correct": []}
        for tup in ds:
            data["question_id"].append(tup[0])
            data["user_id"].append(tup[1])
            data["is_correct"].append(tup[2])

        mat = np.empty((542, 1774))
        mat.fill(math.nan)
        mat[data["user_id"], data["question_id"]] = data["is_correct"]
        matrices.append(mat)

    return matrices


def generate_models(valid_data, sparse_matrices, ks):
    """
    Train and fit data for the three different models. Return their accuracies.
    """
    models_acc = {'model 1': [], 'model 2': [], 'model 3': []}
    models = []
    for i in range(len(ks)):
        m = KNNImputer(n_neighbors=ks[i])
        models.append(m)
        models_acc['model {}'.format(i + 1)] = fit_predict(valid_data, m, sparse_matrices)
    return models, models_acc


def fit_predict(data, model, sparse_matrices):
    """
    Fit and predict the given data and return the accuracies.
    """
    accuracies = []
    for mat in sparse_matrices:
        predictions = model.fit_transform(mat)
        accuracies.append(sparse_matrix_evaluate(data, predictions))

    return accuracies


def main():
    train_path = os.path.join("../data", "train_data.csv")

    train_data = load_data(train_path)
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    datasets = perform_bagging(train_data, 10)
    matrices = sparse_matrix_convert(datasets)

    models, accuracies = generate_models(val_data, matrices, ks=[9, 11, 21])
    combined = accuracies["model 1"] + accuracies["model 2"] + accuracies["model 3"]
    avg_val = sum(combined) / len(combined)
    print("Average validation accuracy: " + str(avg_val))

    test_acc = []
    for model in models:
        test_acc += fit_predict(test_data, model, matrices)
    avg_test = sum(test_acc) / len(test_acc)
    print("Average test accuracy: " + str(avg_test))


if __name__ == "__main__":
    main()
