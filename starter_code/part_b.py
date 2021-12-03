import math
import json
from random import random

from sklearn.impute import KNNImputer
from sklearn.neighbors import RadiusNeighborsClassifier

from starter_code.utils import *
import numpy as np
import ast


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
    Returns a list of tuples holding the sparse matrix and data for each dataset.
    """
    matrices = []
    # for each sampled dataset, create a sparse matrix
    for ds in datasets:
        data = {"user_id": [], "question_id": [], "is_correct": []}
        for tup in ds:
            data["question_id"].append(tup[0])
            data["user_id"].append(tup[1])
            data["is_correct"].append(tup[2])

        mat = np.empty((542, 1774))
        mat.fill(math.nan)
        mat[data["user_id"], data["question_id"]] = data["is_correct"]
        matrices.append(mat.T)

    return matrices


def load_meta(path):
    """
    Returns a dict mapping questions to the average of their subject_id and
    a dict mapping averages to a list of question ids with those averages.
    """
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    ques_data = {}
    avg_data = {}
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subjects = ast.literal_eval(row[1])
                avg = np.mean(np.array(subjects))
                ques_data[int(row[0])] = avg
                if avg not in avg_data:
                    avg_data[avg] = []
                avg_data[avg].append(int(row[0]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return ques_data, avg_data


def sort_by_avg(train_data, avg_data, sparse_mat):
    """
    Returns a sparse matrix, where the question entries are sorted by the average of the sum of their
    subjects in ascending order, and a list of the question ids sorted in ascending order by their average.
    """
    sorted_avgs = sorted(avg_data)
    sorted_ques, sorted_sample_ques = [], []

    # sort the questions by average
    for avg in sorted_avgs:
        sorted_ques += avg_data[avg]

    sorted_mat = np.empty(sparse_mat.shape)
    for i in range(len(sparse_mat)):
        # sort the sparse matrix by the order in sorted_ques
        sorted_mat[i] = sparse_mat[sorted_ques[i]]

        # append the relevant label to sorted_labels

    return sorted_mat, sorted_ques


file_dict = {}


def write_to_file(sparse_mat, ques_data):
    """
    Writes a dictionary into file that maps each sample to a list of averages

    *Parameters*
    sparse_mat: a matrix of question by user
    avg_data: a dictionary mapping each question_id to the average of the sum of its subject ids
    """
    # TODO: DONT PASS IN SORTED SPARSE MATRICES
    global file_dict
    # file_dict = {}
    for i in range(len(sparse_mat)):
        sparse_vec = str(sparse_mat[i])

        if sparse_vec not in file_dict.keys():
            file_dict[sparse_vec] = []
        file_dict[sparse_vec].append(ques_data[i])

    # write to file
    # with open('distances.txt', 'w') as file:
    #     file.write(json.dumps(file_dict))


def dist(x, y, **kwargs):
    """
    Custom distance function for KNNImputer.
    """
    # with open('distances.txt', 'r') as file:
    #     file_str = file.read()
    #     file_dict = json.loads(file_str)

    lst_x = file_dict[str(x)]
    lst_y = file_dict[str(y)]
    avg_x = lst_x[0]
    avg_y = lst_y[0]

    return abs(avg_x - avg_y)


# def generate_models(train_data, valid_data, avg_data, sparse_matrices, ks):
#     """
#     Train and fit data for the three different models. Return their accuracies.
#     """
#     models_acc = {'model 1': [], 'model 2': [], 'model 3': []}
#     models = []
#
#     sorted_dataset = []
#     # sort the matrices
#     for mat in sparse_matrices:
#         sorted_dataset.append(sort_by_avg(train_data, avg_data, mat))
#
#     for i in range(len(ks)):
#         # m = RadiusNeighborsClassifier(radius=rads[i])
#         m = KNNImputer(n_neighbors=ks[i])
#         models.append(m)
#         models_acc['model {}'.format(i + 1)] = fit_predict(valid_data, m, sorted_dataset)
#     return models, models_acc


def generate_models(train_data, valid_data, avg_data, sparse_matrices, ks):
    """
       Train and fit data for the three different models. Return their accuracies.
       """


def fit_predict(val_data, model, sorted_dataset):
    """
    Fit and predict the given data and return the accuracies.
    """
    accuracies = []
    for mat, ques in sorted_dataset:
        predictions = model.fit_transform(mat)
        print(predictions)
        accuracies.append(evaluate(val_data, predictions, ques))

    return accuracies


def evaluate(valid_data, mat):
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
    return total_accurate / float(total_prediction)


# def evaluate(val_data, mat, sorted_ques, threshold=0.5):
#     total_prediction = 0
#     total_accurate = 0
#     for i in range(len(val_data["is_correct"])):
#         cur_user_id = val_data["user_id"][i]
#         cur_question_id = sorted_ques[i]
#         if mat[cur_question_id, cur_user_id] >= threshold and val_data["is_correct"][i]:
#             total_accurate += 1
#         if mat[cur_question_id, cur_user_id] < threshold and not val_data["is_correct"][i]:
#             total_accurate += 1
#         total_prediction += 1
#     return total_accurate / float(total_prediction)


def main():
    train_path = os.path.join("./data", "train_data.csv")
    meta_path = os.path.join("./data", "question_meta.csv")

    train_data = load_data(train_path)
    ques_data, avg_data = load_meta(meta_path)
    # train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    datasets = perform_bagging(train_data, 10)
    matrices = sparse_matrix_convert(datasets)

    write_to_file(matrices[0], ques_data)
    print('written')
    m1 = KNNImputer(n_neighbors=19, metric=dist)
    print('created model')
    predictions = m1.fit_transform(matrices[0])
    print('predicted: ', predictions)
    print(evaluate(val_data, predictions))
    # models, accuracies = generate_models(train_data, val_data, avg_data, matrices, ks=[19, 21, 23])

    # combined = accuracies["model 1"] + accuracies["model 2"] + accuracies["model 3"]
    # avg_val = sum(combined) / len(combined)
    # print("Average validation accuracy: " + str(avg_val))
    #
    # test_acc = []
    # for model in models:
    #     test_acc += fit_predict(test_data, model, matrices)
    # avg_test = sum(test_acc) / len(test_acc)
    # print("Average test accuracy: " + str(avg_test))


if __name__ == "__main__":
    main()
