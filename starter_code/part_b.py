from matplotlib import pyplot
from sklearn.impute import KNNImputer
from starter_code.utils import *
import numpy as np
import ast


def load_meta(path):
    """
    Returns a dict mapping questions to the average of their subject_id and
    a dict mapping averages to a list of question ids with those averages.
    """
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    ques_data = {}

    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subjects = ast.literal_eval(row[1])
                avg = np.mean(np.array(subjects))
                ques_data[int(row[0])] = avg

            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return ques_data


def augment(sparse_mat, ques_data):
    """
    Writes a dictionary into file that maps each sample to a list of averages

    :param sparse_mat: a matrix of question by user
    :param ques_data: a dictionary mapping each question_id to the average
     of its subject ids
    """

    augmented_sparse = np.c_[np.ones(1774), sparse_mat]
    for i in range(len(sparse_mat)):
        augmented_sparse[i][0] = ques_data[i]

    return augmented_sparse


def dist(x, y, **kwargs):
    """
    Custom distance function for KNNImputer which finds the absolute difference
    between the average of the subject ids of x and y.
    """
    return abs(y[0] - x[0])


def evaluate(valid_data, mat):
    """
    Evaluates mat predictions against the valid_data
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(valid_data["is_correct"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        if mat[cur_question_id, cur_user_id + 1] >= 0.5 and \
                valid_data["is_correct"][i]:  # added plus 1
            total_accurate += 1
        if mat[cur_question_id, cur_user_id + 1] < 0.5 and not \
                valid_data["is_correct"][i]:  # added plus 1
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def main():
    meta_path = os.path.join("./data", "question_meta.csv")
    sparse = load_train_sparse("./data").toarray()

    ques_data = load_meta(meta_path)
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    a = augment(sparse.T, ques_data)
    val_acc, test_acc = [], []
    ks = [i for i in range(200, 300, 5)]
    for i in ks:
        model = KNNImputer(n_neighbors=i, metric=dist)
        predictions = model.fit_transform(a)
        v_acc = evaluate(val_data, predictions)
        t_acc = evaluate(test_data, predictions)
        val_acc.append(v_acc)
        test_acc.append(t_acc)
        print(i, v_acc, t_acc)
    pyplot.plot(ks, val_acc)
    pyplot.ylabel("Validation accuracies")
    pyplot.xlabel("k")
    pyplot.title("New KNN impute by item accuracy vs k")
    pyplot.show()
    print(ks)
    print("val accuracy: ", val_acc)
    print(val_acc.index(max(val_acc)))
    print("test accuracy: ", test_acc)
    print(test_acc.index(max(test_acc)))




if __name__ == "__main__":
    main()
