from typing import TextIO, List
from sklearn import *
import matplotlib.pyplot as plt
import pandas as pd


def load_data(real, fake) -> list:
    """
    This function loads the sentimental data from a txt file,
    preprocesses it using CountVectorizer, and then splits it
    into 70% training, 15% validation and 15% test samples.
    :param fake: real:
    :type real: TextIO, fake: TextIO
    :return:
    """
    # load the data
    with open(real, "r") as f1:  # load the real data
        real_data = f1.readlines()
    with open(fake, "r") as f2:  # load the fake data
        fake_data = f2.readlines()
    data = real_data + fake_data  # merge the real and fake data

    # process the data
    cv = feature_extraction.text.CountVectorizer()  # initialize a CountVectorizer
    raw = cv.fit_transform(data)  # transform the merged data

    # save the target results
    label = []
    i = 0
    while i < len(real_data):  # 1 represents real news
        label.append(1)
        i += 1
    while i < len(real_data) + len(fake_data):  # 0 represents fake news
        label.append(0)
        i += 1

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(raw, label, test_size=0.15, random_state=0)
    x_train, x_val, y_train, y_val = \
        model_selection.train_test_split(x_train, y_train, test_size=0.15, random_state=0)

    return [x_train, x_test, x_val, y_train, y_test, y_val]


def select_knn_model(dataset: list) -> float:
    """
    This function uses the default metric to train the dataset, plot the
    k-neighbor value against training and testing accuracy and return the
    optimal k value.
    :return:
    """
    k_range = range(1, 21)
    scores = []
    x_train, x_test, x_val, y_train, y_test, y_val = \
        dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
    for k in k_range:
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(x_train, y_train)
        train_pred, valid_pred = knn.predict(x_train), knn.predict(x_val)
        scores.append(metrics.accuracy_score(y_train, train_pred))
        scores.append(metrics.accuracy_score(y_val, valid_pred))
    train_score = scores[:20]
    valid_score = scores[20:]
    plt.plot(k_range, train_score, label='Training Prediction')
    plt.plot(k_range, valid_score, label='Validation Prediction')
    plt.xlabel('K-Neighbor')
    plt.ylabel('Accuracy')
    plt.title('Testing, Validation Accracy on KNN-Classifer')
    plt.legend()
    plt.show()

    for k in k_range:
        if scores[k - 1] == max(valid_score):
            break

    opt_knn = neighbors.KNeighborsClassifier(k)
    opt_pred = opt_knn.predict(x_test)
    opt_knn.fit(x_train, y_train)
    return metrics.accuracy_score(y_test, opt_pred)


def improved_select_knn_model(dataset: list) -> float:
    """

    :return:
    """
    k_range = range(1, 21)
    scores = []
    x_train, x_test, x_val, y_train, y_test, y_val = \
        dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
    for k in k_range:
        knn = neighbors.KNeighborsClassifier(k, metric='cosine')
        knn.fit(x_train, y_train)
        train_pred, valid_pred = knn.predict(x_train), knn.predict(x_val)
        scores.append(metrics.accuracy_score(y_train, train_pred))
        scores.append(metrics.accuracy_score(y_val, valid_pred))

    train_score = scores[:20]
    valid_score = scores[20:]
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_score, label='Training Prediction')
    plt.plot(k_range, valid_score, label='Validation Prediction')
    plt.xlabel('K-Neighbor')
    plt.ylabel('Accuracy')
    plt.title('Testing, Validation Accuracy on KNN-Classifer')
    plt.legend()
    plt.savefig('KNN Errors Report')
    plt.show()

    for k in k_range:
        if scores[k - 1] == max(valid_score):
            break

    opt_knn = neighbors.KNeighborsClassifier(k, metric='cosine')
    opt_knn.fit(x_train, y_train)
    opt_pred = opt_knn.predict(x_test)
    return metrics.accuracy_score(y_test, opt_pred)


if __name__ == '__main__':
    real_f = 'C:/Users/conqu/PycharmProjects/csc311/a1/data/clean_real.txt'
    fake_f = 'C:/Users/conqu/PycharmProjects/csc311/a1/data/clean_fake.txt'
    dataset = load_data(real_f, fake_f)
    opt_accuracy = improved_select_knn_model(dataset)
    opt_report = pd.DataFrame({'K-value': [7], 'Test Accuracy': [opt_accuracy]})
    print(opt_report)

