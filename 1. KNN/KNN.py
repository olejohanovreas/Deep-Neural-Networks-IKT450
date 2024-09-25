import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# function that returns euclidian distance between two points
def euclidian_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# actual knn implementation
def knn(X_train, Y_train, query_point, k):
    distances = []
    # calculating distance from our query point to all other points and store them
    for i in range(len(X_train)):
        dist = euclidian_distance(query_point, X_train[i])
        distances.append((dist, Y_train[i]))

    # sort list by distance and get the k closest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    # majority voting
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    predict_label = Counter(k_nearest_labels).most_common(1)[0][0]

    return predict_label


def evaluate_knn(X_train, Y_train, X_val, Y_val, k):
    TP = TN = FP = FN = 0
    squared_errors = 0

    # counting True Positives, True Negatives, False Positives, and False Negatives
    for i in range(len(X_val)):
        query_point = X_val[i]
        actual_label = Y_val[i]
        predict_label = knn(X_train, Y_train, query_point, k)

        if predict_label == 1 and actual_label == 1:
            TP += 1
        elif predict_label == 0 and actual_label == 0:
            TN += 1
        elif predict_label == 1 and actual_label == 0:
            FP += 1
        elif predict_label == 0 and actual_label == 1:
            FN += 1

        squared_errors += (predict_label - actual_label) ** 2

    # calculating metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    mse = squared_errors / len(X_val)

    print(f"\nValue of k: {k}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Confusion Matrix: \nTP={TP} TN={TN} \nFP={FP} FN={FN}")

    return accuracy, precision, recall, f1_score, mse, TP, TN, FP, FN


def knn_iterative(X_train, Y_train, X_val, Y_val, k_values):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    mses = []

    # running the knn model with different sizes of k
    for k in k_values:
        accuracy, precision, recall, f1_score, mse, _, _, _, _ = evaluate_knn(X_train, Y_train, X_val, Y_val, k)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        mses.append(mse)

    font = {'size': 22}
    plt.rc('font', **font)
    plt.figure(figsize=(14, 8))

    plt.plot(k_values, accuracies, label="Accuracy")
    plt.plot(k_values, precisions, label="Precision")
    plt.plot(k_values, recalls, label="Recall")
    plt.plot(k_values, f1_scores, label="F1 Score")
    plt.plot(k_values, mses, label="MSE")

    plt.title("KNN Model Metrics for Different Values of k")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # load dataset and split it into input and output
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, :-1]
    Y = dataset[:, -1]

    # shuffle dataset
    np.random.seed(7)
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]

    # split 80/20
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]

    # run the model
    knn_iterative(X_train, Y_train, X_val, Y_val, range(1, 300))
