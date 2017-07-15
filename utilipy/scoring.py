import numpy as np

def confusion_matrix(y_true, y_pred, normalize=False):
    cm = np.zeros((2, 2))
    for y, y_p in zip(y_true, y_pred):
        cm[y][y_p] += 1
    if normalize:
        cm /= len(y_true)
    return cm

def utility(y_true, y_pred, utility_matrix):
    return np.array([utility_matrix[t][p] for t, p in zip(y_true, y_pred)])

def total_utility(y_true, y_pred, utility_matrix):
    cm = confusion_matrix(y_true, y_pred, normalize=False)
    return np.sum(cm * utility_matrix)

def mean_utility(y_true, y_pred, utility_matrix):
    if not isinstance(utility_matrix, np.ndarray):
        utility_matrix = np.array(utility_matrix)
    cm = confusion_matrix(y_true, y_pred, normalize=True)
    return np.sum(cm * utility_matrix)

def regret(y_true, y_pred, utility_matrix):
    return np.array([utility_matrix[t][t] - utility_matrix[t][p] for t, p in zip(y_true, y_pred)])