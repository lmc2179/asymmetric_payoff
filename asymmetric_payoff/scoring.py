import numpy as np

def confusion_matrix(y_true, y_pred, normalize=False):
    cm = np.zeros((2, 2))
    for y, y_p in zip(y_true, y_pred):
        cm[y][y_p] += 1
    if normalize:
        cm /= len(y_true)
    return cm

def total_payoff(y_true, y_pred, payoff_matrix):
    cm = confusion_matrix(y_true, y_pred, normalize=False)
    return np.sum(cm * payoff_matrix)

def mean_payoff(y_true, y_pred, payoff_matrix):
    if not isinstance(payoff_matrix, np.ndarray):
        payoff_matrix = np.array(payoff_matrix)
    cm = confusion_matrix(y_true, y_pred, normalize=True)
    return np.sum(cm * payoff_matrix)