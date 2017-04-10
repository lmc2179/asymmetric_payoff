import numpy as np

def calculate_optimal_threshold(y, y_predicted, payoff_matrix):
    """"""
    y_pred_sorted, y_sorted = zip(*sorted(zip(y_predicted, y)))
    zero_transition_cost = payoff_matrix[0][0] - payoff_matrix[0][1]
    one_transition_cost = payoff_matrix[1][0] - payoff_matrix[1][1]
    possible_thresholds = sorted(list(set(y_pred_sorted)))
    threshold_position = {t: i for i,t in enumerate(possible_thresholds)}
    y_0, y_1 = np.zeros(len(possible_thresholds)), np.zeros(len(possible_thresholds)) # TODO: Better name
    true_counts = [y_0, y_1]
    for y_i, t in zip(y, y_predicted):
        true_counts[y_i][threshold_position[t]] += 1
    initial_utility = sum(true_counts[0] * payoff_matrix[0][1] \
                          + true_counts[1] * payoff_matrix[1][1])
    best_utility = initial_utility
    best_threshold = 0
    current_utility = initial_utility
    for i, t in enumerate(possible_thresholds):
        new_utility = true_counts[0][i-1] * zero_transition_cost \
                    + true_counts[1][i-1] * one_transition_cost \
                    + current_utility
        if new_utility >= best_utility:
            best_utility, best_threshold = new_utility, t
        current_utility = new_utility
    return best_threshold

class AsymmetricPayoffClassifier(object):
    def __init__(self, base_model, payoff_matrix):
        self.model = base_model
        self.payoff_matrix = payoff_matrix
        self.threshold_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        y_predicted = self.model.predict_proba(X)[:,1]
        self.threshold_ = calculate_optimal_threshold(y, y_predicted, self.payoff_matrix)

    def predict(self, X):
        y_predicted_proba = self.model.predict_proba(X)[:,1]
        return np.array([1 if y_p >= self.threshold_ else 0 for y_p in y_predicted_proba])

    def __getattr__(self, item):
        try:
            return getattr(self.model, item)
        except AttributeError:
            raise AttributeError