import unittest
from optimal_threshold_classifier.scoring import confusion_matrix, total_payoff, mean_payoff

class TestConfusionMatrix(unittest.TestCase):
    def test_each_entry(self):
        y, y_pred = [0, 0, 1, 1], [0, 1, 0, 1]
        cm = confusion_matrix(y, y_pred, normalize=False)
        expected_cm = [[1, 1],
                       [1, 1]]
        self.assertEqual(cm.tolist(), expected_cm)


    def test_each_entry_normalized(self):
        y, y_pred = [0, 0, 1, 1], [0, 1, 0, 1]
        cm = confusion_matrix(y, y_pred, normalize=True)
        expected_cm = [[0.25, 0.25],
                       [0.25, 0.25]]
        self.assertEqual(cm.tolist(), expected_cm)

class TestPayoff(unittest.TestCase):
    def test_each_entry_accuracy_total(self):
        y, y_pred = [0, 0, 1, 1], [0, 1, 0, 1]
        payoff_matrix = [[1, 0], [0, 1]]
        payoff = total_payoff(y, y_pred, payoff_matrix)
        self.assertEqual(payoff, 2)

    def test_each_entry_accuracy_mean(self):
        y, y_pred = [0, 0, 1, 1], [0, 1, 0, 1]
        payoff_matrix = [[1, 0], [0, 1]]
        payoff = mean_payoff(y, y_pred, payoff_matrix)
        self.assertEqual(payoff, 0.5)