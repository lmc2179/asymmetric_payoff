import unittest

import numpy as np

from utilipy.classification import OptimalUtilityThreshold, MaxUtilityClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ACCURACY_MATRIX = np.array([[1, 0], [0, 1]])

class ThresholdCalculationTest(unittest.TestCase):
    def test_accuracy(self):
        t = OptimalUtilityThreshold(np.array([0, 0, 1, 1]),
                                    np.array([0.2, 0.4, 0.45, 0.8]),
                                    ACCURACY_MATRIX).calculate_optimal_threshold()
        self.assertEqual(t, 0.45)

    def test_accuracy_non_unique(self):
        t = OptimalUtilityThreshold(np.array([0, 1, 1, 1]),
                                    np.array([0.2, 0.4, 0.4, 0.8]),
                                    ACCURACY_MATRIX).calculate_optimal_threshold()
        self.assertEqual(t, 0.40)

class WrapperTest(unittest.TestCase):
    def test_accuracy(self):
        n = 5000
        X_t = np.sort(np.concatenate((np.random.normal(-1, 1, n), np.random.normal(1, 1, n)), axis=0))
        X = X_t.reshape(-1, 1)
        y = [0] * n + [1] * n
        m = MaxUtilityClassifier(LogisticRegression(), ACCURACY_MATRIX)
        m.fit(X, y)
        self.assertAlmostEqual(m.threshold_, 0.5, delta=0.1)
        y_predicted = m.predict(X)
        test_lr = LogisticRegression()
        test_lr.fit(X, y)
        y_lr_pred = test_lr.predict(X)
        self.assertGreater(accuracy_score(y, y_predicted),
                           accuracy_score(y, y_lr_pred))

    def test_costly_false_positives(self):
        n = 10000
        X_t = np.concatenate((np.random.normal(-1, 1, n), np.random.normal(1, 1, n)), axis=0)
        X = X_t.reshape(-1, 1)
        y = [0] * n + [1] * n
        m = MaxUtilityClassifier(LogisticRegression(), np.array([[1, -5],
                                                                 [0, 1]]))
        m.fit(X, y)
        self.assertAlmostEqual(m.threshold_, 0.85, delta=0.05)

    def test_costly_false_negatives(self):
        n = 10000
        X_t = np.concatenate((np.random.normal(-1, 1, n), np.random.normal(1, 1, n)), axis=0)
        X = X_t.reshape(-1, 1)
        y = [0] * n + [1] * n
        m = MaxUtilityClassifier(LogisticRegression(), np.array([[1, 0],
                                                                 [-5, 1]]))
        m.fit(X, y)
        self.assertAlmostEqual(m.threshold_, 0.15, delta=0.05)