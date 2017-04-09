import unittest

import numpy as np

from asymmetric_payoff.threshold_calculation import calculate_optimal_threshold


class PrototypeTest(unittest.TestCase):
    def test_accuracy(self):
        t = calculate_optimal_threshold(np.array([0, 0, 1, 1]),
                                        np.array([0.2, 0.4, 0.45, 0.8]),
                                        np.array([[1, 0], [0, 1]]))
        self.assertEqual(t, 0.45)

    def test_accuracy_non_unique(self):
        t = calculate_optimal_threshold(np.array([0, 1, 1, 1]),
                                        np.array([0.2, 0.4, 0.4, 0.8]),
                                        np.array([[1, 0], [0, 1]]))
        self.assertEqual(t, 0.40)