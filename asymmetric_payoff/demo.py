import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from asymmetric_payoff.threshold_calculation import calculate_optimal_threshold

n = 500
X_t = np.concatenate((np.random.normal(-1, 1, n), np.random.normal(1, 1, n)), axis=0)
X = X_t.reshape(-1, 1)
y = [0]*n + [1]*n

df = pd.DataFrame()
df['X'] = X_t
df['y'] = y

sns.distplot(X_t[:n], color='r')
sns.distplot(X_t[n:], color='b')
plt.show()

clf = LogisticRegression()
clf.fit(X, y)
y_pred = clf.predict_proba(X)[:,1]

X_plot_t = np.linspace(-4.5, 4.5)
y_plot_positive = clf.predict_proba(X_plot_t.reshape(-1, 1))
plt.plot(X_plot_t, 1- y_plot_positive)
plt.plot(X_plot_t, y_plot_positive)
plt.show()

accuracy_matrix = [[1, 0],
                   [0, 1]]
print(calculate_optimal_threshold(y, y_pred, accuracy_matrix))

costly_false_positives = [[1, -5],
                          [0,  1]]
print(calculate_optimal_threshold(y, y_pred, costly_false_positives))

costly_false_negatives = [[ 1, 0],
                          [-5, 1]]
print(calculate_optimal_threshold(y, y_pred, costly_false_negatives))