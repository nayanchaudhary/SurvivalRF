"""
================================================================
Plot the decision surface of a decision tree on the iris dataset
================================================================

Plot the decision surface of a decision tree trained on pairs
of features of the iris dataset.

See :ref:`decision tree <tree>` for more information on the estimator.

For each pair of iris features, the decision tree learns decision
boundaries made of combinations of simple thresholding rules inferred from
the training samples.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
df = pd.read_csv('./iris.csv')
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target
y = df.petal_width.astype('int')
X = df.drop('petal_width', axis=1)
X['species_cat'] = pd.Categorical(X.species).codes
Z = X.drop('species', axis=1)
Z['species_cat'] = pd.Categorical(Z['species_cat'])
# print(X.species.unique())
# print(Z.dtypes)
# print(y)

clf = DecisionTreeClassifier(criterion='gini').fit(Z[:100], y[:100])
y_hat = clf.predict(Z[100:])
print(y_hat)
print(y[100:])
################### OLD CODE ####################
#     # Train
#     clf = DecisionTreeClassifier(criterion='gini').fit(X, y)

#     # Plot the decision boundary
#     plt.subplot(2, 3, pairidx + 1)

#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#     plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

#     plt.xlabel(iris.feature_names[pair[0]])
#     plt.ylabel(iris.feature_names[pair[1]])

#     # Plot the training points
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                     cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

# # plt.suptitle("Decision surface of a decision tree using paired features")
# plt.suptitle("New Title that I made")
# plt.legend(loc='lower right', borderpad=0, handletextpad=0)
# plt.axis("tight")
# plt.show()
