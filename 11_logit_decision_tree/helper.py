import numpy as np
from matplotlib import pyplot as plt


def plot_bounds2D(X, y, clf, feature_pair=None, precision=0.1, clf_fitted=False):
    # create all pari-wise combinations of
    # x0's possible values and x1's possible values
    #
    # read here for more details:
    # https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python

    if feature_pair is None:
        feature_pair = (0, 1)
    f0, f1 = feature_pair

    d0_possible_values = np.arange(X[:, f0].min() - 1, X[:, f0].max() + 1, precision)
    d1_possible_values = np.arange(X[:, f1].min() - 1, X[:, f1].max() + 1, precision)

    d0, d1 = np.meshgrid(d0_possible_values, d1_possible_values)

    # create data set with d0 and d1 as two columns
    dots = np.c_[d0.ravel(), d1.ravel()]

    # num. rows = 1, num. cols = 1, plotting 1st subplot
    plt.subplot(1, 1, 1)

    plt.subplots_adjust(wspace=0.6, hspace=0.4)
    if clf_fitted:
        dots_predicted = clf.predict(dots)
    else:
        dots_predicted = clf.fit(X[:, feature_pair], y).predict(dots)

    # convert possible string labels to numerical label for plotting
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(dots_predicted)
    coloured_dots = le.transform(dots_predicted)

    # Put the result into a color plot
    coloured_dots = coloured_dots.reshape(d0.shape)

    plt.contourf(d0, d1, coloured_dots, cmap=plt.cm.coolwarm, alpha=0.8)
    #     plt.pcolormesh(d0, d1, coloured_dots, cmap=plt.cm.coolwarm)

    # Plot also the training points
    plt.xlim(d0.min(), d0.max())
    plt.ylim(d1.min(), d1.max())

    # plot the line, the points, and the nearest vectors to the plane
    plt.xlabel('plotXX[:, 0]')
    plt.ylabel('plotXX[:, 1]')

    for yy in np.unique(y):
        plt.scatter(x=X[:, f0][y == yy], y=X[:, f1][y == yy], label='y_train==' + str(yy))
    plt.legend()


def plot_bounds2D_on_axis(X, y, clf, ax, feature_pair=None, precision=0.1, clf_fitted=False):
    # create all pari-wise combinations of
    # x0's possible values and x1's possible values
    #
    # read here for more details:
    # https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python

    if feature_pair is None:
        feature_pair = (0, 1)
    f0, f1 = feature_pair

    d0_possible_values = np.arange(X[:, f0].min() - 1, X[:, f0].max() + 1, precision)
    d1_possible_values = np.arange(X[:, f1].min() - 1, X[:, f1].max() + 1, precision)

    d0, d1 = np.meshgrid(d0_possible_values, d1_possible_values)

    # create data set with d0 and d1 as two columns
    dots = np.c_[d0.ravel(), d1.ravel()]

    if clf_fitted:
        dots_predicted = clf.predict(dots)
    else:
        dots_predicted = clf.fit(X[:, feature_pair], y).predict(dots)

    # convert possible string labels to numerical label for plotting
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(dots_predicted)
    coloured_dots = le.transform(dots_predicted)

    # Put the result into a color plot
    coloured_dots = coloured_dots.reshape(d0.shape)

    ax.contourf(d0, d1, coloured_dots, cmap=plt.cm.coolwarm, alpha=0.8)
    #     plt.pcolormesh(d0, d1, coloured_dots, cmap=plt.cm.coolwarm)

    # Plot also the training points
    ax.set_xlim(d0.min(), d0.max())
    ax.set_ylim(d1.min(), d1.max())

    # plot the line, the points, and the nearest vectors to the plane
    ax.set_xlabel('plotXX[:, 0]')
    ax.set_ylabel('plotXX[:, 1]')

    for yy in np.unique(y):
        ax.scatter(x=X[:, f0][y == yy], y=X[:, f1][y == yy], label='y_train==' + str(yy))


def getRoundedThresholdv1(a, MinClip):
    return round(float(a) / MinClip) * MinClip


def getRoundedThresholdv2(a, MinClip):
    return np.round(np.array(a, dtype=float) / MinClip) * MinClip