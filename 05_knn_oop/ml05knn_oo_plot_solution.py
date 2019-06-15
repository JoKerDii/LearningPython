import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris


def l2norm(u):
    return np.sqrt(np.sum(u) ** 2)


def euclidean_dist(u, v):
    return l2norm(u - v)


def most_frequent(ar):
    u, indices = np.unique(ar, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        predictions = []
        for i, unknown_sample in enumerate(X_test):
            neighbours = []
            for j, observed_sample in enumerate(self.X_train):
                distance = euclidean_dist(unknown_sample, observed_sample)
                neighbours.append([distance, self.y_train[j]])
            neighbours = np.array(neighbours)
            knn = neighbours[neighbours[:, 0].argsort()][:self.k]
            predictions.append(most_frequent(knn[:, 1]))
        return np.ravel(predictions)


def run():
    data = load_iris()
    data.keys()

    x = data.data
    y = data.target
    x = x[:, [0, 1]]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y)
    yn = le.transform(y)

    # 用train_test_split 将数据分为训练集和测试集
    # test_size 测试集的占所有数据的份额， e.g. test_size=0.5 50%的数据为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # 用线性svm拟合训练数据
    clf = svm.SVC(kernel='poly', C=0.0001).fit(x_train, y_train)
    clf = KNN(k=3).fit(x_train, y_train)
    # 预测值
    y_pred = clf.predict(x_test)

    # 进行一次模型校验
    # 模型误差，正确率
    from sklearn import metrics

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    # # 交叉验证
    # from sklearn.model_selection import cross_val_score
    # cvscore = cross_val_score(svm.SVC(kernel='linear'), x, y=y)
    # print(cvscore)

    from sklearn import neighbors

    plot_bounds2D(x_train, y_train, clf)
    plt.show()


def plot_bounds2D(X, y, clf):
    # create all pari-wise combinations of
    # x0's possible values and x1's possible values
    #
    # read here for more details:
    # https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python

    d0_possible_values = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1)
    d1_possible_values = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1)

    d0, d1 = np.meshgrid(d0_possible_values, d1_possible_values)

    # create data set with d0 and d1 as two columns
    dots = np.c_[d0.ravel(), d1.ravel()]

    # num. rows = 1, num. cols = 1, plotting 1st subplot
    plt.subplot(1, 1, 1)

    plt.subplots_adjust(wspace=0.6, hspace=0.4)
    coloured_dots = clf.fit(X[:, [0, 1]], y).predict(dots)

    # Put the result into a color plot
    coloured_dots = coloured_dots.reshape(d0.shape)
    plt.contourf(d0, d1, coloured_dots, cmap=plt.cm.coolwarm, alpha=0.8)
    # plt.pcolormesh(d0, d1, coloured_dots, cmap=plt.cm.coolwarm)

    # Plot also the training points
    plt.xlim(d0.min(), d0.max())
    plt.ylim(d1.min(), d1.max())

    # plot the line, the points, and the nearest vectors to the plane
    plt.xlabel('X[:, 0]')
    plt.ylabel('X[:, 1]')

    for yy in np.unique(y):
        plt.scatter(x=X[:, 0][y == yy], y=X[:, 1][y == yy], label='y_train==' + str(yy))
    plt.legend()


if __name__ == '__main__':
    run()
