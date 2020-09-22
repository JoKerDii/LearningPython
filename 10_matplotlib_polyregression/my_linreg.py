import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import animation
from IPython.display import HTML


class MyLinearRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def ones_augment_to_left(X):
        X = np.array(X)
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])

    @staticmethod
    def gradient_descent(X, y, n_iters=1000, alpha=0.001,
                         weight=None, debug=False):
        # if we had an computated weight, we update this old weight
        w = weight
        # if there is no predefined weight, we init a weight
        if w is None:
            # create a random vector of size= num. of columns in X
            w = np.random.rand(X.shape[1])

        for i in range(1, n_iters + 1):
            ###### write your code below ######
            # update w to a new value according to the old value of w
            model_output = y_pred = X.dot(w)
            deviation = model_output - y
            update = - 2 * alpha * deviation.dot(X) / X.shape[0]
            w = w + update
            if debug:
                if i % (n_iters / 20) == 0:
                    print('iter:%d \ttraining MSE=%.3f \tMAE=%.3f \tr^2=%.3f' % (
                        i,
                        np.linalg.norm(deviation),
                        np.linalg.norm(deviation, ord=1),
                        r2_score(y, y_pred)
                    )
                          )
        ###### write your code above ######
        return w

    @staticmethod
    def closed_form(X, y):
        product = np.dot(X.T, X)
        theInverse = np.linalg.inv(product)
        return np.dot(np.dot(theInverse, X.T), y)

    def fit(self, X_train, y_train, method='closed form', **kwargs):
        X = self.ones_augment_to_left(X_train)
        y = np.array(y_train)

        if method == 'closed form':
            self.w = self.closed_form(X, y)
        elif method == 'gradient descent':
            self.w = self.gradient_descent(X, y, **kwargs)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = self.ones_augment_to_left(X_test)
        return augX_test.dot(self.w)


last_weight = None


def animated_regression(X, y,
                        true_coef=None,
                        true_bias=None,
                        alpha=0.02,
                        n_iters=1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    # close all matplotlib output
    plt.close('all')

    # 新建一个figure, 视频基于figure
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)

    # 代表回归模型的直线
    line, = ax.plot([], [],
                    color='blue',
                    linewidth=3, label='Fitted Line')

    # 视频的背景
    def init():
        line.set_data([], [])
        ax.scatter(X_train[:, 0], y_train,
                   color='purple', marker='^',
                   alpha=0.5, label='Train Data')

        ax.scatter(X_test[:, 0], y_test,
                   color='blue',
                   alpha=0.5, label='Test Data')
        ax.set_title('\n'.join(np.arange(5).astype(str)))
        fig.tight_layout()
        return (line,)

    # 视频的每一帧
    def animate(i):
        global last_weight
        n_iters = i
        mlr = MyLinearRegression()

        # 每次回归继续使用上次回归的权重, 只迭代一次
        # weight 为 None 时fit方法会自动初始化权重
        mlr.fit(X_train, y_train,
                n_iters=1,
                alpha=alpha,
                method='gradient descent',
                weight=last_weight)

        # 将本次权重设为 "上次回归的权重"
        last_weight = mlr.w

        y_pred = mlr.predict(X_test)
        y_train_pred = mlr.predict(X_train)

        line.set_data(X_test[:, 0], y_pred)

        axtitle = []

        axtitle.append('n_iter=%s, alpha=%s' % (n_iters, alpha))
        axtitle.append('model w=%s' % (mlr.w))

        if not (true_bias is None or true_coef is None):
            axtitle.append('true w=[%s, %s]' % (true_bias, true_coef))

        axtitle.append('training MSE=%.2f MAE=%.2f r^2=%.4f' % (
            np.linalg.norm(y_train - y_train_pred),
            np.linalg.norm(y_train - y_train_pred, ord=1),
            r2_score(y_train, y_train_pred)
        ))
        axtitle.append('testing MSE=%.2f MAE=%.2f r^2=%.4f' % (
            mean_squared_error(y_test, y_pred),
            mean_absolute_error(y_test, y_pred),
            r2_score(y_test, y_pred)))
        ax.set_title('\n'.join(axtitle))
        return (line,)

    # call the animator. blit=True means only re-draw the parts
    # that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_iters, interval=20, blit=True)
    ax.legend()
    return HTML(anim.to_html5_video())


def steps_of_regression(X, y, steps=(1, 2, 5, 10, 20), show_scores=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    alpha = 0.1
    fig = plt.figure(figsize=(5, 15))
    for i, n in enumerate(steps):
        y_pred = MyLinearRegression().fit(
            X_train, y_train,
            n_iters=n,
            alpha=alpha,
            method='gradient descent').predict(X_test)

        ax = fig.add_subplot(len(steps), 1, i + 1)
        print(len(steps), i + 1, n)

        ax.set_xlabel('X[:, 0]')

        ax.set_ylabel('y')

        ax.scatter(X_train[:, 0], y_train,
                   color='purple', marker='^',
                   alpha=0.5, label='Train Data')

        ax.scatter(X_test[:, 0], y_test,
                   color='blue',
                   alpha=0.5, label='Test Data')

        ax.plot(X_test[:, 0], y_pred,
                color='blue',
                linewidth=3, label='Fitted Line')
        if show_scores:
            ax.set_title('n_iter=%s, alpha=%s, MSE=%.2f, r^2=%.2f' % (
                n, alpha,
                mean_squared_error(y_test, y_pred),
                r2_score(y_test, y_pred)
            ))
        else:
            ax.set_title('n_iter=%s, alpha=%s' % (n, alpha))
        ax.legend()
    return fig


true_bias = 5
alpha = 0.02

iris = datasets.load_iris()
X, y = iris.data, iris.target

X = X[y <= 1]
X = X[:, [0]]
y = y[y <= 1]

fig = steps_of_regression(X, y, steps=[1, 50, 100, 200, 300])
fig.tight_layout()
plt.show()
