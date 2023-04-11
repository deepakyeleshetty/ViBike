import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SVM:

    def __init__(self, X_train, y_train, X_test, y_test, val=True, val_type='k_fold', val_distribution='balanced', k=5,
                 learning_rate=0.001, lambda_param=0.01, n_iters=1000):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.val_distribution = val_distribution
        self.val = val
        self.val_type = val_type
        self.val_distribution = val_distribution
        self.k = k

        self.w = np.array([])
        self.b = None

    def Gradient_update(self, X_train, y_train, X_val=None, y_val=None):

        n_samples, n_features = X_train.shape
        y_ = np.where(y_train == 0, 1, -1) # this line makes it a 2-class problem with labels -1 and 1

        if self.w.size == 0 and self.b is None:
            self.w = np.zeros(n_features)
            self.b = 0

        w_best = np.zeros(n_features)
        b_best = 0

        acc_list = []
        for i in range(0, self.n_iters):
            for idx, x_i in enumerate(X_train):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 # checks if the sample belongs to class "1"
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

            if i % 10 == 0 and self.val:
                approx_w = np.dot(X_val, self.w) - self.b
                approx_w = np.sign(approx_w)
                res_w = np.where(approx_w < 0, 0, approx_w)

                approx_w_best = np.dot(X_val, w_best) - b_best
                approx_w_best = np.sign(approx_w_best)
                res_w_best = np.where(approx_w_best < 0, 0, approx_w_best)

                if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
                    w_best = copy.deepcopy(self.w)
                    b_best = copy.deepcopy(self.b)
                else:
                    self.w = copy.deepcopy(w_best)
                    self.b = copy.deepcopy(b_best)
                    break

    def Cross_validation(self, val_split):

        if (self.val_distribution == 'balanced'):
            X_train0, X_val0, y_train0, y_val0 = train_test_split(self.X_train[0], self.y_train[0], test_size=val_split)
            X_train1, X_val1, y_train1, y_val1 = train_test_split(self.X_train[1], self.y_train[1], test_size=val_split)

            X_train = np.concatenate((X_train0, X_train1), axis=0)
            y_train = np.concatenate((y_train0, y_train1), axis=0)

            X_val = np.concatenate((X_val0, X_val1), axis=0)
            y_val = np.concatenate((y_val0, y_val1), axis=0)

        elif (self.val_distribution == 'unbalanced'):
            X_train = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
            y_train = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)
            #       X_train = self.X_train
            #       y_train = self.y_train

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)

        X_train, y_train = self.random_shuffle(X_train, y_train)
        self.Gradient_update(X_train, y_train, X_val, y_val)

    def k_fold_cross_validation(self):

        if (self.val_distribution == 'unbalanced'):
            X_train = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
            y_train = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)

            X_train_, X_train0, y_train_, y_train0 = train_test_split(X_train, y_train, test_size=round(1 / self.k, 2),
                                                                      shuffle=True)

            X_train = []
            y_train = []

            X_train.append(copy.deepcopy(X_train0))
            y_train.append(copy.deepcopy(y_train0))
            k = self.k - 1

            X_train0 = np.array_split(X_train_, k)
            y_train0 = np.array_split(y_train_, k)

            for i in range(k):
                X_train.append(X_train0[i])
                y_train.append(y_train0[i])

        elif (self.val_distribution == 'balanced'):
            X_train0 = np.array_split(self.X_train[0], self.k)
            X_train1 = np.array_split(self.X_train[1], self.k)
            y_train0 = np.array_split(self.y_train[0], self.k)
            y_train1 = np.array_split(self.y_train[1], self.k)
            X_train = []
            y_train = []
            for i in range(self.k):
                X_train.append(np.concatenate((X_train0[i], X_train1[i]), axis=0))
                y_train.append(np.concatenate((y_train0[i], y_train1[i]), axis=0))

        if self.w.size == 0 and self.b == None:
            w = np.zeros(self.X_train[0].shape[1])
            b = 0
        else:
            w = copy.deepcopy(self.w)
            b = self.b

        w_list = []
        b_list = []
        acc_list = []
        for i in range(self.k):
            X_train_temp = np.zeros((1, X_train[0].shape[1]))
            y_train_temp = np.array([])

            for j in range(self.k):
                if (j != i):
                    X_train_temp = np.concatenate((X_train_temp, X_train[j]), axis=0)
                    y_train_temp = np.concatenate((y_train_temp, y_train[j]), axis=0)
                else:
                    X_val = X_train[j]
                    y_val = y_train[j]

            X_train_temp = np.delete(X_train_temp, 0, 0)
            X_train_temp, y_train_temp = self.random_shuffle(X_train_temp, y_train_temp)
            self.Gradient_update(X_train_temp, y_train_temp, X_val, y_val)
            print(self.accuracy())
            w_list.append(self.w)
            b_list.append(self.b)

            test_w = np.dot(X_val, self.w) - self.b
            test_w = np.sign(test_w)
            res_val = np.where(test_w < 0, 0, test_w)

            acc_list.append(accuracy_score(y_val, res_val))

            self.w = copy.deepcopy(w)
            self.b = b

        self.w = copy.deepcopy(w_list[acc_list.index(max(acc_list))])
        self.b = b_list[acc_list.index(max(acc_list))]

    def fit(self):
        if self.val_type == 'k_fold' and self.val:
            self.k_fold_cross_validation()

        elif self.val_type == 'cross_val' and self.val:
            self.Cross_validation(0.2)

        elif not self.val:
            X_train = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
            y_train = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)
            #       X_train = self.X_train
            #       y_train = self.y_train
            X_train, y_train = self.random_shuffle(X_train, y_train)
            self.Gradient_update(X_train, y_train)

    def random_shuffle(self, X_train, y_train):
        self.x_tr, self.x_te, self.y_tr, self.y_te = train_test_split(X_train, y_train, test_size=0.5)
        return np.concatenate((self.x_tr, self.x_te), axis=0), np.concatenate((self.y_tr, self.y_te), axis=0)

    def predict(self):
        approx = np.dot(self.X_test, self.w) - self.b
        approx = np.sign(approx)
        return np.where(approx < 0, 0, approx)

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict()) * 100


class Federated_SVM:

    def __init__(self, n_clients=4, val=True, val_type='k_fold', val_distribution='balanced', k=5, learning_rate=0.001,
                 lambda_param=0.01, n_iters=100):
        self.n_clients = n_clients
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.val = val
        self.val_type = val_type
        self.val_distribution = val_distribution
        self.client_distribution = []
        self.k = k
        self.X_test = None
        self.y_test = None
        self.noise = None

    def create_clients(self, X_train, y_train, X_test, y_test):
        self.clients = []
        for i in range(self.n_clients):
            self.client_distribution.append(X_train[i][0].shape[0] + X_train[i][1].shape[0])
            self.clients.append(
                SVM(X_train[i], y_train[i], X_test, y_test, self.val, self.val_type, self.val_distribution, self.k,
                    self.learning_rate, self.lambda_param, self.n_iters))
        self.X_test = copy.deepcopy(X_test)
        self.y_test = copy.deepcopy(y_test)

    def average_aggregator(self, parameter_list):
        w = np.zeros(parameter_list[0].shape[0])
        b = 0
        for i in range(0, 2 * self.n_clients, 2):
            w = np.add(w, parameter_list[i] * self.client_distribution[i // 2] / sum(self.client_distribution))
            b = b + parameter_list[i + 1]
        return (w, b / self.n_clients)

    def fit(self, g_iters, aggregator):
        w_best = np.zeros(self.X_test.shape[1])
        b_best = 0
        for i in range(0, g_iters):
            print('global round', i + 1)
            for j in range(0, self.n_clients):
                if i == 0:
                    self.clients[j].fit()
                else:
                    self.clients[j].w = copy.deepcopy(w_agg)
                    self.clients[j].b = copy.deepcopy(b_agg)
                    self.clients[j].fit()
                print('client', j + 1, self.clients[j].accuracy())
            parameter_list = []
            for k in range(0, self.n_clients):
                parameter_list.append(self.clients[k].w)
                parameter_list.append(self.clients[k].b)

            w_agg, b_agg = aggregator(parameter_list)

            if self.accuracy(w_agg, b_agg) > self.accuracy(w_best, b_best) or i == 0:
                w_best = copy.deepcopy(w_agg)
                b_best = copy.deepcopy(b_agg)
            print('global test acc', self.accuracy(w_best, b_best))

    def predict(self, w, b):
        approx = np.dot(self.X_test, w) - b
        approx = np.sign(approx)
        return np.where(approx < 0, 0, 1)

    def accuracy(self, w, b):
        return accuracy_score(self.y_test, self.predict(w, b)) * 100


class SVM2:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        # return y
        return np.where(y == 0, 1, -1)

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx] * linear_model >= 1

    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db

        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db

    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        # return prediction
        return np.where(prediction >= 0, 0, -1)


class support_vector_machine:
    def __init__(self, C=10, features=2, sigma_sq=0.1, kernel="None"):
        self.C = C
        self.features = features
        self.sigma_sq = sigma_sq
        self.kernel = kernel
        self.weights = np.zeros(features)
        self.bias = 0.

    def __similarity(self, x, l):
        return np.exp(-sum((x - l) ** 2) / (2 * self.sigma_sq))

    def gaussian_kernel(self, x1, x):
        m = x.shape[0]
        n = x1.shape[0]
        op = [[self.__similarity(x1[x_index], x[l_index]) for l_index in range(m)] for x_index in range(n)]
        return np.array(op)

    def loss_function(self, y, y_hat):
        sum_terms = 1 - y * y_hat
        sum_terms = np.where(sum_terms < 0, 0, sum_terms)
        return (self.C * np.sum(sum_terms) / len(y) + sum(self.weights ** 2) / 2)

    def fit(self, x_train, y_train, epochs=1000, print_every_nth_epoch=100, learning_rate=0.01):
        y = y_train.copy()
        x = x_train.copy()
        self.initial = x.copy()

        assert x.shape[0] == y.shape[0], "Samples of x and y don't match."
        assert x.shape[1] == self.features, "Number of Features don't match"

        if (self.kernel == "gaussian"):
            x = self.gaussian_kernel(x, x)
            m = x.shape[0]
            self.weights = np.zeros(m)

        n = x.shape[0]

        for epoch in range(epochs):
            y_hat = np.dot(x, self.weights) + self.bias
            grad_weights = (-self.C * np.multiply(y, x.T).T + self.weights).T

            for weight in range(self.weights.shape[0]):
                grad_weights[weight] = np.where(1 - y_hat <= 0, self.weights[weight], grad_weights[weight])

            grad_weights = np.sum(grad_weights, axis=1)
            self.weights -= learning_rate * grad_weights / n
            grad_bias = -y * self.bias
            grad_bias = np.where(1 - y_hat <= 0, 0, grad_bias)
            grad_bias = sum(grad_bias)
            self.bias -= grad_bias * learning_rate / n
            if ((epoch + 1) % print_every_nth_epoch == 0):
                print("--------------- Epoch {} --> Loss = {} ---------------".format(epoch + 1,
                                                                                      self.loss_function(y, y_hat)))

    def evaluate(self, x, y):
        pred = self.predict(x)
        pred = np.where(pred == -1, 0, 1)
        diff = np.abs(np.where(y == -1, 0, 1) - pred)
        return ((len(diff) - sum(diff)) / len(diff))

    def predict(self, x):
        if (self.kernel == "gaussian"):
            x = self.gaussian_kernel(x, self.initial)
        return np.where(np.dot(x, self.weights) + self.bias > 0, 1, -1)