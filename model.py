import torch
from torch.nn import Linear, Module, Dropout, Conv1d, MaxPool1d
import torch.nn.functional as F
import numpy as np


class MLP(Module):
    def __init__(self, mnist_flag=False):
        super(MLP, self).__init__()
        self.batch_size = 20
        self.mnist_flag=mnist_flag
        if mnist_flag:
            self.linear = Linear(784, 128)
        else:
            self.linear = Linear(95, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 3)
        # self.linear4 = Linear(64, 3)
        self.dropout = Dropout(0.4)

    def forward(self, out):
        if self.mnist_flag: out = out.view(-1, 28*28)
        out = F.relu(self.linear(out))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        out = self.dropout(out)
        out = self.linear3(out)
        # out = self.dropout(out)
        # out = self.linear4(out)
        return out

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv1d(1, 6, 5)
        self.conv2 = Conv1d(6, 8, 3)
        self.pool = MaxPool1d(2)
        self.dropout = Dropout(0.4)
        self.flatten = torch.nn.Flatten()
        self.fc1 = Linear(270, 64) #without conv2 layer
        self.fc2 = Linear(64, 3)

    def forward(self, out):
        out = out.unsqueeze(1)
        out = F.relu(self.conv1(out))
        out = self.dropout(out)
        # out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        return out

class SVM2:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.constraint_list = []

    def _init_weights_bias(self, X, w_init):
        n_features = X.shape[1]
        if w_init:
            self.w = np.random.rand(n_features)
        else:
            self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        # return y
        return np.where(y <= 0, -1, 1)

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

    def fit(self, X, y, w_init=False):
        self._init_weights_bias(X, w_init)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                self.constraint_list.append(constrain)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)

    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        # return prediction
        return np.where(prediction == -1, 0, 1)

    def accuracy(self, y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

class MultiClassSVM():
    def __init__(self, learning_rate=1e-2, lambda_param=1e-1, num_iters=1500):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.num_classes = 3
        self.loss = 0
        self.lambda_param = lambda_param
        self.delta_param = 1.0
        self.learning_rate = learning_rate
        self.losses = []
        self.num_iters = num_iters

    def _initialize_variables(self, X, y):
        self.X = X
        self.y = y
        self.w = np.zeros((X.shape[1], self.num_classes)) #96x3
        self.b = 0.0
        self.dw = np.zeros(self.w.shape)
        self.db = 0.0


    def _compute_scores(self, x, y, W):
        return np.dot(x, W) #result size: 1 x 3

    def _compute_loss(self, x, y, W):#on each sample
        scores = np.dot(x,W)
        margin = np.maximum(0, scores - scores[y] + self.delta_param)
        margin[y] = 0
        loss_i = np.sum(margin)
        return loss_i

    def _compute_gradients(self, x, y, loss):
        if y == 0:
            j = [1,2]
        elif y == 1:
            j = [0,2]
        else:
            j = [0,1]
        self.dw[:,y] -= -loss * x
        for jx in j:
            self.dw[:,jx] += loss * x
            self.dw[:,jx] += loss * x



    def _update_weights(self, dw):
        self.w -= self.learning_rate * dw
        # self.b -= self.learning_rate * db

    def predict(self, X):
        estimate = np.dot(X, self.w)
        prediction = []
        for i in range(estimate.shape[0]):
            prediction.append(np.argmax(estimate[i]))
        prediction = np.asarray(prediction)
        return prediction

    # def accuracy(self, y_true, y_pred):
    #     acc = np.sum(y_true == y_pred) / len(y_true)
    #     return acc


    def fit(self, X, y):
        self._initialize_variables(X, y)
        y_int = y.astype(np.int64)
        n_train = X.shape[0]
        # losses = []
        for i in range(self.num_iters):
            #### Vectorized
            n_train = X.shape[0]
            scores = X.dot(self.w)
            correct_scores = scores[np.arange(n_train), y_int]
            margins = np.maximum(0, scores - correct_scores[:, np.newaxis]+self.delta_param)
            margins[np.arange(n_train), y_int] = 0
            loss = np.sum(margins)
            loss/= n_train
            loss += 0.5 * self.lambda_param * np.sum(self.w * self.w)
            self.losses.append(loss)
            X_mask = np.zeros(margins.shape)
            X_mask[margins > 0] = 1
            count = np.sum(X_mask, axis = 1)
            X_mask[np.arange(n_train), y_int] = -count
            self.dw = X.T.dot(X_mask)
            self.dw /= n_train
            self.dw += self.lambda_param*self.w
            self.w -= self.learning_rate * self.dw
            # ### Non-Vectorized
            # for idx in range(X.shape[0]):
            #     self.loss += self._compute_loss(X[idx], y[idx], self.w)
            #     self._compute_gradients(X[idx],y[idx], self.loss)
            # self.dw /= X.shape[0]
            # self.dw += self.lambda_param*self.w #doubtful
            # if i % 10 == 0:
            #     self.loss+=self.lambda_param
            #     self._update_weights(self.dw)




