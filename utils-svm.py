import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def get_clients(class1, class2, class3, n_clients=2):
    clients_X = []
    clients_y = []

    clientsXtest = []
    clientsYtest = []

    clusters_1 = KMeans(n_clusters=n_clients, random_state=0).fit_predict(class1)
    clusters_2 = KMeans(n_clusters=n_clients, random_state=0).fit_predict(class2)
    clusters_3 = KMeans(n_clusters=n_clients, random_state=0).fit_predict(class3)

    for i in range(n_clients):
        X_train0, X_test0, y_train0, y_test0 = train_test_split(class1[clusters_1 == i],
                                                                np.zeros((class1[clusters_1 == i].shape[0],)),
                                                                test_size=0.2)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(class2[clusters_2 == i],
                                                                np.ones((class2[clusters_2 == i].shape[0],)),
                                                                test_size=0.2)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(class3[clusters_3 == i],
                                                                -1 * np.ones((class3[clusters_2 == i].shape[0],)),
                                                                test_size=0.2)

        clients_X.append([X_train0, X_train1, X_train2])
        clients_y.append([y_train0, y_train1, y_train2])

        clientsXtest.extend([X_test0, X_test1, X_test2])
        clientsYtest.extend([y_test0, y_test1, y_test2])

    X_test = np.concatenate(clientsXtest, axis=0)
    y_test = np.concatenate(clientsYtest, axis=0)

    return clients_X, clients_y, X_test, y_test

def get_total_from_clients(clients_X,clients_y):
  x_train0 = [i[0] for i in clients_X]
  x_train0 = np.concatenate(x_train0, axis=0)
  x_train1 = [i[1] for i in clients_X]
  x_train1 = np.concatenate(x_train1, axis=0)
  y_train0 = [i[0] for i in clients_y]
  y_train0 = np.concatenate(y_train0, axis=0)
  y_train1 = [i[1] for i in clients_y]
  y_train1 = np.concatenate(y_train1, axis=0)

  return ([x_train0,x_train1],[y_train0,y_train1])

def create_kmeans_clusters(X, Y, n_clusters = 3, random_state = 0):
  clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)
  result = []
  for i in range(n_clusters):
    result.append(X[clusters == i])
    result.append(Y[clusters == i])
  return tuple(result)
