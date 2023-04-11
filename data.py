import numpy as np
import random
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils import shuffle
import torch
import pandas as pd
import os

def read_client_data(c):
    trainX0 = pd.read_csv(f'RoadData/{c}/asphalt_x.csv', sep = ',', header=0)
    trainX1 = pd.read_csv(f'RoadData/{c}/brick_x.csv', sep=',', header=0)
    trainX2 = pd.read_csv(f'RoadData/{c}/bad_x.csv', sep=',', header=0)

    trainy0 = pd.read_csv(f'RoadData/{c}/asphalt_y.csv', sep=',', header=None)
    train_y0 = trainy0[0].tolist()
    trainy1 = pd.read_csv(f'RoadData/{c}/brick_y.csv', sep=',', header=None)
    train_y1 = trainy1[0].tolist()
    trainy2 = pd.read_csv(f'RoadData/{c}/bad_y.csv', sep=',', header=None)
    train_y2 = trainy2[0].tolist()

    testX0 = pd.read_csv(f'RoadData/{c}/asphalt_test_x.csv', sep=',', header=0)
    testX1 = pd.read_csv(f'RoadData/{c}/brick_test_x.csv', sep=',', header=0)
    testX2 = pd.read_csv(f'RoadData/{c}/bad_test_x.csv', sep=',', header=0)

    testy0 = pd.read_csv(f'RoadData/{c}/asphalt_test_y.csv', sep=',', header=None)
    test_y0 = testy0[0].tolist()
    testy1 = pd.read_csv(f'RoadData/{c}/brick_test_y.csv', sep=',', header=None)
    test_y1 = testy1[0].tolist()
    testy2 = pd.read_csv(f'RoadData/{c}/bad_test_y.csv', sep=',', header=None)
    test_y2 = testy2[0].tolist()

    X = np.vstack((np.array(trainX0), np.array(trainX1), np.array(trainX2)))
    y= np.hstack((np.array(train_y0), np.array(train_y1), -2 * np.array(train_y2)))
    test_X= np.vstack((np.array(testX0), np.array(testX1), np.array(testX2)))
    test_y= np.hstack((np.array(test_y0), np.array(test_y1), -2 * np.array(test_y2)))

    clients_X, clients_y, client_val_X, client_val_y, clients_test_X, clients_test_y = create_tensors(X, y, test_X, test_y)
    
    return clients_X, clients_y, client_val_X, client_val_y, clients_test_X, clients_test_y

def load_test_data(mode):
    test0X0 = pd.read_csv(f'RoadData/0/asphalt_test_x.csv', sep=',', header=0)
    test0X1 = pd.read_csv(f'RoadData/0/brick_test_x.csv', sep=',', header=0)
    test0X2 = pd.read_csv(f'RoadData/0/bad_test_x.csv', sep=',', header=0)

    test1X0 = pd.read_csv(f'RoadData/1/asphalt_test_x.csv', sep=',', header=0)
    test1X1 = pd.read_csv(f'RoadData/1/brick_test_x.csv', sep=',', header=0)
    test1X2 = pd.read_csv(f'RoadData/1/bad_test_x.csv', sep=',', header=0)

    test0y0 = pd.read_csv(f'RoadData/0/asphalt_test_y.csv', sep=',', header=None)
    test0_y0 = test0y0[0].tolist()
    test0y1 = pd.read_csv(f'RoadData/0/brick_test_y.csv', sep=',', header=None)
    test0_y1 = test0y1[0].tolist()
    test0y2 = pd.read_csv(f'RoadData/0/bad_test_y.csv', sep=',', header=None)
    test0_y2 = test0y2[0].tolist()

    test1y0 = pd.read_csv(f'RoadData/1/asphalt_test_y.csv', sep=',', header=None)
    test1_y0 = test1y0[0].tolist()
    test1y1 = pd.read_csv(f'RoadData/1/brick_test_y.csv', sep=',', header=None)
    test1_y1 = test1y1[0].tolist()
    test1y2 = pd.read_csv(f'RoadData/1/bad_test_y.csv', sep=',', header=None)
    test1_y2 = test1y2[0].tolist()

    if mode == 0:
        test_X = np.vstack((np.array(test0X0), np.array(test0X1), np.array(test0X2)))
        test_y = np.hstack((np.array(test0_y0), np.array(test0_y1), -2 * np.array(test0_y2)))
    elif mode == 1:
        test_X = np.vstack((np.array(test1X0), np.array(test1X1), np.array(test1X2)))
        test_y = np.hstack((np.array(test1_y0), np.array(test1_y1), -2 * np.array(test1_y2)))
    else:
        test_X = np.vstack((np.array(test0X0), np.array(test0X1), np.array(test0X2), np.array(test1X0), np.array(test1X1), np.array(test1X2)))
        test_y = np.hstack((np.array(test0_y0), np.array(test0_y1), -2 * np.array(test0_y2), np.array(test1_y0), np.array(test1_y1), -2 * np.array(test1_y2)))

    tens_test_X = torch.tensor(test_X)
    tens_test_y = torch.tensor(test_y).type(torch.LongTensor)
    return tens_test_X, tens_test_y

def load_3classes_mnist(clients):
    transform = transforms.ToTensor()
    # choose the training and test datasets
    all_train_data = datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
    all_test_data = datasets.MNIST(root='mnist', train=False, download=True, transform=transform)
    idx = (all_train_data.targets == 0) | (all_train_data.targets == 1) | (all_train_data.targets == 2)
    all_train_data.classes = ['0 - zero', '1 - one', '2 - two']
    all_train_data.targets = all_train_data.targets[idx]
    all_train_data.data = all_train_data.data[idx]
    idx_test = (all_test_data.targets == 0) | (all_test_data.targets == 1) | (all_test_data.targets == 2)
    all_test_data.classes = ['0 - zero', '1 - one', '2 - two']
    all_test_data.targets = all_test_data.targets[idx_test]
    all_test_data.data = all_test_data.data[idx_test]
    return all_train_data, all_test_data


def split_data_to_clients(clients):
    train_ds, test_ds = load_3classes_mnist(clients)
    cl0_perclient = int(sum(train_ds.targets == 0) / clients)
    cl1_perclient = int(sum(train_ds.targets == 1) / clients)
    cl2_perclient = int(sum(train_ds.targets == 2) / clients)

    cl0_test_perclient = int(sum(test_ds.targets == 0) / clients)
    cl1_test_perclient = int(sum(test_ds.targets == 1) / clients)
    cl2_test_perclient = int(sum(test_ds.targets == 2) / clients)
    train_ds_split_per_client = []
    test_ds_split_per_client = []
    start_index_zero = 0
    start_index_one = 0
    start_index_two = 0
    test_start_index_zero = 0
    test_start_index_one = 0
    test_start_index_two = 0
    for c in range(clients):
        temp_train_data, temp_test_data = load_3classes_mnist(clients)
        zero_indices = (temp_train_data.targets == 0).nonzero(as_tuple=True)[0][start_index_zero:cl0_perclient]
        one_indices = (temp_train_data.targets == 1).nonzero(as_tuple=True)[0][start_index_one:cl1_perclient]
        two_indices = (temp_train_data.targets == 2).nonzero(as_tuple=True)[0][start_index_two:cl2_perclient]
        start_index_zero = cl0_perclient
        start_index_one = cl1_perclient
        start_index_two = cl2_perclient
        cl0_perclient += cl0_perclient
        cl1_perclient += cl1_perclient
        cl2_perclient += cl2_perclient

        train_ind_per_client = torch.cat((zero_indices, one_indices, two_indices), 0)
        temp_train_data.data = temp_train_data.data[train_ind_per_client]
        temp_train_data.targets = temp_train_data.targets[train_ind_per_client]
        train_ds_split_per_client.append(temp_train_data)

        test_zero_indices = (temp_test_data.targets == 0).nonzero(as_tuple=True)[0][test_start_index_zero:cl0_test_perclient]
        test_one_indices = (temp_test_data.targets == 1).nonzero(as_tuple=True)[0][test_start_index_one:cl1_test_perclient]
        test_two_indices = (temp_test_data.targets == 2).nonzero(as_tuple=True)[0][test_start_index_two:cl2_test_perclient]
        test_start_index_zero = cl0_test_perclient
        test_start_index_one = cl1_test_perclient
        test_start_index_two = cl2_test_perclient
        cl0_test_perclient += cl0_test_perclient
        cl1_test_perclient += cl1_test_perclient
        cl2_test_perclient += cl2_test_perclient

        test_ind_per_client = torch.cat((test_zero_indices, test_one_indices, test_two_indices), 0)
        temp_test_data.data = temp_test_data.data[test_ind_per_client]
        temp_test_data.targets = temp_test_data.targets[test_ind_per_client]
        test_ds_split_per_client.append(temp_test_data)

    return train_ds_split_per_client, test_ds_split_per_client

# def load_test_mnist_data(mode):
#



def read_data(mode):
    # if mode =="mlp":
    # b = 1  # to make the class -1 == 2
    b = -2  # to make the class -1 == 2
    deepak_train_0 = pd.read_csv("RoadData/0/asphalt_x.csv", sep=",", header=0)
    deepak_labels_0 = pd.read_csv("RoadData/0/asphalt_y.csv", sep=",", header=None)
    deepak_train_labels_0 = deepak_labels_0[0].tolist()

    deepak_train_1 = pd.read_csv("RoadData/0/brick_x.csv", sep=",", header=0)
    deepak_labels_1 = pd.read_csv("RoadData/0/brick_y.csv", sep=",", header=None)
    deepak_train_labels_1 = deepak_labels_1[0].tolist()

    deepak_train_2 = pd.read_csv("RoadData/0/bad_x.csv", sep=",", header=0)
    deepak_labels_2 = pd.read_csv("RoadData/0/bad_y.csv", sep=",", header=None)
    deepak_train_labels_2 = deepak_labels_2[0].tolist()

    deepak_test_0 = pd.read_csv("RoadData/0/asphalt_test_x.csv", sep=",", header=0)
    deepak_labels_0 = pd.read_csv("RoadData/0/asphalt_test_y.csv", sep=",", header=None)
    deepak_test_labels_0 = deepak_labels_0[0].tolist()

    deepak_test_1 = pd.read_csv("RoadData/0/brick_test_x.csv", sep=",", header=0)
    deepak_labels_1 = pd.read_csv("RoadData/0/brick_test_y.csv", sep=",", header=None)
    deepak_test_labels_1 = deepak_labels_1[0].tolist()

    deepak_test_2 = pd.read_csv("RoadData/0/bad_test_x.csv", sep=",", header=0)
    deepak_labels_2 = pd.read_csv("RoadData/0/bad_test_y.csv", sep=",", header=None)
    deepak_test_labels_2 = deepak_labels_2[0].tolist()

    khalil_train_0 = pd.read_csv("RoadData/1/asphalt_x.csv", sep=",", header=0)
    khalil_labels_0 = pd.read_csv("RoadData/1/asphalt_y.csv", sep=",", header=None)
    khalil_train_labels_0 = khalil_labels_0[0].tolist()

    khalil_train_1 = pd.read_csv("RoadData/1/brick_x.csv", sep=",", header=0)
    khalil_labels_1 = pd.read_csv("RoadData/1/brick_y.csv", sep=",", header=None)
    khalil_train_labels_1 = khalil_labels_1[0].tolist()

    khalil_train_2 = pd.read_csv("RoadData/1/bad_x.csv", sep=",", header=0)
    khalil_labels_2 = pd.read_csv("RoadData/1/bad_y.csv", sep=",", header=None)
    khalil_train_labels_2 = khalil_labels_2[0].tolist()

    khalil_test_0 = pd.read_csv("RoadData/1/asphalt_test_x.csv", sep=",", header=0)
    khalil_labels_0 = pd.read_csv("RoadData/1/asphalt_test_y.csv", sep=",", header=None)
    khalil_test_labels_0 = khalil_labels_0[0].tolist()

    khalil_test_1 = pd.read_csv("RoadData/1/brick_test_x.csv", sep=",", header=0)
    khalil_labels_1 = pd.read_csv("RoadData/1/brick_test_y.csv", sep=",", header=None)
    khalil_test_labels_1 = khalil_labels_1[0].tolist()

    khalil_test_2 = pd.read_csv("RoadData/1/bad_test_x.csv", sep=",", header=0)
    khalil_labels_2 = pd.read_csv("RoadData/1/bad_test_y.csv", sep=",", header=None)
    khalil_test_labels_2 = khalil_labels_2[0].tolist()

    cl_train_0 = pd.read_csv("RoadData/Centralized/asphalt_x.csv", sep=",", header=0)
    cl_labels_0 = pd.read_csv("RoadData/Centralized/asphalt_y.csv", sep=",", header=None)
    cl_train_labels_0 = cl_labels_0[0].tolist()

    cl_train_1 = pd.read_csv("RoadData/Centralized/brick_x.csv", sep=",", header=0)
    cl_labels_1 = pd.read_csv("RoadData/Centralized/brick_y.csv", sep=",", header=None)
    cl_train_labels_1 = cl_labels_1[0].tolist()

    cl_train_2 = pd.read_csv("RoadData/Centralized/bad_x.csv", sep=",", header=0)
    cl_labels_2 = pd.read_csv("RoadData/Centralized/bad_y.csv", sep=",", header=None)
    cl_train_labels_2 = cl_labels_2[0].tolist()

    cl_test_0 = pd.read_csv("RoadData/Centralized/asphalt_test_x.csv", sep=",", header=0)
    cl_labels_0 = pd.read_csv("RoadData/Centralized/asphalt_test_y.csv", sep=",", header=None)
    cl_test_labels_0 = cl_labels_0[0].tolist()

    cl_test_1 = pd.read_csv("RoadData/Centralized/brick_test_x.csv", sep=",", header=0)
    cl_labels_1 = pd.read_csv("RoadData/Centralized/brick_test_y.csv", sep=",", header=None)
    cl_test_labels_1 = cl_labels_1[0].tolist()

    cl_test_2 = pd.read_csv("RoadData/Centralized/bad_test_x.csv", sep=",", header=0)
    cl_labels_2 = pd.read_csv("RoadData/Centralized/bad_test_y.csv", sep=",", header=None)
    cl_test_labels_2 = b * cl_labels_2[0].tolist()

    xtrain_gl = [np.array(cl_train_0), np.array(cl_train_1), np.array(cl_train_2)]
    ytrain_gl = [np.array(cl_train_labels_0), np.array(cl_train_labels_1), np.array(cl_train_labels_2)]

    xtest_gl = np.vstack((np.array(cl_test_0), np.array(cl_test_1), np.array(cl_test_2)))
    ytest_gl = np.hstack((np.array(cl_test_labels_0), np.array(cl_test_labels_1), np.array(cl_test_labels_2)))

    client1_X = [np.array(deepak_train_0), np.array(deepak_train_1), np.array(deepak_train_2)]
    client1_y = [np.array(deepak_train_labels_0), np.array(deepak_train_labels_1), b * np.array(deepak_train_labels_2)]

    client2_X = [np.array(khalil_train_0), np.array(khalil_train_1), np.array(khalil_train_2)]
    client2_y = [np.array(khalil_train_labels_0), np.array(khalil_train_labels_1), b * np.array(khalil_train_labels_2)]

    xtest1_fl = np.vstack((np.array(deepak_test_0), np.array(deepak_test_1), np.array(deepak_test_2)))
    ytest1_fl = np.hstack(
        (np.array(deepak_test_labels_0), np.array(deepak_test_labels_1), b * np.array(deepak_test_labels_2)))
    # xtest1_fl = np.vstack((np.array(deepak_test_0), np.array(deepak_test_1)))
    # ytest1_fl = np.hstack((np.array(deepak_test_labels_0), np.array(deepak_test_labels_1)))

    xtest2_fl = np.vstack((np.array(khalil_test_0), np.array(khalil_test_1), np.array(khalil_test_2)))
    ytest2_fl = np.hstack(
        (np.array(khalil_test_labels_0), np.array(khalil_test_labels_1), b * np.array(khalil_test_labels_2)))
    # xtest2_fl = np.vstack((np.array(khalil_test_0), np.array(khalil_test_1)))
    # ytest2_fl = np.hstack((np.array(khalil_test_labels_0), np.array(khalil_test_labels_1)))

    if mode == "cl":
        return xtrain_gl, ytrain_gl, xtest_gl, ytest_gl
    elif mode == "fl":
        return client1_X, client1_y, xtest1_fl, ytest1_fl, client2_X, client2_y, xtest2_fl, ytest2_fl
    elif mode == "svm":
        return client1_X, client1_y, client2_X, client2_y, xtest1_fl, ytest1_fl, xtest2_fl, ytest2_fl
    else:
        return xtrain_gl, ytrain_gl, xtest_gl, ytest_gl, client1_X, client1_y, client2_X, client2_y, xtest1_fl, ytest1_fl, xtest2_fl, ytest2_fl


def create_tensors(train_x, train_y, xtest1_fl, ytest1_fl, val_split_percent=20):
    shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y, random_state=0)
    tens_x = torch.tensor(shuffled_train_x)
    tens_y = torch.tensor(shuffled_train_y).type(torch.LongTensor)
    cl_val_split_percentage = val_split_percent
    cl_val_split_index = tens_x.shape[0] - int(0.01 * cl_val_split_percentage * tens_x.shape[0])
    tens_train_x = tens_x[:cl_val_split_index, :]
    tens_train_y = tens_y[:cl_val_split_index]
    tens_val_x = tens_x[cl_val_split_index:, :]
    tens_val_y = tens_y[cl_val_split_index:]
    tens_test_x = torch.tensor(xtest1_fl)
    tens_test_y = torch.tensor(ytest1_fl).type(torch.LongTensor)
    return tens_train_x, tens_train_y, tens_val_x, tens_val_y, tens_test_x, tens_test_y
