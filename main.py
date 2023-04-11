import argparse
import sys
import os

import matplotlib.pyplot as plt
from data import read_client_data, load_test_data, split_data_to_clients, load_3classes_mnist
import numpy as np
import torch
import torch.nn as nn
from model import MLP, CNN
from IMUDataloader import IMUDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support

from old_functions import do_svm, do_mlp, do_multi_svm #ignore


def do_local_models(opt, r):
    models = []
    for c in range(opt.clients):
        if opt.pretrain:
            global_parameters_location = 'runs/startingpoint.pt'
        else:
            global_parameters_location = opt.save_dir + f'global_parameters.pt'
        client_out_dir_name = opt.save_dir + f'{c}/'
        if opt.action =="cl":
            client_out_dir_name = opt.save_dir
            c = "Centralized"
        client_X, client_y, client_val_X, client_val_y, client_test_X, client_test_y = read_client_data(c)
        if opt.model == "mlp":
            model = MLP().double()
        elif opt.model == "cnn":
            model = CNN().double()
        if not opt.action =="cl": model.load_state_dict(torch.load(global_parameters_location))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_ds = IMUDataset(client_X, client_y)
        val_ds = IMUDataset(client_val_X, client_val_y)
        test_ds = IMUDataset(client_test_X, client_test_y)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=opt.batch_size, drop_last=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=opt.batch_size, drop_last=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=opt.batch_size, drop_last=True)
        val_loss_min = np.Inf
        train_losses = []
        valid_losses = []
        for e in range(1, opt.epochs+1):
            train_loss = 0.0
            valid_loss = 0.0
            model.train() #training mode
            for data, target in train_dl:
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            model.eval() #evaluation mode
            for data, target in val_dl:
                out = model(data)
                loss = criterion(out, target)
                valid_loss += loss.item()
            train_loss = train_loss/len(train_dl)
            valid_loss = valid_loss/len(val_dl)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if e % 10 == 0:
                print('Round {} Client{} Epoch {}\tTraining Loss:{:.6f} \tValidation Loss: {:.6f}'.format(r, c, e, train_loss, valid_loss))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f'Training & Validation loss curves for client -{c}')
        fig = plt.plot(train_losses, label="Training")
        fig = plt.plot(valid_losses, label="Validation")
        plt.legend()
        # plt.show()
        plt.savefig(client_out_dir_name+f'LocalClient-{c}-Round-{r}.png')
        plt.clf()
        with torch.no_grad(): #test
            torch.save(model.state_dict(), client_out_dir_name+f'LocalModel-{c}-Round{r}.pt')
        test_loss = test_loop(opt, model, test_dl)


def test_loop(opt, model, test_dl):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    class_correct = list(0. for i in range(opt.classes))
    class_total = list(0. for i in range(opt.classes))
    model.eval()
    # test loop
    test_ds_pred_list = []
    test_ds_target_list = []
    with torch.no_grad():
        for data, target in test_dl:
            out = model(data)
            loss = criterion(out, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(out, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(opt.batch_size):
                label = target.data[i]
                test_ds_target_list.append(label)
                test_ds_pred_list.append(pred[i])
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    test_loss = test_loss / len(test_dl.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(opt.classes):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
        # else:
        #     print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
    f1score = f1_score(test_ds_target_list, test_ds_pred_list, average='weighted')
    print('Weighted F1-Score: {:.6f}\n'.format(f1score))

    return test_loss, f1score


def test_model_updates(opt):
    if opt.mnist:
        aggregated_model = MLP(mnist_flag=opt.mnist)
    elif opt.model == "mlp":
        aggregated_model = MLP(mnist_flag=opt.mnist).double()
    elif opt.model == "cnn":
        aggregated_model = CNN().double()
    aggregated_model.load_state_dict(torch.load(opt.save_dir+f'agg_parameters.pt'))
    for i in range(3):
        if i == 2:
            print(f'\n\t$$$$$$$$$$$$$ TESTING AGGREGATED MODEL on BOTH CLIENTS TEST DATA! $$$$$$$$$$$$$$$$$$$$')
            if opt.mnist:
                _, test_ds = load_3classes_mnist(2)
        else:
            print(f'\n\t$$$$$$$$$$$$$ TESTING AGGREGATED MODEL on CLIENT {i} TEST DATA! $$$$$$$$$$$$$$$$$$$$')
            if opt.mnist:
                _, test_list = split_data_to_clients(2)
                test_ds = test_list[i]
        if not opt.mnist:
            CL_testX, CL_testy = load_test_data(i)
            test_ds = IMUDataset(CL_testX, CL_testy)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=opt.batch_size, drop_last=True)
        test_loss, f1score = test_loop(opt, aggregated_model, test_dl)
    return test_loss, f1score


def do_fl(opt):
    if opt.mnist:
        global_model = MLP(mnist_flag=opt.mnist)
    elif opt.model == "mlp":
        global_model = MLP(mnist_flag=opt.mnist).double()
    elif opt.model == "cnn":
        global_model = CNN().double()
    with torch.no_grad():
        torch.save(global_model.state_dict(), opt.save_dir+f'global_parameters.pt')
    losses_per_round = []
    f1score_per_round = []
    for r in range(opt.round):
        if opt.mnist:
            do_mnist(opt, r)
        else:
            do_local_models(opt, r)
        local_models = []
        if (r == 0) and (not opt.pretrain):
            params_sd = {}
            for key in global_model.state_dict():
                params_sd[key] = torch.zeros(size=global_model.state_dict()[key].shape)
            global_model.load_state_dict(params_sd)
        for c in range(opt.clients):
            client_out_dir_name = opt.save_dir + f'{c}/'
            mname = client_out_dir_name+f'LocalModel-{c}-Round{r}.pt'
            if opt.mnist:
                local_model = MLP(mnist_flag=opt.mnist)
            elif opt.model == "mlp":
                local_model = MLP(mnist_flag=opt.mnist).double()
            elif opt.model == "cnn":
                local_model = CNN().double()
            local_model.load_state_dict(torch.load(mname))
            local_models.append(local_model)
        params_agg = {}
        for key in global_model.state_dict(): #for only 2 clients
            params_agg[key] = (local_models[0].state_dict()[key].data + local_models[1].state_dict()[key].data)/opt.clients
        global_model.load_state_dict(params_agg)
        print(f'########### AGGREGATION STARTS FOR ROUND {r}!! #############')
        if r == 0:
            torch.save(global_model.state_dict(), opt.save_dir + f'global_parameters.pt')
        torch.save(global_model.state_dict(), opt.save_dir+f'agg_parameters.pt')
        test_loss, f1score = test_model_updates(opt)
        losses_per_round.append(test_loss)
        f1score_per_round.append(f1score)
        if r > 0 and f1score > f1score_per_round[r-1]:
            torch.save(global_model.state_dict(), opt.save_dir + f'global_parameters.pt')
        print(f'########### AGGREGATION ENDS FOR ROUND {r}!! #############')
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Testing Loss curve")
    fig = plt.plot(losses_per_round, label="Test Loss")
    # fig = plt.plot(valid_losses, label="Validation")
    plt.legend()
    plt.savefig(opt.save_dir+f'{r}-roundLoss_Curve.png')

def do_mnist(opt, r):
    train_list, test_list = split_data_to_clients(opt.clients)
    for c in range(opt.clients):
        if opt.pretrain:
            global_parameters_location = 'runs/startingpoint.pt'
        else:
            global_parameters_location = opt.save_dir + f'global_parameters.pt'
        client_out_dir_name = opt.save_dir + f'{c}/'
        valid_size = 0.2
        num_train = len(train_list[c])
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders
        train_dl = torch.utils.data.DataLoader(train_list[c], batch_size=opt.batch_size,
                                                   sampler=train_sampler, num_workers=0, drop_last=True)
        val_dl = torch.utils.data.DataLoader(train_list[c], batch_size=opt.batch_size,
                                                   sampler=valid_sampler, num_workers=0, drop_last=True)
        test_dl = torch.utils.data.DataLoader(test_list[c], batch_size=opt.batch_size,
                                                  num_workers=0, drop_last=True)

        mlp = MLP(mnist_flag=opt.mnist)
        mlp.load_state_dict(torch.load(global_parameters_location))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)
        val_loss_min = np.Inf
        train_losses = []
        valid_losses = []
        for e in range(1, opt.epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0
            mlp.train()  # training mode
            for data, target in train_dl:
                optimizer.zero_grad()
                out = mlp(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            mlp.eval()  # evaluation mode
            for data, target in val_dl:
                out = mlp(data)
                loss = criterion(out, target)
                valid_loss += loss.item()
            train_loss = train_loss / len(train_dl)
            valid_loss = valid_loss / len(val_dl)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if e % 5 == 0:
                print('Round {} Client{} Epoch {}\tTraining Loss:{:.6f} \tValidation Loss: {:.6f}'.format(r, c, e,
                                                                                                          train_loss,
                                                                                                          valid_loss))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f'Training & Validation loss curves for client -{c}')
        fig = plt.plot(train_losses, label="Training")
        fig = plt.plot(valid_losses, label="Validation")
        plt.legend()
        # plt.show()
        plt.savefig(client_out_dir_name + f'LocalClient-{c}-Round-{r}.png')
        plt.clf()
        with torch.no_grad():  # test
            torch.save(mlp.state_dict(), client_out_dir_name + f'LocalModel-{c}-Round{r}.pt')
        test_loss, f1score = test_loop(opt, mlp, test_dl)

def do_local_cnn(opt):
    for c in range(opt.clients):
        if opt.pretrain:
            global_parameters_location = 'runs/startingpoint.pt'
        else:
            global_parameters_location = opt.save_dir + f'global_parameters.pt'
        client_out_dir_name = opt.save_dir + f'{c}/'

def init(opt):
    # save_dir = opt.save_dir+f'-{opt.name}/'
    #create folder for each run
    os.mkdir(opt.save_dir)
    if opt.action == "fl":
        for c in range(opt.clients):
            client_dir_name = opt.save_dir+f'{c}/'
            os.mkdir(client_dir_name)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='', help='action to execute: either [mlp, svm, fedavg]')
    parser.add_argument('--round', type=int, default=0, help='# of iterations')
    parser.add_argument('--clients', type=int, default=0, help='# of clients')
    parser.add_argument('--save_dir', type=str, default='runs/test', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--classes', type=int, default=3, help='# of classes')
    parser.add_argument('--epochs', type=int, default=20,
                        help='amount of epochs that will be trained when action is train')
    parser.add_argument('--pretrain', type=bool, default=False, help='')
    parser.add_argument('--model', type=str, default='mlp', help='local model to be used')
    parser.add_argument('--name', type=str, default='model_abc', help='')
    parser.add_argument('--mnist', type=bool, default=False, help='')

    opt = parser.parse_args(sys.argv[1:])
    opt.clients_dir = f'RoadData/'
    opt.save_dir += f'run-{opt.name}-{opt.round}r-{opt.batch_size}bs-{opt.epochs}ep/'
    opt.classes = 3
    if opt.action == "mlp":
        do_mlp(opt)
    elif opt.action == "svm":
        do_svm(opt)
    elif opt.action == "multi_svm":
        do_multi_svm(opt)
    elif opt.action == "fl":
        init(opt)
        do_fl(opt)
    elif opt.action =="cl":
        init(opt)
        opt.clients = 1
        do_local_models(opt, 0)

    elif opt.mnist == True:
        do_mnist(opt)
    # else:
    #     pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
