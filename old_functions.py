import numpy as np
from model import MLP, SVM2, MultiClassSVM
from sklearn.utils import shuffle
from data import read_data, create_tensors, read_client_data, load_test_data
import matplotlib.pyplot as plt

def do_svm(opt):
    opt = "svm"
    client1_X, client1_y, client2_X, client2_y, xtest1_fl, ytest1_fl, xtest2_fl, ytest2_fl = read_data(opt)
    ##### SVM #####
    train_x = np.vstack((client1_X[0], client1_X[1]))
    # train_x = np.hstack([train_x, np.ones((client1_X[0].shape[0]+client1_X[1].shape[0], 1))])
    train_y = np.hstack((client1_y[0], client1_y[1]))
    shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y, random_state=0)

    train2_x = np.vstack((client2_X[0], client2_X[1]))
    # train2_x = np.hstack([train2_x, np.ones((client2_X[0].shape[0]+client2_X[1].shape[0], 1))])
    train2_y = np.hstack((client2_y[0], client2_y[1]))
    shuffled_train2_x, shuffled_train2_y = shuffle(train2_x, train2_y, random_state=0)

    clf = SVM2(learning_rate=1e-3, lambda_param=1e-1, n_iters=1000)
    # clf =
    clf.fit(shuffled_train_x, shuffled_train_y, w_init=False)
    clf_rand_w = SVM2(learning_rate=1e-3, lambda_param=1e-1, n_iters=1000)
    clf_rand_w.fit(shuffled_train_x, shuffled_train_y, w_init=True)
    pred = clf.predict(xtest1_fl)
    pred_rw = clf_rand_w.predict(xtest1_fl)

    # clf2 = SVM2(learning_rate=1e-3, lambda_param=1e-1, n_iters=1000)
    # clf2.fit(shuffled_train2_x, shuffled_train2_y)
    # pred2 = clf2.predict(xtest2_fl)

    print("SVM Model 1 (init_w = 0) Accuracy:\t", clf.accuracy(ytest1_fl, pred))
    print("SVM Model 2 (init_w = random) Accuracy:\t", clf_rand_w.accuracy(ytest1_fl, pred_rw))
    print("\nSVM1 Weights:\n", clf.w)
    print("\nSVM2 Weights:\n", clf_rand_w.w)

    print("\nnp.subtract (SVM1.w , SVM2.w) ==\n", np.subtract(clf.w , clf_rand_w.w))
    print("\nAbs (SVM1.w - SVM2.w) ==\n", np.abs(clf.w - clf_rand_w.w))
    print("\nAbs (SVM2.w - SVM1.w) ==\n", np.abs(clf_rand_w.w - clf.w))

def do_mlp(opt):
    xtrain_gl, ytrain_gl, xtest_gl, ytest_gl, client1_X, client1_y, client2_X, client2_y, xtest1_fl, ytest1_fl, xtest2_fl, ytest2_fl = read_data(
        opt)
    train_x = np.vstack((client1_X[0], client1_X[1], client1_X[2]))
    train_y = np.hstack((client1_y[0], client1_y[1], client1_y[2]))
    train2_x = np.vstack((client2_X[0], client2_X[1], client2_X[2]))
    train2_y = np.hstack((client2_y[0], client2_y[1], client2_y[2]))
    tens_train1_x, tens_train1_y, tens_val1_x, tens_val1_y, tens_test1_x, tens_test1_y = create_tensors(train_x,
                                                                                                        train_y,
                                                                                                        xtest1_fl,
                                                                                                        ytest1_fl)
    tens_train2_x, tens_train2_y, tens_val2_x, tens_val2_y, tens_test2_x, tens_test2_y = create_tensors(train2_x,
                                                                                                        train2_y,
                                                                                                        xtest2_fl,
                                                                                                        ytest2_fl)

    mlp = MLP().double()
    mlp.batch_size = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)


    # shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y, random_state=0)
    #
    # tens_x = torch.tensor(shuffled_train_x)
    # tens_y = torch.tensor(shuffled_train_y).type(torch.LongTensor)
    # cl_val_split_percentage = 20
    # cl_val_split_index = tens_x.shape[0]-int(0.01*cl_val_split_percentage*tens_x.shape[0])
    # tens_train_x = tens_x[:cl_val_split_index, :]
    # tens_train_y = tens_y[:cl_val_split_index]
    # tens_val_x = tens_x[cl_val_split_index:, :]
    # tens_val_y = tens_y[cl_val_split_index:]
    # tens_test_x = torch.tensor(xtest1_fl)
    # tens_test_y = torch.tensor(ytest1_fl).type(torch.LongTensor)

    train_ds = IMUDataset(tens_train1_x, tens_train1_y)
    val_ds = IMUDataset(tens_val1_x, tens_val1_y)
    test_ds = IMUDataset(tens_test1_x, tens_test1_y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=mlp.batch_size, drop_last = True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=mlp.batch_size, drop_last = True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=mlp.batch_size, drop_last = True)

    # val_loss_min = np.Inf
    # train_losses = []
    # valid_losses = []
    #
    # for e in range(1, 201):
    #     train_loss = 0.0
    #     valid_loss = 0.0
    #
    #     mlp.train()
    #     for data, target in train_dl:
    #         optimizer.zero_grad()
    #         out = mlp(data)
    #         loss = criterion(out, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #
    #     mlp.eval()
    #     for data, target in val_dl:
    #         out = mlp(data)
    #         loss = criterion(out, target)
    #         valid_loss += loss.item()
    #     train_loss = train_loss/len(train_dl)
    #     valid_loss = valid_loss/len(val_dl)
    #     train_losses.append(train_loss)
    #     valid_losses.append(valid_loss)
    #
    #     print('Epoch {}\tTraining Loss:{:.6f} \tValidation Loss: {:.6f}'.format(e, train_loss, valid_loss))
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss curve")
    # fig = plt.plot(train_losses, label="Training")
    # fig = plt.plot(valid_losses, label="Validation")
    # plt.legend()
    # plt.savefig("MLP_Clien2_LossCurve_BatchSize40_Epochs200.png")
    # torch.save(mlp.state_dict(), "MLP_lr1e-3_200ep_BS40_client2.pt")
    mlp_client1 = MLP().double()
    mlp_client2 = MLP().double()
    mlp_client2.load_state_dict(torch.load("Results/MLP_lr1e-3_200ep_BS40_client2.pt"))
    mlp_client1.load_state_dict(torch.load("results/MLP_lr1e-3_100ep_BS40_client1.pt"))
    mlp_cl = MLP().double()
    mlp_cl.load_state_dict(mlp_client2.state_dict())
    for key in mlp_client1.state_dict():
        mlp_cl.state_dict()[key] = (mlp_client1.state_dict()[key]+mlp_client2.state_dict()[key]) / 2.
    test_loss = 0.0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    mlp_cl.eval()
    #test loop
    for data, target in test_dl:
        out = mlp_cl(data)
        loss = criterion(out, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(out, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(mlp_cl.batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_dl.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(3):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
        # else:
        #     print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))

def do_multi_svm(opt):
    xtrain_gl, ytrain_gl, xtest_gl, ytest_gl, client1_X, client1_y, client2_X, client2_y, xtest1_fl, ytest1_fl, xtest2_fl, ytest2_fl = read_data(
        opt)
    train_x = np.vstack((client1_X[0], client1_X[1], client1_X[2]))
    train_x = np.hstack([train_x, np.ones((client1_X[0].shape[0]+client1_X[1].shape[0]+client1_X[2].shape[0], 1))])
    train_y = np.hstack((client1_y[0], client1_y[1], client1_y[2]))
    shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y, random_state=0)
    m_svm = MultiClassSVM()
    m_svm.fit(shuffled_train_x, shuffled_train_y)
    losses = m_svm.losses
    plt.plot(losses)
    plt.show()
    x_test1_fl_with1 = np.append(xtest1_fl, np.ones(xtest1_fl.shape[0]).reshape(xtest1_fl.shape[0],1), axis=1)
    pred = m_svm.predict(x_test1_fl_with1)